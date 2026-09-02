// Disk-persisted chunk KV cache: saves prefix states after each prefill chunk
// and reloads them to skip recomputation. Layout:
//   ~/.slotstream/kvcache/
//     metadata.db                 SQLite index (parent chains, last_used, sizes)
//     <key>/
//       data.kv                   saved state arrays (one .kv.partial is
//                                 rewritten atomically; final name is data.kv)
//       parent_sha.bin            parent chunk's key (32 bytes hex), empty at depth 0
//       emb_sha.bin               sha256 of the chunk's embedding rows (32 bytes hex)
// Key derivation lives in ChunkIndex.makeKey and binds chunks to their
// ancestors: sha256(parent_sha || sha256(chunk_embeddings)). Two conversations
// that share the first 4096 tokens but diverge at position 4097 get two
// distinct keys at depth 1, so a content match can never produce a wrong
// KVCache (the linear-attention / MTP recurrent states can't be rewound).

import Foundation
import CryptoKit
import MLX

public enum DiskCache {
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["SLOTSTREAM_KVCACHE_DISABLE"] == nil
    }
    /// Explicit process-global override. Set by the CLI's --kv-store-dir;
    /// takes precedence over the env var so a single command-line invocation
    /// is hermetic. Reads from `DiskCache.dir` everywhere, so it just has
    /// to be set before any chunk is touched.
    public static var dirOverride: String?
    static var dir: URL {
        if let o = dirOverride, !o.isEmpty {
            return URL(fileURLWithPath: o)
        }
        if let s = ProcessInfo.processInfo.environment["SLOTSTREAM_KVCACHE_DIR"], !s.isEmpty {
            return URL(fileURLWithPath: s)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".slotstream/kvcache", isDirectory: true)
    }
    /// Default on-disk quota. Old entries beyond this are evicted leaf-first.
    static var maxBytes: Int {
        let envGB = ProcessInfo.processInfo.environment["SLOTSTREAM_KVCACHE_MAX_GB"]
            .flatMap { Double($0) } ?? 20.0
        return Int(envGB * 1_073_741_824.0)
    }

    /// Serial queue so chunk saves during one prefill (4096, 8192, 12288...)
    /// write one at a time instead of three ~1 GB flushes racing each other
    /// and the running prefill for memory and disk bandwidth.
    static let saveQueue = DispatchQueue(label: "slotstream.kvcache.save", qos: .utility)

    /// Block until every queued save has finished. Safe to call from the
    /// main thread; never call from inside a save callback.
    public static func flush() {
        saveQueue.sync { }
    }

    static func log(_ s: String) {
        FileHandle.standardError.write(Data("[kvcache] \(s)\n".utf8))
    }

    // MARK: - key derivation (delegates to ChunkIndex; defined here for legacy callers)

    /// Legacy key used by callers that don't yet have a parent chain. Only
    /// depth-0 saves call this; deeper depths go through ChunkIndex.makeKey.
    static func key(parentSha: String?, embeddingSha: String) -> String {
        ChunkIndex.makeKey(parentSha: parentSha, embeddings: Array(decodingHex(embeddingSha).withUnsafeBytes {
            $0.bindMemory(to: Float.self)
        }))
    }

    private static func decodingHex(_ s: String) -> Data {
        var data = Data(capacity: s.count / 2)
        var idx = s.startIndex
        while idx < s.endIndex {
            let next = s.index(idx, offsetBy: 2, limitedBy: s.endIndex) ?? s.endIndex
            if let b = UInt8(s[idx..<next], radix: 16) { data.append(b) }
            idx = next
        }
        return data
    }

    /// Compute sha256 of a row-major embedding slice.
    public static func embeddingSha(embeddings: [Float]) -> String {
        var hasher = SHA256()
        var buf = Data(capacity: embeddings.count * 4)
        for v in embeddings {
            var f = v.bitPattern.littleEndian
            withUnsafeBytes(of: &f) { buf.append(contentsOf: $0) }
        }
        hasher.update(data: buf)
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - save

    /// Save the chunk that ends at `tokens` (a multiple of `chunk`). The
    /// key is computed synchronously by the caller so the next chunk's
    /// parent_sha can be derived without waiting for the IO. The actual
    /// write runs on the serial save queue; if the same key is queued twice
    /// (e.g. two requests racing on the same prefix), the second one no-ops.
    /// `parentSha` is the previous chunk's key, or nil for depth 0.
    public static func saveAsync(
        state: Qwen4ExpModel.State,
        tokenIds: [Int],
        key: String,
        parentSha: String?,
        depth: Int,
        embeddings: [Float],
        model: String = "qwen38"
    ) {
        guard enabled else { return }
        let embSha = embeddingSha(embeddings: embeddings)
        let base = dir.appendingPathComponent(key, isDirectory: true)
        let cp = state.checkpoint()
        log("save queued: \(tokenIds.count) tokens depth=\(depth) key=\(key.prefix(12))")
        let t0 = Date()
        saveQueue.async {
            do {
                // Already complete: skip. The check happens after queue dispatch
                // so concurrent saves of the same key serialize on the queue.
                let dataPath = base.appendingPathComponent("data.kv")
                if FileManager.default.fileExists(atPath: dataPath.path) {
                    log("save skipped: \(tokenIds.count) tokens depth=\(depth) key=\(key.prefix(12)) already complete on disk")
                    ChunkIndex.shared.touch(key: key)
                    return
                }
                try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
                let partialPath = base.appendingPathComponent("data.kv.partial")
                let parentPath = base.appendingPathComponent("parent_sha.bin")
                let embPath = base.appendingPathComponent("emb_sha.bin")

                // Crash-safe write: write to .partial, rename to data.kv on
                // completion. POSIX rename on the same FS is atomic, so a
                // crash mid-write leaves either the old data.kv or the
                // .partial — never a half-written data.kv.
                if FileManager.default.fileExists(atPath: partialPath.path) {
                    try? FileManager.default.removeItem(at: partialPath)
                }
                let fm = FileManager.default
                fm.createFile(atPath: partialPath.path, contents: nil)
                let outHandle = try FileHandle(forWritingTo: partialPath)

                // Header line is one small JSON with the offsets the loader
                // needs to restore the state without reading every .bin
                // separately. Array blobs follow in a length-prefixed
                // <shape,json,bytes> format identical to the old layout.
                let meta: [String: Any] = [
                    "tokenCount": cp.tokenCount,
                    "ngramCtx": cp.ngramCtx,
                    "kvOffsets": cp.kvOffsets.reduce(into: [String: Int]()) { $0[String($1.key)] = $1.value },
                    "indexerOffsets": cp.indexerOffsets.reduce(into: [String: Int]()) { $0[String($1.key)] = $1.value },
                    "mtpOffset": cp.mtpOffset,
                    "version": 3
                ]
                let metaData = try JSONSerialization.data(withJSONObject: meta, options: [])
                try outHandle.write(contentsOf: metaData)
                try outHandle.write(contentsOf: Data("\n".utf8))

                func writeArray(_ arr: MLXArray?, name: String) throws {
                    guard let a = arr else { return }
                    let f32 = a.asType(.float32)
                    let shape = f32.shape
                    let arrData = f32.asArray(Float.self)
                    let header: [String: Any] = ["name": name, "shape": shape, "dtype": "float32"]
                    let headerData = try JSONSerialization.data(withJSONObject: header, options: [])
                    var len = UInt32(headerData.count).littleEndian
                    var combined = Data()
                    withUnsafeBytes(of: &len) { combined.append(contentsOf: $0) }
                    combined.append(headerData)
                    arrData.withUnsafeBytes { combined.append(contentsOf: $0) }
                    try outHandle.write(contentsOf: combined)
                }

                for (l, arr) in cp.conv { try writeArray(arr, name: "conv_\(l)") }
                for (l, arr) in cp.ssm { try writeArray(arr, name: "ssm_\(l)") }
                for (l, arr) in cp.pleConv { try writeArray(arr, name: "pleConv_\(l)") }
                if let lm = cp.lastMulti { try writeArray(lm, name: "lastMulti") }
                // KV/indexer need live snapshots — checkpoint already pulled the
                // buffers into cp.mtpKV / cp.mtpIndexer but the main caches
                // have to be snapshotted here.
                var kvSnap: [(Int, MLXArray?, MLXArray?)] = []
                for (l, cache) in state.kv { kvSnap.append((l, cache.keys, cache.values)) }
                for (l, k, v) in kvSnap {
                    try writeArray(k, name: "kv_k_\(l)")
                    try writeArray(v, name: "kv_v_\(l)")
                }
                var idxSnap: [(Int, MLXArray?)] = []
                for (l, cache) in state.indexer { idxSnap.append((l, cache.snapshot())) }
                for (l, b) in idxSnap { try writeArray(b, name: "indexer_\(l)") }
                if let (k, v) = cp.mtpKV {
                    try writeArray(k, name: "mtp_kv_k")
                    try writeArray(v, name: "mtp_kv_v")
                }
                if let b = cp.mtpIndexer { try writeArray(b, name: "mtp_indexer") }

                try outHandle.close()

                // Atomic publish.
                if fm.fileExists(atPath: dataPath.path) { try fm.removeItem(at: dataPath) }
                try fm.moveItem(at: partialPath, to: dataPath)

                // parent_sha and emb_sha are 32-byte binary, hex-encoded for
                // cheap concatenation in makeKey.
                let parentBytes = parentSha?.data(using: .utf8) ?? Data()
                try parentBytes.write(to: parentPath)
                try Data(embSha.utf8).write(to: embPath)

                // Register in the index only after the file is published and
                // committed. A crash before this point leaves no row, which
                // is harmless: the next lookup won't find the chunk and will
                // rebuild.
                let dirSize = directorySize(at: base)
                ChunkIndex.shared.register(
                    key: key, parentSha: parentSha, depth: depth, sizeBytes: dirSize)

                // Eviction: if over quota, drop leaves until under quota.
                let total = ChunkIndex.shared.totalBytes()
                if total > maxBytes {
                    let freed = ChunkIndex.shared.evictLeaves(
                        bytesNeeded: total - maxBytes, maxBytes: maxBytes)
                    if freed > 0 {
                        // Sweep directories the index no longer knows about
                        // (orphaned from earlier crashes or manual deletes).
                        sweepOrphans()
                        log("evicted \(freed) bytes, now \(ChunkIndex.shared.totalBytes()) bytes on disk")
                    }
                }

                log("save done: \(tokenIds.count) tokens depth=\(depth) key=\(key.prefix(12)) "
                    + "\(String(format: "%.1f", Double(dirSize) / 1e6)) MB in "
                    + String(format: "%.1fs", -t0.timeIntervalSinceNow))
            } catch {
                log("SAVE FAILED for \(tokenIds.count) tokens depth=\(depth): \(error)")
                try? FileManager.default.removeItem(
                    at: base.appendingPathComponent("data.kv.partial"))
            }
        }
    }

    // MARK: - lookup

    /// Walk the chain depth-by-depth. For each depth i, the caller supplies
    /// the embeddings of the chunk at `[i*chunk, (i+1)*chunk)` via `embed`.
    /// We compute the per-chunk key and stop at the first miss. Returns the
    /// longest prefix length whose chain is fully present, or nil.
    public static func longestPrefixHit(
        chunk: Int, model: String = "qwen38",
        embed: (Int) -> [Float]?
    ) -> Int? {
        guard enabled, chunk > 0 else { return nil }
        let depth = ChunkIndex.shared.chainLength(chunk: chunk, embedForDepth: embed) ?? 0
        return depth == 0 ? nil : depth * chunk
    }

    /// Returns the parent_sha for the chunk at `depth`, or nil for depth 0.
    /// Used by callers reconstructing the chain after a hit to populate
    /// `loadState` correctly.
    public static func parentShaFor(depth: Int, embed: (Int) -> [Float]?) -> String? {
        guard depth > 0 else { return nil }
        var prev: String? = nil
        for i in 0..<(depth - 1) {
            guard let e = embed(i) else { return nil }
            prev = ChunkIndex.makeKey(parentSha: prev, embeddings: e)
        }
        return prev
    }

    // MARK: - load

    /// Load a saved chunk into a fresh State. `depth` is the chain depth
    /// (0 = first chunk). The caller already verified the chain by walking
    /// it; this just decodes the on-disk file at the resolved key.
    public static func loadState(
        for embed: (Int) -> [Float]?,
        depth: Int,
        tokenIds: [Int],
        template: Qwen4ExpModel.State,
        model: String = "qwen38"
    ) -> Qwen4ExpModel.State? {
        guard enabled, depth >= 0 else { return nil }
        guard let chunkEmb = embed(depth - 1 >= 0 ? depth - 1 : 0) else { return nil }
        // Resolve the parent's chain key: walk depths 0..<(depth-1), composing
        // each step from the per-chunk embeddings. For depth=0 the parent is
        // nil (no chunks consumed); for depth=N it's chain[N-1] from walking
        // 0..<(N-1).
        var parentSha: String? = nil
        if depth >= 1 {
            for i in 0..<(depth - 1) {
                guard let e = embed(i) else { return nil }
                parentSha = ChunkIndex.makeKey(parentSha: parentSha, embeddings: e)
            }
        }
        let key = ChunkIndex.makeKey(parentSha: parentSha, embeddings: chunkEmb)
        let base = dir.appendingPathComponent(key, isDirectory: true)
        let dataPath = base.appendingPathComponent("data.kv")
        guard FileManager.default.fileExists(atPath: dataPath.path) else {
            // Index says exists but file is gone — clean up.
            ChunkIndex.shared.remove(key: key)
            return nil
        }
        guard let data = try? Data(contentsOf: dataPath) else {
            log("load: data.kv unreadable for key=\(key.prefix(12))")
            return nil
        }
        guard let nlIndex = data.firstIndex(of: 0x0a) else {
            log("load: no header newline in data.kv for key=\(key.prefix(12)) (size \(data.count))")
            return nil
        }
        let metaData = data[0..<nlIndex]
        guard let meta = try? JSONSerialization.jsonObject(with: metaData) as? [String: Any]
        else {
            log("load: bad meta header JSON for key=\(key.prefix(12))")
            return nil
        }
        if (meta["version"] as? Int) != 3 {
            log("load: bad version \(meta["version"] ?? -1) for key=\(key.prefix(12))")
            return nil
        }
        var body = data[(nlIndex + 1)...]

        // Restore scalar fields.
        let state = template
        if let ngram = meta["ngramCtx"] as? [Int64] { state.ngramCtx = ngram }
        else if let ngram = meta["ngramCtx"] as? [Int] { state.ngramCtx = ngram.map { Int64($0) } }
        if let tc = meta["tokenCount"] as? Int { state.tokenCount = tc }

        // Read the whole body once into a name -> MLXArray map. Doing it in
        // a single pass avoids name-order brittleness across the layout.
        var arraysByName: [String: MLXArray] = [:]
        while body.count >= 4 {
            let len = body[body.startIndex..<body.startIndex+4]
                .withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }
            let hEnd = body.startIndex + 4 + Int(len)
            guard body.count >= hEnd else { break }
            let headerData = body[(body.startIndex + 4)..<hEnd]
            guard let header = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any],
                  let name = header["name"] as? String,
                  let shape = header["shape"] as? [Int]
            else { break }
            let floatsBytes = body.count - hEnd
            guard floatsBytes % 4 == 0 else { break }
            let raw = body[hEnd..<(hEnd + floatsBytes)]
            body = body[(hEnd + floatsBytes)...]
            let floats = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            arraysByName[name] = MLXArray(floats, shape).asType(.bfloat16)
        }

        for l in template.linear.keys {
            if let a = arraysByName["conv_\(l)"] { template.linear[l]?.convState = a }
            if let a = arraysByName["ssm_\(l)"] { template.linear[l]?.ssmState = a }
            if let a = arraysByName["pleConv_\(l)"] { template.linear[l]?.pleConvState = a }
        }
        if let a = arraysByName["lastMulti"] { state.lastMulti = a }

        if let kvOffsets = meta["kvOffsets"] as? [String: Int] {
            for (kStr, off) in kvOffsets {
                guard let l = Int(kStr), let cache = state.kv[l],
                      let k = arraysByName["kv_k_\(l)"],
                      let v = arraysByName["kv_v_\(l)"]
                else { continue }
                cache.restoreFromArrays(keys: k, values: v, offset: off)
            }
        }
        if let idxOffsets = meta["indexerOffsets"] as? [String: Int] {
            for (kStr, off) in idxOffsets {
                guard let l = Int(kStr), let cache = state.indexer[l],
                      let b = arraysByName["indexer_\(l)"]
                else { continue }
                cache.restore(from: b, offset: off)
            }
        }
        if let mtpOff = meta["mtpOffset"] as? Int, mtpOff > 0 {
            // Stale entry without MTP arrays would crash the next consume();
            // bail so the caller rebuilds from scratch.
            guard arraysByName["mtp_kv_k"] != nil, arraysByName["mtp_kv_v"] != nil
            else { return nil }
        }
        if let mtpOff = meta["mtpOffset"] as? Int,
           let k = arraysByName["mtp_kv_k"], let v = arraysByName["mtp_kv_v"]
        {
            if state.mtp == nil { state.mtp = MTPState() }
            state.mtp!.kv.restoreFromArrays(keys: k, values: v, offset: mtpOff)
            if let b = arraysByName["mtp_indexer"] {
                state.mtp!.indexer.restore(from: b, offset: mtpOff)
            }
        }

        ChunkIndex.shared.touch(key: key)
        return state
    }

    // MARK: - utilities

    private static func directorySize(at url: URL) -> Int {
        let fm = FileManager.default
        guard let it = fm.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey])
        else { return 0 }
        var total = 0
        for case let f as URL in it {
            let vals = try? f.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if vals?.isRegularFile == true { total += vals?.fileSize ?? 0 }
        }
        return total
    }

    /// Remove chunk directories whose key is no longer in the index. Called
    /// after eviction so a stale process can't be tripped up by the disk
    /// outlasting the DB.
    private static func sweepOrphans() {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil) else { return }
        for entry in entries {
            let dataPath = entry.appendingPathComponent("data.kv")
            if !fm.fileExists(atPath: dataPath.path) {
                try? fm.removeItem(at: entry)
            }
        }
    }
}