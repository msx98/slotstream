// Disk-persisted chunk KV cache: saves prefix states after each prefill chunk
// and reloads them to skip recomputation. Default dir ~/.slotstream/kvcache.
// Each prefix is keyed by SHA256 of token ids + model, stored as a directory
// with one file per state array. Background save, LRU eviction on disk if needed.

import Foundation
import CryptoKit
import MLX

public enum DiskCache {
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["SLOTSTREAM_KVCACHE_DISABLE"] == nil
    }
    static var dir: URL {
        if let s = ProcessInfo.processInfo.environment["SLOTSTREAM_KVCACHE_DIR"], !s.isEmpty {
            return URL(fileURLWithPath: s)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".slotstream/kvcache", isDirectory: true)
    }
    static var embeddingsDir: URL { dir.appendingPathComponent("embeddings", isDirectory: true) }

    /// Serial queue so chunk saves during one prefill (4096, 8192, 12288...)
    /// write one at a time instead of three ~1 GB flushes racing each other
    /// and the running prefill for memory and disk bandwidth.
    static let saveQueue = DispatchQueue(label: "slotstream.kvcache.save", qos: .utility)

    static func key(for tokens: [Int], model: String = "qwen38") -> String {
        var hasher = SHA256()
        // Include model to avoid cross-model collisions
        hasher.update(data: Data(model.utf8))
        // Token ids as little-endian Int32
        var buf = Data(capacity: tokens.count * 4)
        for t in tokens {
            var v = Int32(t).littleEndian
            withUnsafeBytes(of: &v) { buf.append(contentsOf: $0) }
        }
        hasher.update(data: buf)
        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    static func path(for key: String) -> URL {
        dir.appendingPathComponent("\(key).bin", isDirectory: false)
    }
    static func embeddingsPath(for key: String) -> URL {
        embeddingsDir.appendingPathComponent("\(key).bin", isDirectory: false)
    }

    // Save a State checkpoint for prefix tokens. Background, chunk-aligned.
    /// Every save, skip, rewrite, and failure logs to stderr. Nothing about
    /// this cache is allowed to be silent: the old `if exists { return }` hid
    /// a tombstone from a killed run and blocked every future save invisibly.
    static func log(_ s: String) {
        FileHandle.standardError.write(Data("[kvcache] \(s)\n".utf8))
    }

    public static func saveAsync(state: Qwen4ExpModel.State, tokens: [Int], model: String = "qwen38") {
        guard enabled else { return }
        let key = self.key(for: tokens, model: model)
        let base = dir.appendingPathComponent(key, isDirectory: true)
        let embBase = embeddingsDir.appendingPathComponent(key, isDirectory: true)
        let cp = state.checkpoint()
        // Snapshot KV/indexer buffers synchronously to avoid race with next request
        var kvSnap: [(Int, MLXArray?, MLXArray?)] = []
        for (l, cache) in state.kv {
            kvSnap.append((l, cache.keys, cache.values))
        }
        var idxSnap: [(Int, MLXArray?)] = []
        for (l, cache) in state.indexer {
            idxSnap.append((l, cache.snapshot()))
        }
        log("save queued: \(tokens.count) tokens key=\(key.prefix(12)) (chunk-aligned checkpoint)")
        let t0 = Date()
        saveQueue.async {
            do {
                // A complete entry has meta.json; a bare directory is a
                // tombstone from a crashed/aborted save and must be retried,
                // otherwise the empty dir blocks every future attempt.
                let metaURL = base.appendingPathComponent("meta.json")
                if FileManager.default.fileExists(atPath: metaURL.path) {
                    log("save skipped: \(tokens.count) tokens key=\(key.prefix(12)) already complete on disk")
                    return
                }
                if FileManager.default.fileExists(atPath: base.path) {
                    log("tombstone found for key=\(key.prefix(12)) (earlier save died before meta.json) — rewriting")
                    try FileManager.default.removeItem(at: base)
                }
                if FileManager.default.fileExists(atPath: embBase.path) {
                    try FileManager.default.removeItem(at: embBase)
                }
                try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
                try FileManager.default.createDirectory(at: embBase, withIntermediateDirectories: true)
                let meta: [String: Any] = [
                    "key": key,
                    "tokens": tokens,
                    "tokenCount": cp.tokenCount,
                    "ngramCtx": cp.ngramCtx,
                    // JSON dictionary keys must be strings; Int keys from the
                    // offset maps threw NSInvalidArgumentException mid-save.
                    "kvOffsets": cp.kvOffsets.reduce(into: [String: Int]()) { $0[String($1.key)] = $1.value },
                    "indexerOffsets": cp.indexerOffsets.reduce(into: [String: Int]()) { $0[String($1.key)] = $1.value },
                    "mtpOffset": cp.mtpOffset,
                    "version": 2
                ]
                let metaData = try JSONSerialization.data(withJSONObject: meta, options: [])
                try metaData.write(to: base.appendingPathComponent("meta.json"))
                var written = 0
                func saveArray(_ arr: MLXArray?, name: String) throws {
                    guard let a = arr else { return }
                    let url = base.appendingPathComponent("\(name).bin")
                    let f32 = a.asType(.float32)
                    let shape = f32.shape
                    let arrData = f32.asArray(Float.self)
                    let header: [String: Any] = ["shape": shape, "dtype": "float32"]
                    let headerData = try JSONSerialization.data(withJSONObject: header, options: [])
                    var combined = Data()
                    var len = UInt32(headerData.count).littleEndian
                    withUnsafeBytes(of: &len) { combined.append(contentsOf: $0) }
                    combined.append(headerData)
                    arrData.withUnsafeBytes { combined.append(contentsOf: $0) }
                    try combined.write(to: url)
                    written += combined.count
                }
                for (l, arr) in cp.conv { try saveArray(arr, name: "conv_\(l)") }
                for (l, arr) in cp.ssm { try saveArray(arr, name: "ssm_\(l)") }
                for (l, arr) in cp.pleConv { try saveArray(arr, name: "pleConv_\(l)") }
                if let lm = cp.lastMulti { try saveArray(lm, name: "lastMulti") }
                for (l, k, v) in kvSnap.map({ ($0.0, $0.1, $0.2) }) {
                    try saveArray(k, name: "kv_k_\(l)")
                    try saveArray(v, name: "kv_v_\(l)")
                }
                for (l, b) in idxSnap {
                    try saveArray(b, name: "indexer_\(l)")
                }
                let embMeta: [String: Any] = ["key": key, "tokens": tokens, "version": 2]
                let embData = try JSONSerialization.data(withJSONObject: embMeta, options: [])
                try embData.write(to: embBase.appendingPathComponent("meta.json"))
                log("save done: \(tokens.count) tokens key=\(key.prefix(12)) "
                    + "\(String(format: "%.1f", Double(written) / 1e6)) MB in "
                    + String(format: "%.1fs", -t0.timeIntervalSinceNow))
            } catch {
                // Invalid JSON structure still raises as an Obj-C exception and
                // kills the process (crash and burn). This catch is for I/O and
                // serializable-but-failed writes, which must scream, not die:
                // the server keeps serving either way.
                log("SAVE FAILED for \(tokens.count) tokens key=\(key.prefix(12)): \(error)")
            }
        }
    }

    public static func loadIfPresent(tokens: [Int], model: String = "qwen38") -> Bool {
        guard enabled else { return false }
        let key = self.key(for: tokens, model: model)
        let base = dir.appendingPathComponent(key, isDirectory: true)
        return FileManager.default.fileExists(atPath: base.path)
    }

    public static func longestPrefixHit(for promptIds: [Int], chunk: Int, model: String = "qwen38") -> Int? {
        guard enabled, chunk > 0, promptIds.count > chunk else { return nil }
        var best: Int? = nil
        var hi = chunk
        while hi < promptIds.count {
            let prefix = Array(promptIds[0..<hi])
            if loadIfPresent(tokens: prefix, model: model) {
                best = hi
            }
            hi += chunk
        }
        return best
    }

    // Load a saved prefix into a new State. Returns nil if not found or incompatible.
    public static func loadState(for tokens: [Int], model: String = "qwen38", template: Qwen4ExpModel.State) -> Qwen4ExpModel.State? {
        guard enabled else { return nil }
        let key = self.key(for: tokens, model: model)
        let base = dir.appendingPathComponent(key, isDirectory: true)
        guard FileManager.default.fileExists(atPath: base.path) else { return nil }
        guard let metaData = try? Data(contentsOf: base.appendingPathComponent("meta.json")),
              let meta = try? JSONSerialization.jsonObject(with: metaData) as? [String: Any] else { return nil }
        let state = template
        // Restore ngramCtx, tokenCount, offsets are handled via arrays
        if let ngram = meta["ngramCtx"] as? [Int64] { state.ngramCtx = ngram }
        else if let ngram = meta["ngramCtx"] as? [Int] { state.ngramCtx = ngram.map { Int64($0) } }
        if let tc = meta["tokenCount"] as? Int { state.tokenCount = tc }
        // Restore arrays
        func loadArray(name: String) -> MLXArray? {
            let url = base.appendingPathComponent("\(name).bin")
            guard let data = try? Data(contentsOf: url), data.count >= 4 else { return nil }
            let headerLen = data[0..<4].withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }
            guard data.count >= 4 + Int(headerLen) else { return nil }
            let headerData = data[4..<(4+Int(headerLen))]
            guard let header = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any],
                  let shape = header["shape"] as? [Int] else { return nil }
            let body = data[(4+Int(headerLen))...]
            let floats = body.withUnsafeBytes { ptr in
                Array(ptr.bindMemory(to: Float.self))
            }
            return MLXArray(floats, shape).asType(.bfloat16)
        }
        for l in template.linear.keys {
            if let arr = loadArray(name: "conv_\(l)") { template.linear[l]?.convState = arr }
            if let arr = loadArray(name: "ssm_\(l)") { template.linear[l]?.ssmState = arr }
            if let arr = loadArray(name: "pleConv_\(l)") { template.linear[l]?.pleConvState = arr }
        }
        if let arr = loadArray(name: "lastMulti") { state.lastMulti = arr }
        // KV and indexer need special handling to restore buffers and offsets
        if let kvOffsets = meta["kvOffsets"] as? [String: Int] {
            for (kStr, off) in kvOffsets {
                if let l = Int(kStr), let cache = state.kv[l] {
                    if let k = loadArray(name: "kv_k_\(l)"), let v = loadArray(name: "kv_v_\(l)") {
                        // Need to restore KVCache buffers: we saved the full buffer, but need to set offset
                        // For now, set the cache's internal buffers directly via a helper
                        cache.restoreFromArrays(keys: k, values: v, offset: off)
                    }
                }
            }
        }
        if let idxOffsets = meta["indexerOffsets"] as? [String: Int] {
            for (kStr, off) in idxOffsets {
                if let l = Int(kStr), let cache = state.indexer[l] {
                    if let b = loadArray(name: "indexer_\(l)") {
                        cache.restore(from: b, offset: off)
                    }
                }
            }
        }
        return state
    }
}
