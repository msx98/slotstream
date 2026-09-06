// SSD expert records for DeepSeek-V4-Flash, streamed from the GGUF on demand.
//
// Each layer carries three MXFP4 tensors — blk.N.ffn_{gate,up,down}_exps with
// ne {4096, 2048, 256} (gate/up: logical [E=256, N=2048, K=4096]) and
// ne {2048, 4096, 256} (down: logical [E, out=4096, in=2048]). The expert is
// the slowest ne axis, so one expert's rows are one contiguous span per
// tensor: rows × (K/32) blocks × 17 B = 4,456,448 B, and a whole expert
// record is 13,369,344 B across the three tensors (computed from the dims +
// GGUFQuant, never hardcoded).
//
// The GGUF interleaves scale and payload ([1 E8M0 scale byte][16 payload
// bytes] per 32 elements, payload nibbles low-first) while the slot pool
// needs them separated for gatherQuantizedMM(.mxfp4): w uint32 [rows, K/8]
// (payload bytes concatenated — uint32 little-endian makes byte order ==
// element order, zero element work) and scales uint8 [rows, K/32] (the E8M0
// bytes in order). Reads deinterleave on the fly through ~256 KiB row-aligned
// staging chunks, one buffer per lane; buffers handed to MLX are 16 KiB
// aligned malloc blocks MLX owns via finalizer. All bytes arrive through
// pread on an F_NOCACHE/F_RDAHEAD-off descriptor — the file is never mmapped
// and never read through the page cache.
//
// readBatch is the decode path (one job per expert × tensor, arbitrary
// order); readRuns is the sweep-staging path (ascending experts, contiguous
// runs collapse into sequential chunked reads over one span). Both mirror
// ExpertStore's contracts: row j of the returned arrays holds keys[j] /
// experts[j], and the caller (engine) holds ModelProcessGuard — this class
// allocates only staging, never the model.

import Foundation
import MLX

public final class DS4ExpertStore {
    /// Pool piece order: gate w, gate scales, up w, up scales, down w,
    /// down scales — the six gathered arrays per expert.
    public static let pieceCount = 6
    /// Staging chunk target (~256 KiB), rounded down to whole rows so a
    /// chunk is always whole 17-byte blocks.
    public static let chunkBytes = 256 * 1024

    /// Read lanes for the sweep's long contiguous runs (`readRuns`). Same
    /// default and env knob as ExpertStore: the runs are throughput-bound,
    /// 4 lanes lose a third, 32 gains nothing (MEASUREMENTS, 2026-08-30).
    public static let defaultQueueDepth: Int = {
        guard let raw = ProcessInfo.processInfo.environment["SLOTSTREAM_IO_QUEUE_DEPTH"],
            let n = Int(raw)
        else { return 12 }
        return min(max(n, 1), 128)
    }()

    /// Read lanes for the pool path (`readBatch`), which is latency-bound:
    /// a layer's handful of misses is six pieces per record. Same default
    /// and env knob as ExpertStore (32).
    public static let poolQueueDepth: Int = {
        guard let raw = ProcessInfo.processInfo.environment["SLOTSTREAM_POOL_QUEUE_DEPTH"],
            let n = Int(raw)
        else { return 32 }
        return min(max(n, 1), 128)
    }()

    public let expertsPerLayer: Int
    public let layerCount: Int
    /// Per-expert 2-D shape of each pool piece: gate/up w [2048, 512] uint32
    /// and scales [2048, 128] uint8 (K = 4096 → K/8, K/32); down w [4096,
    /// 256] uint32 and scales [4096, 64] uint8 (K = 2048).
    public let poolShapes: [(rows: Int, cols: Int)]
    public let poolDtypes: [DType]
    /// Exact bytes of one expert record across the three tensors:
    /// 13,369,344 on the reference checkpoint.
    public let recordBytes: Int
    /// Bytes of each pool piece per expert, in piece order.
    public let pieceBytes: [Int]

    private struct TensorSpec {
        let fileOffset: UInt64  // absolute: dataSectionOffset + dataOffset
        let rows: Int           // per-expert matrix rows (N gate/up, out down)
        let k: Int              // reduction dim
        var blocksPerRow: Int { k / blockSize }
        var rowBytes: Int { blocksPerRow * bytesPerBlock }
        var expertBytes: Int { rows * rowBytes }
        let blockSize: Int
        let bytesPerBlock: Int
    }

    /// [layer][kind 0 = gate, 1 = up, 2 = down]
    private let specs: [[TensorSpec]]
    private let fd: Int32
    deinit { close(fd) }

    public init(ggufPath: String, cfg: DS4Config) throws {
        let gguf = try GGUFFile(path: ggufPath)
        try cfg.validate(gguf: gguf)
        fd = try DS4IO.openFD(ggufPath)

        let E = cfg.embeddingLength
        let ff = cfg.expertFeedForwardLength
        let nExperts = cfg.expertCount
        expertsPerLayer = nExperts
        layerCount = cfg.blockCount
        let blockSize = GGUFQuant.blockSize(.mxfp4)!
        let bytesPerBlock = GGUFQuant.bytesPerBlock(.mxfp4)!

        var specs: [[TensorSpec]] = []
        specs.reserveCapacity(cfg.blockCount)
        for l in 0 ..< cfg.blockCount {
            func spec(_ name: String, _ ne: [Int], rows: Int, k: Int) throws -> TensorSpec {
                let t = try DS4Weights.requireTensor("blk.\(l).\(name)", .mxfp4, ne, gguf)
                return TensorSpec(
                    fileOffset: gguf.dataSectionOffset + t.dataOffset,
                    rows: rows, k: k,
                    blockSize: blockSize, bytesPerBlock: bytesPerBlock)
            }
            // gate/up: ne {K, N, E} → [E, N, K]; down: ne {ff, E, E} → [E, out, in]
            specs.append([
                try spec("ffn_gate_exps.weight", [E, ff, nExperts], rows: ff, k: E),
                try spec("ffn_up_exps.weight", [E, ff, nExperts], rows: ff, k: E),
                try spec("ffn_down_exps.weight", [ff, E, nExperts], rows: E, k: ff),
            ])
        }
        self.specs = specs

        poolShapes = [
            (rows: ff, cols: E / 8), (rows: ff, cols: E / 32),
            (rows: ff, cols: E / 8), (rows: ff, cols: E / 32),
            (rows: E, cols: ff / 8), (rows: E, cols: ff / 32),
        ]
        poolDtypes = [.uint32, .uint8, .uint32, .uint8, .uint32, .uint8]
        pieceBytes = [
            ff * (E / 8) * 4, ff * (E / 32),
            ff * (E / 8) * 4, ff * (E / 32),
            E * (ff / 8) * 4, E * (ff / 32),
        ]
        var total = 0
        for kind in 0 ..< 3 {
            let (next, overflow) = total.addingReportingOverflow(specs[0][kind].expertBytes)
            if overflow { throw ModelError("expert record byte size overflows") }
            total = next
        }
        // Uniform across layers — every layer was validated to the same dims.
        for l in 1 ..< cfg.blockCount {
            for kind in 0 ..< 3 where specs[l][kind].expertBytes != specs[0][kind].expertBytes {
                throw ModelError("blk.\(l) expert tensors differ in size from blk.0 — check --model")
            }
        }
        recordBytes = total
    }

    // MARK: - readBatch (pool path)

    /// Read `keys` into fresh staging arrays, one per piece: piece p has
    /// shape [keys.count, poolShapes[p].rows, poolShapes[p].cols], row j
    /// holding keys[j]. Mirrors ExpertStore.readBatch (the pool scatters with
    /// `pools[p][idx] = batch[p]`, which needs returned arrays — MLXArray is
    /// not writable in place).
    public func readBatch(
        _ keys: [ExpertKey], queueDepth: Int = DS4ExpertStore.poolQueueDepth
    ) throws -> [MLXArray] {
        guard !keys.isEmpty else {
            throw ModelError("DS4ExpertStore.readBatch: empty key list")
        }
        for k in keys { try validate(k) }
        let n = keys.count
        let buffers = try allocStaging(rows: n)
        let box = ErrorBox()
        // One job per (key, tensor): each tensor's expert span is contiguous,
        // and one job fills that key's w + scales pieces for it.
        let jobs: [(slot: Int, kind: Int)] =
            (0 ..< n).flatMap { s in (0 ..< 3).map { (slot: s, kind: $0) } }
        let lanes = min(max(queueDepth, 1), jobs.count)
        DispatchQueue.concurrentPerform(iterations: lanes) { lane in
            let chunk: ChunkBuffer
            do { chunk = try ChunkBuffer() } catch { box.set(error); return }
            var j = lane
            while j < jobs.count {
                if box.error != nil { return }
                let (s, kind) = jobs[j]
                let key = keys[s]
                let wDst = buffers[2 * kind] + s * pieceBytes[2 * kind]
                let sDst = buffers[2 * kind + 1] + s * pieceBytes[2 * kind + 1]
                do {
                    try readDeinterleave(
                        kind: kind, layer: key.layer, expert: key.expert,
                        wDst: wDst, sDst: sDst, chunk: chunk)
                } catch {
                    box.set(error)
                }
                j += lanes
            }
        }
        if let e = box.error {
            buffers.forEach { free($0) }
            throw e
        }
        return stagingArrays(buffers, rows: n)
    }

    // MARK: - readRuns (sweep staging path)

    /// Read one layer's experts into fresh staging arrays (row j holds
    /// experts[j]); ascending runs of expert ids collapse into sequential
    /// chunked reads over one contiguous file span. Mirrors
    /// ExpertStore.readRuns — the sweep passes ascending ids, and the run
    /// detection exploits that; unsorted ids merely produce more runs.
    public func readRuns(
        layer l: Int, experts: [Int], queueDepth: Int = DS4ExpertStore.defaultQueueDepth
    ) throws -> [MLXArray] {
        guard l >= 0 && l < layerCount else {
            throw ModelError("DS4ExpertStore.readRuns: layer \(l) out of range")
        }
        guard !experts.isEmpty else {
            throw ModelError("DS4ExpertStore.readRuns: empty expert list")
        }
        for e in experts { try validate(ExpertKey(l, e)) }
        let n = experts.count
        let buffers = try allocStaging(rows: n)

        var runs: [(row: Int, len: Int)] = []
        var j = 0
        while j < n {
            var k = j + 1
            while k < n, experts[k] == experts[k - 1] + 1 { k += 1 }
            runs.append((j, k - j))
            j = k
        }
        // One job per (run, tensor), largest first so the lanes finish
        // together (ExpertStore.readRuns' discipline).
        var jobs: [(kind: Int, row: Int, len: Int)] = []
        jobs.reserveCapacity(runs.count * 3)
        for r in runs { for kind in 0 ..< 3 { jobs.append((kind, r.row, r.len)) } }
        jobs.sort {
            specs[l][$0.kind].expertBytes * $0.len > specs[l][$1.kind].expertBytes * $1.len
        }

        let lanes = min(max(queueDepth, 1), jobs.count)
        let box = ErrorBox()
        let jobLock = NSLock()
        var next = 0
        DispatchQueue.concurrentPerform(iterations: lanes) { _ in
            let chunk: ChunkBuffer
            do { chunk = try ChunkBuffer() } catch { box.set(error); return }
            while true {
                if box.error != nil { return }
                jobLock.lock()
                let j = next
                next += 1
                jobLock.unlock()
                if j >= jobs.count { return }
                let (kind, row, len) = jobs[j]
                do {
                    // The run is contiguous in the file (expert-major layout):
                    // len × rows sequential blocks, one chunked stream.
                    let spec = specs[l][kind]
                    let base = spec.fileOffset + UInt64(experts[row]) * UInt64(spec.expertBytes)
                    let totalRows = len * spec.rows
                    var done = 0
                    while done < totalRows {
                        let rows = min(chunk.rows(rowBytes: spec.rowBytes), totalRows - done)
                        try DS4IO.preadFully(
                            fd, chunk.buffer,
                            fileOffset: base + UInt64(done * spec.rowBytes),
                            count: rows * spec.rowBytes)
                        deinterleave(
                            chunk.buffer, rows: rows, spec: spec,
                            firstRow: row * spec.rows + done,
                            wDst: buffers[2 * kind],
                            sDst: buffers[2 * kind + 1])
                        done += rows
                    }
                } catch {
                    box.set(error)
                }
            }
        }
        if let e = box.error {
            buffers.forEach { free($0) }
            throw e
        }
        return stagingArrays(buffers, rows: n)
    }

    // MARK: - internals

    private func validate(_ key: ExpertKey) throws {
        guard key.layer >= 0 && key.layer < layerCount else {
            throw ModelError("expert key layer \(key.layer) out of range 0..<\(layerCount)")
        }
        guard key.expert >= 0 && key.expert < expertsPerLayer else {
            throw ModelError(
                "expert key (\(key.layer), \(key.expert)) out of range 0..<\(expertsPerLayer)")
        }
    }

    private func allocStaging(rows n: Int) throws -> [UnsafeMutableRawPointer] {
        var buffers: [UnsafeMutableRawPointer] = []
        buffers.reserveCapacity(Self.pieceCount)
        do {
            for (p, bytes) in pieceBytes.enumerated() {
                buffers.append(try DS4IO.aligned(n * bytes, what: "expert piece \(p)"))
            }
        } catch {
            buffers.forEach { free($0) }
            throw error
        }
        return buffers
    }

    /// Stream one expert's tensor span in row-aligned ~256 KiB chunks,
    /// deinterleaving each chunk into the packed pool pieces.
    private func readDeinterleave(
        kind: Int, layer: Int, expert: Int,
        wDst: UnsafeMutableRawPointer, sDst: UnsafeMutableRawPointer,
        chunk: ChunkBuffer
    ) throws {
        let spec = specs[layer][kind]
        let base = spec.fileOffset + UInt64(expert) * UInt64(spec.expertBytes)
        var row = 0
        while row < spec.rows {
            let rows = min(chunk.rows(rowBytes: spec.rowBytes), spec.rows - row)
            try DS4IO.preadFully(
                fd, chunk.buffer,
                fileOffset: base + UInt64(row * spec.rowBytes),
                count: rows * spec.rowBytes)
            deinterleave(chunk.buffer, rows: rows, spec: spec, firstRow: row, wDst: wDst, sDst: sDst)
            row += rows
        }
    }

    /// Deinterleave `rows` whole rows from `src` (interleaved MXFP4 blocks:
    /// 1 E8M0 scale byte + 16 payload bytes per 32 elements) into the packed
    /// pool layout starting at destination row `firstRow`: w = payload bytes
    /// concatenated (uint32 words are little-endian, so byte order ==
    /// element order — pure 16-byte moves, no element work), scales = the
    /// scale bytes in order.
    private func deinterleave(
        _ src: UnsafeRawPointer, rows: Int, spec: TensorSpec, firstRow: Int,
        wDst: UnsafeMutableRawPointer, sDst: UnsafeMutableRawPointer
    ) {
        let src8 = src.assumingMemoryBound(to: UInt8.self)
        let w8 = wDst.assumingMemoryBound(to: UInt8.self)
        let s8 = sDst.assumingMemoryBound(to: UInt8.self)
        let wRowBytes = spec.k / 2
        for r in 0 ..< rows {
            let row = src8 + r * spec.rowBytes
            let wRow = w8 + (firstRow + r) * wRowBytes
            let sRow = s8 + (firstRow + r) * spec.blocksPerRow
            for b in 0 ..< spec.blocksPerRow {
                let blk = row + b * spec.bytesPerBlock
                sRow[b] = blk[0]
                // 16 payload bytes → 4 uint32 words, low nibble first.
                let dst = UnsafeMutableRawPointer(wRow + b * 16)
                for j in 0 ..< 4 {
                    let o = blk + 1 + 4 * j
                    let word = UInt32(o[0]) | UInt32(o[1]) << 8
                        | UInt32(o[2]) << 16 | UInt32(o[3]) << 24
                    dst.advanced(by: 4 * j).assumingMemoryBound(to: UInt32.self).pointee = word
                }
            }
        }
    }

    /// Hand the six staging buffers to MLX zero-copy (finalizer frees) as
    /// [n, rows, cols] arrays, row j = experts[j]; then materialize.
    private func stagingArrays(_ buffers: [UnsafeMutableRawPointer], rows n: Int) -> [MLXArray] {
        var out: [MLXArray] = []
        out.reserveCapacity(Self.pieceCount)
        for (p, shape) in poolShapes.enumerated() {
            let buf = buffers[p]
            out.append(
                MLXArray(rawPointer: buf, [n, shape.rows, shape.cols], dtype: poolDtypes[p]) {
                    free(buf)
                })
        }
        eval(out)
        return out
    }
}

// MARK: - staging helpers

/// One ~256 KiB read staging buffer per lane, reused across that lane's jobs.
final class ChunkBuffer {
    let buffer: UnsafeMutableRawPointer
    init() throws {
        buffer = try DS4IO.aligned(DS4ExpertStore.chunkBytes, what: "read chunk")
    }
    deinit { free(buffer) }
    /// Whole rows per chunk for a given row byte size (always ≥ 1).
    func rows(rowBytes: Int) -> Int {
        max(1, DS4ExpertStore.chunkBytes / rowBytes)
    }
}

/// Lock-protected first-error capture for concurrentPerform lanes.
final class ErrorBox {
    private let lock = NSLock()
    private var e: Error? = nil
    var error: Error? {
        lock.lock()
        defer { lock.unlock() }
        return e
    }
    func set(_ error: Error) {
        lock.lock()
        if e == nil { e = error }
        lock.unlock()
    }
}
