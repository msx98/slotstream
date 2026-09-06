// DeepSeek-V4-Flash caches and state. Per layer: a raw SWA ring (last
// `swaWindow` F16 rows, position-addressed), append-only compressed KV rows
// (E4M3 payload + per-64 F16 scales + F16 rope dims), append-only indexer rows
// (ratio-4 layers, F32 post Hadamard/FP4), and the compressor recurrent state
// (replaced functionally on every step so checkpoints can hold references).
// Extend-only: rows are appended, never rewound; rollback trims counts.

import Foundation
import MLX

/// Compressor recurrent state for one compressed stream (attention or indexer).
/// `kv`/`score` are `[rows, width]` F32; rows not yet written in the current
/// group hold 0 / -inf so the per-dimension pool ignores them.
public struct DS4CompState {
    var kv: MLXArray?
    var score: MLXArray?

    mutating func ensure(width: Int, rows: Int) {
        if kv == nil {
            kv = MLXArray.zeros([rows, width], dtype: .float32)
            score = MLXArray.full([rows, width], values: MLXArray(-Float.infinity))
        }
    }

    mutating func write(row: Int, kvRow: MLXArray, scoreRow: MLXArray, width: Int) {
        let idx = MLXArray([Int32(row)]).reshaped([1, 1])
        kv = putAlong(kv!, idx, values: kvRow.reshaped([1, width]), axis: 0)
        score = putAlong(score!, idx, values: scoreRow.reshaped([1, width]), axis: 0)
    }

    mutating func rotateRatio4() {
        guard let k = kv, let s = score, k.dim(0) == 8 else { return }
        kv = concatenated([k[4..<8], k[4..<8]], axis: 0)
        score = concatenated([s[4..<8], s[4..<8]], axis: 0)
    }
}

public final class DS4LayerCache {
    public let layer: Int
    public let compressRatio: Int
    public let swaWindow: Int
    public let headDim: Int
    public let rotDim: Int

    // Raw window ring; slot for position p is p % swaWindow.
    public private(set) var rawRing: MLXArray? // [swaWindow, headDim] float16
    // Compressed rows, append-only.
    var compPayload: MLXArray? // [cap, headDim-rotDim] uint8
    var compScales: MLXArray? // [cap, chunks] float16
    var compRope: MLXArray? // [cap, rotDim] float16
    public private(set) var compCount = 0
    // Indexer rows (ratio 4 only), F32 post QAT, append-only.
    var indexRows: MLXArray? // [cap, indexerDim] float32
    public private(set) var indexCount = 0

    var compState = DS4CompState()
    var indexState = DS4CompState()

    private let growStep = 64

    init(layer: Int, cfg: DS4Config) {
        self.layer = layer
        self.compressRatio = cfg.compressRatio(at: layer) ?? 0
        self.swaWindow = cfg.slidingWindow
        self.headDim = cfg.keyLength
        self.rotDim = cfg.ropeDimensionCount
    }

    var nopeDim: Int { headDim - rotDim }
    var scaleChunks: Int { nopeDim / 64 }

    // MARK: raw ring

    public func pushRaw(_ rows: MLXArray, firstPosition: Int) {
        let t = rows.dim(0)
        if rawRing == nil {
            rawRing = MLXArray.zeros([swaWindow, headDim], dtype: .float16)
        }
        var done = 0
        while done < t {
            let n = min(128, t - done)
            var slots: [Int32] = []
            slots.reserveCapacity(n)
            for i in 0..<n { slots.append(Int32((firstPosition + done + i) % swaWindow)) }
            rawRing = putAlong(rawRing!, MLXArray(slots).reshaped([n, 1]),
                               values: rows[done..<(done + n)], axis: 0)
            done += n
        }
    }

    /// Raw rows visible to a query at `position`: positions
    /// `max(0, position+1-swaWindow) ... position`, oldest first.
    public func rawWindow(_ position: Int) -> MLXArray {
        let n = min(position + 1, swaWindow)
        let start = position + 1 - n
        var slots: [Int32] = []
        slots.reserveCapacity(n)
        for i in 0..<n { slots.append(Int32((start + i) % swaWindow)) }
        return rawRing![MLXArray(slots)]
    }

    // MARK: compressed + indexer rows

    func appendCompRow(payload: MLXArray, scales: MLXArray, ropeDims: MLXArray) {
        compPayload = appendRow(compPayload, compCount, row: payload, width: nopeDim, dtype: .uint8)
        compScales = appendRow(compScales, compCount, row: scales, width: scaleChunks, dtype: .float16)
        compRope = appendRow(compRope, compCount, row: ropeDims, width: rotDim, dtype: .float16)
        compCount += 1
    }

    func appendIndexRow(_ row: MLXArray) {
        indexRows = appendRow(indexRows, indexCount, row: row, width: 128, dtype: .float32)
        indexCount += 1
    }

    /// Append one row into a grown-in-blocks buffer; rows past the owning
    /// count are dead and overwritten by later appends.
    private func appendRow(_ buf: MLXArray?, _ count: Int, row: MLXArray, width: Int, dtype: DType) -> MLXArray {
        let r = row.reshaped([1, width]).asType(dtype)
        guard var updated = buf else { return r }
        let cap = updated.dim(0)
        if count < cap {
            updated[count] = r[0]
            return updated
        }
        let newCap = max(growStep, cap + max(1, cap / 2))
        let grown = MLXArray.zeros([newCap, width], dtype: dtype)
        grown[0..<cap] = updated[0..<cap]
        grown[count] = r[0]
        return grown
    }

    /// Dequantized compressed rows `[count, headDim]` F16 (payload · scale for
    /// the non-RoPE dims, stored F16 rope dims appended).
    func dequantCompRows(_ count: Int) -> MLXArray? {
        guard count > 0, let payload = compPayload else { return nil }
        let codes = payload[0..<count].asType(.int32)
        let vals = DS4Math.e4m3LUT[codes] // [count, nopeDim] f32
        let sc = compScales![0..<count].asType(.float32).reshaped([count, scaleChunks, 1])
        let nope = (vals.reshaped([count, scaleChunks, 64]) * sc)
            .reshaped([count, nopeDim]).asType(.float16)
        return concatenated([nope, compRope![0..<count]], axis: -1)
    }

    // MARK: rollback / accounting

    /// Trim the append-only rows to `comp`/`index` counts (speculative rollback).
    func trim(comp: Int, index: Int) {
        compCount = min(compCount, max(0, comp))
        indexCount = min(indexCount, max(0, index))
    }

    /// Fixed per-layer bytes: raw ring + compressor recurrent state.
    public func fixedBytes() -> Int {
        var bytes = swaWindow * headDim * 2
        if compressRatio > 0 {
            let rows = compressRatio == 4 ? 2 * compressRatio : compressRatio
            let width = (compressRatio == 4 ? 2 : 1) * headDim
            bytes += rows * width * 2 * 4 // kv + score, F32
        }
        if compressRatio == 4 {
            bytes += 8 * 256 * 2 * 4 // indexer compressor state
        }
        return bytes
    }

    /// Amortized bytes per consumed token (compressed + indexer rows).
    public func perTokenBytes() -> Double {
        let compRow = nopeDim + scaleChunks * 2 + rotDim * 2
        switch compressRatio {
        case 0: return 0
        case 4: return Double(compRow + 128 * 4) / 4.0
        default: return Double(compRow) / Double(compressRatio)
        }
    }
}

public struct DS4LayerSnapshot {
    var compState: DS4CompState
    var indexState: DS4CompState
    var compCount: Int
    var indexCount: Int
}

/// Zero-copy snapshot of a DS4State for speculative rollback. Compressor
/// states are replaced functionally on every step, so holding references is
/// enough; append-only rows roll back by count (rows past the count are dead
/// and overwritten); the raw ring is position-addressed and needs no restore.
public struct DS4StateCheckpoint {
    let tokenCount: Int
    let layers: [DS4LayerSnapshot]
}

public final class DS4State {
    public private(set) var tokenCount = 0
    public let caches: [DS4LayerCache]

    // Speculative-decode recording: recorded[l][t] is layer l's state after
    // the pass consumed t+1 tokens.
    var recording = false
    var recorded: [[DS4LayerSnapshot]] = []

    public init(cfg: DS4Config) {
        caches = (0..<cfg.blockCount).map { DS4LayerCache(layer: $0, cfg: cfg) }
    }

    public static func makeState(cfg: DS4Config) -> DS4State { DS4State(cfg: cfg) }

    /// Advance the consumed-position counter (called by the model).
    func advance(_ n: Int) { tokenCount += n }

    public func checkpoint() -> DS4StateCheckpoint {
        DS4StateCheckpoint(
            tokenCount: tokenCount,
            layers: caches.map {
                DS4LayerSnapshot(
                    compState: $0.compState, indexState: $0.indexState,
                    compCount: $0.compCount, indexCount: $0.indexCount)
            })
    }

    public func restore(_ c: DS4StateCheckpoint) {
        precondition(c.layers.count == caches.count, "checkpoint layer count mismatch")
        tokenCount = c.tokenCount
        for (l, s) in c.layers.enumerated() {
            caches[l].compState = s.compState
            caches[l].indexState = s.indexState
            caches[l].trim(comp: s.compCount, index: s.indexCount)
        }
    }

    public func setRecording(_ on: Bool) {
        recording = on
        if !on { recorded = [] }
    }

    /// Called by the model after layer `l` consumed token `i` of the pass
    /// (layers run outer, tokens inner, so snapshots are stored [layer][token]).
    func recordLayer(layer l: Int, token i: Int, _ snapshot: DS4LayerSnapshot) {
        guard recording else { return }
        while recorded.count <= l { recorded.append([]) }
        while recorded[l].count <= i { recorded[l].append(snapshot) }
        recorded[l][i] = snapshot
    }

    /// After a recorded pass consumed `ids` from checkpoint `c`, keep only its
    /// first `n` tokens: layer l's compressor states become its recorded state
    /// after position n-1, append-only row counts trim to the recorded counts,
    /// and the position counter rolls back. No model compute.
    public func rollback(keeping n: Int, of ids: [Int], from c: DS4StateCheckpoint) {
        precondition(n >= 1 && n <= ids.count, "rollback: keep \(n) of \(ids.count)")
        if n == ids.count {
            setRecording(false)
            return
        }
        precondition(recorded.count == caches.count,
            "rollback: \(recorded.count) recorded layers for \(caches.count) layers")
        for (l, snaps) in recorded.enumerated() {
            precondition(snaps.count == ids.count,
                "rollback: \(snaps.count) recorded positions for \(ids.count) tokens")
            let s = snaps[n - 1]
            caches[l].compState = s.compState
            caches[l].indexState = s.indexState
            caches[l].trim(comp: s.compCount, index: s.indexCount)
        }
        tokenCount = c.tokenCount + n
        setRecording(false)
    }

    // MARK: memory accounting

    public func cacheMemoryBytes() -> Int {
        var total = 0
        for c in caches {
            total += c.fixedBytes() + Int(c.perTokenBytes() * Double(tokenCount))
        }
        return total
    }

    public var cacheMemoryGB: Double {
        Double(cacheMemoryBytes()) / (1_024.0 * 1_024.0 * 1_024.0)
    }
}
