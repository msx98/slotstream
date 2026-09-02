// Streams routed-expert records from the original checkpoint shards into the
// slot pool. One expert = 9 tensor pieces (gate/up/down × weight/scales/biases),
// each contiguous per expert inside its [512, R, C] shard tensor (verified M0).

import Foundation
import MLX

public struct ExpertKey: Hashable {
    public let layer: Int
    public let expert: Int
    public init(_ l: Int, _ e: Int) {
        layer = l
        expert = e
    }
}

private final class StagedExpertRecord {
    fileprivate var pieces: [UnsafeMutableRawPointer?]

    init(pieceBytes: [Int]) {
        pieces = []
        for bytes in pieceBytes {
            var pointer: UnsafeMutableRawPointer? = nil
            guard posix_memalign(&pointer, 16384, bytes) == 0, let pointer else {
                for case let allocated? in pieces { free(allocated) }
                fatalError("out of memory staging prefetched expert record")
            }
            pieces.append(pointer)
        }
    }

    deinit {
        for case let pointer? in pieces { free(pointer) }
    }
}

public final class ExpertStore {
    public let index: CheckpointIndex
    private let cfg: ModelConfig
    // per (layer, piece) tensor refs; pieces ordered gw,gs,gb,uw,us,ub,dw,ds,db
    static let pieces = [
        "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
        "up_proj.weight", "up_proj.scales", "up_proj.biases",
        "down_proj.weight", "down_proj.scales", "down_proj.biases",
    ]
    private var refs: [[TensorRef]] = []  // [layer][piece]
    public private(set) var pieceRowBytes: [Int] = []  // bytes per expert per piece
    public var recordBytes: Int {
        var total = 0
        for bytes in pieceRowBytes {
            let (next, overflow) = total.addingReportingOverflow(bytes)
            if overflow { return 0 }
            total = next
        }
        return total
    }

    public init(index: CheckpointIndex) throws {
        self.index = index
        self.cfg = index.config
        let h = cfg.hiddenSize
        let ff = cfg.moeIntermediate
        let g = cfg.qGroup
        let expected: [(shape: [Int], dtype: String)] = [
            ([cfg.numExperts, ff, h / 8], "U32"),
            ([cfg.numExperts, ff, h / g], "BF16"),
            ([cfg.numExperts, ff, h / g], "BF16"),
            ([cfg.numExperts, ff, h / 8], "U32"),
            ([cfg.numExperts, ff, h / g], "BF16"),
            ([cfg.numExperts, ff, h / g], "BF16"),
            ([cfg.numExperts, h, ff / 8], "U32"),
            ([cfg.numExperts, h, ff / g], "BF16"),
            ([cfg.numExperts, h, ff / g], "BF16"),
        ]
        for l in 0 ..< cfg.numLayers {
            let base = "model.layers.\(l).mlp.switch_mlp."
            let layer = Self.pieces.map { index.ref(base + $0) }
            for p in layer.indices {
                guard layer[p].shape == expected[p].shape,
                    layer[p].dtype == expected[p].dtype,
                    layer[p].rowBytes > 0
                else {
                    throw ModelError(
                        "tensor `\(base + Self.pieces[p])` has \(layer[p].dtype) "
                            + "\(layer[p].shape), expected \(expected[p].dtype) "
                            + "\(expected[p].shape) — check --model")
                }
            }
            refs.append(layer)
        }
        pieceRowBytes = refs[0].map { $0.rowBytes }
    }

    /// Read a batch of experts (QD-parallel pread) and return the 9 stacked
    /// MLXArrays shaped [n, ...piece shape...] ready to scatter into the pool.
    /// Reads outstanding at once. An expert record is nine separate pieces
    /// (gate/up/down x weight/scales/biases), so this issues 9N preads of
    /// ~307 KB rather than N of 2.76 MB. Swept 2026-08-30: 12 and 32 tie at
    /// ~4.5 GB/s and 64 and 128 are worse, so this is not queue-depth-limited
    /// and raising it only oversubscribes.
    public static let defaultQueueDepth: Int = {
        guard let raw = ProcessInfo.processInfo.environment["SLOTSTREAM_IO_QUEUE_DEPTH"],
            let n = Int(raw)
        else { return 12 }
        // Zero used to launch no workers and copy uninitialized staging memory;
        // a negative value could trap in concurrentPerform.
        return min(max(n, 1), 128)
    }()

    /// Maximum expert records whose nine staging tensors coexist. A 256-token
    /// prefill can route all 512 experts of a layer: loading that as one unit is
    /// 1.4 GB before MLX materializes the scatter inputs, and was the unexplained
    /// multi-GB RSS step on the first nontrivial prompt. Reads stay QD-parallel
    /// inside each batch; only the peak working set is bounded here.
    public static let defaultLoadBatch: Int = {
        guard let raw = ProcessInfo.processInfo.environment["SLOTSTREAM_EXPERT_LOAD_BATCH"],
            let n = Int(raw)
        else { return 32 }
        return min(max(n, 1), Geometry.expertsPerLayer)
    }()

    public func readBatch(_ keys: [ExpertKey], queueDepth: Int = ExpertStore.defaultQueueDepth)
        -> [MLXArray]
    {
        let n = keys.count
        precondition(n > 0)
        // one staging buffer per piece
        var buffers: [UnsafeMutableRawPointer] = []
        for pb in pieceRowBytes {
            var p: UnsafeMutableRawPointer? = nil
            let rc = posix_memalign(&p, 16384, n * pb)
            guard rc == 0, let p else {
                buffers.forEach { free($0) }
                // Nothing above can recover a partial expert batch, but the
                // message should say what ran out rather than trapping on nil.
                fatalError(
                    "out of memory staging \(n) expert records (\(n * pb) B): "
                        + "lower --experts-per-layer or --memory-gb")
            }
            buffers.append(p)
        }
        // 9n reads, spread across worker lanes
        let jobs: [(piece: Int, slot: Int)] = (0 ..< n).flatMap { s in (0 ..< 9).map { (piece: $0, slot: s) } }
        let lanes = min(max(queueDepth, 1), jobs.count)
        DispatchQueue.concurrentPerform(iterations: lanes) { lane in
            var j = lane
            while j < jobs.count {
                let (p, s) = jobs[j]
                let key = keys[s]
                let r = refs[key.layer][p]
                let pb = pieceRowBytes[p]
                index.pread(
                    into: buffers[p] + s * pb, r, offset: key.expert * pb, count: pb)
                j += lanes
            }
        }

        // Transfer each aligned staging buffer directly to MLX. The previous
        // path copied every 1.4 GB batch into a Swift Array and then copied it
        // again into MLX, so raw + Swift + MLX copies coexisted at peak and
        // were invisible to MLX's allocator counter.
        var out: [MLXArray] = []
        for (p, r) in refs[0][0 ..< 9].enumerated() {
            let shape = [n] + Array(r.shape.dropFirst())
            let dtype: DType
            switch r.dtype {
            case "U32": dtype = .uint32
            case "BF16": dtype = .bfloat16
            default:
                for q in p ..< buffers.count { free(buffers[q]) }
                fatalError("unexpected expert dtype \(r.dtype)")
            }
            let owned = buffers[p]
            out.append(MLXArray(rawPointer: owned, shape, dtype: dtype) { free(owned) })
        }
        eval(out)
        return out
    }

    /// CPU-only prefetch path. Ownership stays in one record per ring slot;
    /// MLX arrays are created later on the decode thread when a record is used.
    fileprivate func readRecords(
        _ keys: [ExpertKey], queueDepth: Int = ExpertStore.defaultQueueDepth
    ) -> [StagedExpertRecord] {
        let records = keys.map { _ in StagedExpertRecord(pieceBytes: pieceRowBytes) }
        let jobs = (0 ..< keys.count).flatMap { record in
            (0 ..< Self.pieces.count).map { (record: record, piece: $0) }
        }
        let lanes = min(max(queueDepth, 1), jobs.count)
        DispatchQueue.concurrentPerform(iterations: lanes) { lane in
            var j = lane
            while j < jobs.count {
                let job = jobs[j]
                let key = keys[job.record]
                let ref = refs[key.layer][job.piece]
                let bytes = pieceRowBytes[job.piece]
                index.pread(
                    into: records[job.record].pieces[job.piece]!, ref,
                    offset: key.expert * bytes, count: bytes)
                j += lanes
            }
        }
        return records
    }

    fileprivate func materialize(_ record: StagedExpertRecord) -> [MLXArray] {
        refs[0].indices.map { piece in
            let ref = refs[0][piece]
            let dtype: DType = ref.dtype == "U32" ? .uint32 : .bfloat16
            let pointer = record.pieces[piece]!
            record.pieces[piece] = nil
            return MLXArray(
                rawPointer: pointer, [1] + Array(ref.shape.dropFirst()), dtype: dtype
            ) { free(pointer) }
        }
    }
}

// MARK: - Slot pool

/// Fixed-size speculative staging. A generation on each ring slot is the CPU
/// fence: demand waits only for a matching in-flight record, and a late worker
/// can never publish into a slot that has since rotated to another key.
private final class ExpertPrefetchRing {
    private struct Entry {
        var key: ExpertKey?
        var generation = 0
        var loading = false
        var record: StagedExpertRecord?
    }

    private let store: ExpertStore
    private let queue = DispatchQueue(label: "slotstream.expert-prefetch", qos: .userInitiated)
    private let fence = NSCondition()
    private var entries: [Entry]
    private var cursor = 0

    init(capacity: Int, store: ExpertStore) {
        self.store = store
        entries = Array(repeating: Entry(), count: capacity)
    }

    var capacity: Int { entries.count }

    func reconcile(_ desired: [ExpertKey]) {
        guard !entries.isEmpty else { return }
        let desiredSet = Set(desired)
        var work: [(slot: Int, generation: Int, key: ExpertKey)] = []
        fence.lock()
        let present = Set(entries.compactMap { entry -> ExpertKey? in
            guard let key = entry.key, desiredSet.contains(key) else { return nil }
            return key
        })
        for key in desired where !present.contains(key) {
            var selected: Int?
            for _ in 0 ..< entries.count {
                let slot = cursor
                cursor = (cursor + 1) % entries.count
                let entry = entries[slot]
                if !entry.loading && (entry.key == nil || !desiredSet.contains(entry.key!)) {
                    selected = slot
                    break
                }
            }
            guard let slot = selected else { break }
            entries[slot].generation += 1
            entries[slot].key = key
            entries[slot].loading = true
            entries[slot].record = nil
            work.append((slot, entries[slot].generation, key))
        }
        fence.unlock()

        for lo in stride(from: 0, to: work.count, by: ExpertStore.defaultLoadBatch) {
            let hi = min(lo + ExpertStore.defaultLoadBatch, work.count)
            let batchWork = Array(work[lo ..< hi])
            queue.async { [weak self] in
                guard let self else { return }
                let batch = self.store.readRecords(batchWork.map(\.key))
                self.fence.lock()
                for (i, item) in batchWork.enumerated() {
                    guard self.entries[item.slot].generation == item.generation,
                        self.entries[item.slot].key == item.key
                    else { continue }
                    self.entries[item.slot].record = batch[i]
                    self.entries[item.slot].loading = false
                }
                self.fence.broadcast()
                self.fence.unlock()
            }
        }
    }

    func take(_ key: ExpertKey) -> StagedExpertRecord? {
        guard !entries.isEmpty else { return nil }
        fence.lock()
        defer { fence.unlock() }
        while true {
            guard let slot = entries.firstIndex(where: { $0.key == key }) else { return nil }
            if entries[slot].loading {
                fence.wait()
                continue
            }
            let record = entries[slot].record
            entries[slot].key = nil
            entries[slot].record = nil
            return record
        }
    }

    func discard(_ key: ExpertKey) {
        fence.lock()
        if let slot = entries.firstIndex(where: { $0.key == key }), !entries[slot].loading {
            entries[slot].key = nil
            entries[slot].record = nil
        }
        fence.unlock()
    }

    func clear() {
        queue.sync {}
        fence.lock()
        for i in entries.indices {
            entries[i].generation += 1
            entries[i].key = nil
            entries[i].loading = false
            entries[i].record = nil
        }
        fence.broadcast()
        fence.unlock()
    }
}

/// A fixed pool of expert slots shared across all layers (uniform shape), with
/// CLOCK eviction. `ensure` maps (layer, expert) keys to slot indices, loading
/// misses in one batched read + scatter. Bit-exact: the pool holds the same
/// quantized bytes the checkpoint does.
public final class SlotPool {
    public private(set) var slots: Int
    private let cfg: ModelConfig
    private let store: ExpertStore

    // pools, same order as ExpertStore.pieces
    public private(set) var pools: [MLXArray] = []

    private var map: [ExpertKey: Int] = [:]
    private var keyOf: [ExpertKey?]
    private var refBit: [Bool]
    private var pinned: [Bool]
    private var hand = 0
    public private(set) var hits = 0
    public private(set) var misses = 0
    public private(set) var prefetchPromotions = 0
    // Per-layer normalized heat: heat[l][e] sums to 1 per layer.
    // Update on each MoE dispatch: hit experts get + (1-alpha)/k, all decay by alpha.
    // alpha <1, controlled by SLOTSTREAM_HEAT_ALPHA (or legacy SLOTSTREAM_LFU_DECAY).
    private var heat: [[Double]] = Array(repeating: Array(repeating: 1.0/512.0, count: 512), count: 48)
    private var lastAccess: [ExpertKey: Int] = [:]
    private var lfuClock = 0
    static let prefetchRecords: Int = {
        guard let raw = ProcessInfo.processInfo.environment["SLOTSTREAM_PREFETCH_RECORDS"],
            let n = Int(raw)
        else { return 32 }
        return min(max(n, 0), 256)
    }()
    static let prefetchLookahead: Int = {
        guard let raw = ProcessInfo.processInfo.environment["SLOTSTREAM_PREFETCH_LAYERS"],
            let n = Int(raw)
        else { return 4 }
        return min(max(n, 1), Geometry.layers)
    }()
    static let prefetchDistancePower: Double = {
        guard let raw = ProcessInfo.processInfo.environment["SLOTSTREAM_PREFETCH_DISTANCE_POWER"],
            let value = Double(raw), value >= 0, value <= 4
        else { return 1.0 }
        return value
    }()
    private lazy var prefetch = ExpertPrefetchRing(capacity: Self.prefetchRecords, store: store)
    static let heatAlpha: Double = {
        if let s = ProcessInfo.processInfo.environment["SLOTSTREAM_HEAT_ALPHA"], let v = Double(s), v >= 0, v <= 1 { return v }
        if let s = ProcessInfo.processInfo.environment["SLOTSTREAM_LFU_DECAY"], let v = Double(s), v >= 0, v <= 1 { return v }
        return 0.99
    }()
    // Kept for compat: decay interval now unused (per-access decay)
    static let lfuDecay: Double = heatAlpha
    static let lfuDecayInterval: Int = {
        if let s = ProcessInfo.processInfo.environment["SLOTSTREAM_LFU_DECAY_INTERVAL"], let v = Int(s), v > 0 { return v }
        return 0
    }()
    static let lfuDecodeOnly: Bool = {
        ProcessInfo.processInfo.environment["SLOTSTREAM_LFU_DECODE_ONLY"] != nil
    }()

    public var poolBytes: Int { pools.reduce(0) { $0 + $1.nbytes } }
    /// Bytes per expert record, measured from the checkpoint headers.
    public var recordBytes: Int { store.recordBytes }
    /// The cache size in the per-layer unit of intuition (the pool itself is
    /// global and shared -- hot layers borrow from cold ones).
    public var slotsPerLayer: Double { Double(slots) / Double(cfg.numLayers) }

    /// Per-piece shapes for a pool of `n` slots (order = ExpertStore.pieces).
    private static func poolShapes(_ n: Int, _ cfg: ModelConfig) -> [(shape: [Int], dtype: DType)] {
        let h = cfg.hiddenSize
        let ff = cfg.moeIntermediate
        let g = cfg.qGroup
        return [
            ([n, ff, h / 8], .uint32), ([n, ff, h / g], .bfloat16), ([n, ff, h / g], .bfloat16),
            ([n, ff, h / 8], .uint32), ([n, ff, h / g], .bfloat16), ([n, ff, h / g], .bfloat16),
            ([n, h, ff / 8], .uint32), ([n, h, ff / g], .bfloat16), ([n, h, ff / g], .bfloat16),
        ]
    }

    public init(slots: Int, store: ExpertStore) {
        self.slots = slots
        self.store = store
        self.cfg = store.index.config
        self.keyOf = Array(repeating: nil, count: slots)
        self.refBit = Array(repeating: false, count: slots)
        self.pinned = Array(repeating: false, count: slots)
        pools = Self.poolShapes(slots, cfg).map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
        eval(pools)
    }

    /// Resize the pool. Must only be called between requests (the caller holds
    /// the engine's generation lock); stale pins are cleared, not honored.
    ///
    /// Grow keeps the cached contents: each piece is gathered into its larger
    /// replacement one at a time, so the transient overhead stays bounded by
    /// one piece — and growth only happens when availability covers the new
    /// pool anyway. Shrink FREES the old tensors before allocating the small
    /// ones (transient = max(old, new), never the sum) and restarts cold:
    /// shrink happens under memory pressure, where holding two pools to
    /// preserve cache warmth would spike memory at exactly the wrong moment.
    /// The cache refills from SSD in a few seconds of subsequent requests.
    /// Byte-exactness is unaffected either way (golden-equivalence invariant:
    /// pool size and content never change the math).
    public func resize(to newSlots: Int) {
        let n = max(newSlots, 1)
        if n == slots { return }
        clearPrefetch()
        unpinAll()
        if n > slots {
            // grow, preserving contents in the slot-index prefix
            let occupied = (0 ..< slots).filter { keyOf[$0] != nil }
            let idx = MLXArray(occupied.map(Int32.init))
            var newKeyOf: [ExpertKey?] = Array(repeating: nil, count: n)
            var newRef = Array(repeating: false, count: n)
            map.removeAll(keepingCapacity: true)
            for (i, s) in occupied.enumerated() {
                newKeyOf[i] = keyOf[s]
                newRef[i] = refBit[s]
                map[keyOf[s]!] = i
            }
            for (p, spec) in Self.poolShapes(n, cfg).enumerated() {
                let np = MLXArray.zeros(spec.shape, dtype: spec.dtype)
                if !occupied.isEmpty { np[0 ..< occupied.count] = pools[p][idx] }
                eval(np)
                pools[p] = np  // old piece freed here, bounding the transient
            }
            keyOf = newKeyOf
            refBit = newRef
            pinned = Array(repeating: false, count: n)
            hand = occupied.count % n
        } else {
            // shrink: free first, allocate after, start cold
            pools = []
            map.removeAll(keepingCapacity: true)
            keyOf = Array(repeating: nil, count: n)
            refBit = Array(repeating: false, count: n)
            pinned = Array(repeating: false, count: n)
            hand = 0
            pools = Self.poolShapes(n, cfg).map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
            eval(pools)
        }
        slots = n
    }

    private func updateHeat(layer: Int, hits: Set<Int>) {
        let alpha = Self.heatAlpha
        guard alpha >= 0, alpha < 1, hits.count > 0 else { return }
        let k = Double(hits.count)
        let boost = (1.0 - alpha) / k
        for e in 0..<512 {
            if hits.contains(e) {
                heat[layer][e] = heat[layer][e] * alpha + boost
            } else {
                heat[layer][e] *= alpha
            }
        }
        // Renormalize to 1 (floating error)
        let sum = heat[layer].reduce(0, +)
        if sum > 0 {
            for e in 0..<512 { heat[layer][e] /= sum }
        }
    }

    private func victim() -> Int {
        // Coldest heat first, LRU breaks ties — O(slots) scan, slots ≤ 24576
        var best: Int? = nil
        var bestHeat = Double.greatestFiniteMagnitude
        var bestTime = Int.max
        var pinnedCount = 0
        for s in 0..<slots {
            if pinned[s] { pinnedCount += 1; continue }
            let key = keyOf[s]
            let h: Double
            let t: Int
            if let k = key {
                h = heat[k.layer][k.expert]
                t = lastAccess[k] ?? -1
            } else {
                h = -1 // empty slot is coldest
                t = -1
            }
            if h < bestHeat || (h == bestHeat && t < bestTime) {
                bestHeat = h; bestTime = t; best = s
            }
        }
        guard let b = best else {
            preconditionFailure(
                "slot pool exhausted: all \(slots) slots pinned (\(pinnedCount)). The pool must "
                    + "hold at least one prefill chunk's expert set per layer "
                    + "(~512 + margin); raise --experts-per-layer.")
        }
        return b
    }

    /// Keep a rotating, distance-adjusted window ahead of decode. Nearby
    /// layers receive more ring records; within each layer, LFU heat predicts
    /// which experts are worth reading. Reconciliation preserves records still
    /// in the target window and rotates the rest through the bounded ring.
    public func prefetchDecode(afterLayer: Int, layerCount: Int) {
        guard Self.prefetchRecords > 0, layerCount > 0 else { return }
        let staleLayer = (afterLayer % layerCount + layerCount) % layerCount
        let lookahead = min(Self.prefetchLookahead, layerCount)
        let weights = (1 ... lookahead).map { 1.0 / pow(Double($0), Self.prefetchDistancePower) }
        let weightSum = weights.reduce(0, +)
        var targets = weights.map { Int((Double(prefetch.capacity) * $0 / weightSum).rounded(.down)) }
        var remainder = prefetch.capacity - targets.reduce(0, +)
        var d = 0
        while remainder > 0 {
            targets[d] += 1
            remainder -= 1
            d = (d + 1) % lookahead
        }

        var desired: [ExpertKey] = []
        for distance in 1 ... lookahead {
            let layer = (staleLayer + distance) % layerCount
            let hottest = (0 ..< cfg.numExperts)
                .filter { map[ExpertKey(layer, $0)] == nil }
                .sorted { heat[layer][$0] > heat[layer][$1] }
            desired.append(contentsOf: hottest.prefix(targets[distance - 1]).map { ExpertKey(layer, $0) })
        }
        prefetch.reconcile(desired)
    }

    public func clearPrefetch() {
        prefetch.clear()
    }

    /// 48 x 512 heat matrix: per-layer probability (sums to 1 per layer).
    public func heatMatrix() -> [[Double]] {
        heat
    }
    public func heatMatrixInt(scaled: Int = 1000000) -> [[Int]] {
        heat.map { row in row.map { Int(($0 * Double(scaled)).rounded()) } }
    }

    /// Sparse heat for API: [[layer, expert, heat*1e6, lastAccess]] sorted hottest first.
    public func heatSparse() -> [[Int]] {
        var out: [[Int]] = []
        out.reserveCapacity(48*10)
        for l in 0..<48 {
            for e in 0..<512 {
                let h = heat[l][e]
                if h > 0.0001 {
                    out.append([l, e, Int((h * 1_000_000).rounded()), lastAccess[ExpertKey(l, e)] ?? 0])
                }
            }
        }
        out.sort { $0[2] > $1[2] }
        return out
    }

    /// Ensure all keys resident; returns slot index per key (same order).
    /// Pins the returned slots until `unpinAll()`. Per-layer heat: each layer
    /// sums to 1, hits get boost, others decay by alpha, LRU breaks ties.
    public func ensure(_ keys: [ExpertKey]) -> [Int] {
        var result = Array(repeating: -1, count: keys.count)
        var missKeys: [ExpertKey] = []
        var missPos: [Int] = []
        var prefetchedRecords: [StagedExpertRecord?] = []
        let isDecodeOnly = Self.lfuDecodeOnly
        let isPrefillBatch = keys.count > 30
        let shouldCount = !(isDecodeOnly && isPrefillBatch)
        var hitsPerLayer: [Int: Set<Int>] = [:]
        for (i, k) in keys.enumerated() {
            if shouldCount {
                hitsPerLayer[k.layer, default: Set()].insert(k.expert)
                lastAccess[k] = lfuClock; lfuClock += 1
            }
            if let s = map[k] {
                prefetch.discard(k)
                result[i] = s
                refBit[s] = true
                pinned[s] = true
                hits += 1
            } else {
                missKeys.append(k)
                missPos.append(i)
                prefetchedRecords.append(prefetch.take(k))
                misses += 1
            }
        }
        if shouldCount {
            for (layer, hits) in hitsPerLayer {
                updateHeat(layer: layer, hits: hits)
            }
        }
        if !missKeys.isEmpty {
            let tMiss = Date()
            // choose victims first (so scatter is one batched op)
            var slotIdx: [Int32] = []
            for k in missKeys {
                let s = victim()
                if let old = keyOf[s] { map.removeValue(forKey: old) }
                keyOf[s] = k
                map[k] = s
                refBit[s] = true
                pinned[s] = true
                slotIdx.append(Int32(s))
            }
            var demand: [Int] = []
            var promoted = false
            for i in missKeys.indices {
                if let record = prefetchedRecords[i] {
                    let rows = store.materialize(record)
                    let idx = MLXArray([slotIdx[i]])
                    for p in 0 ..< 9 { pools[p][idx] = rows[p] }
                    prefetchPromotions += 1
                    promoted = true
                } else {
                    demand.append(i)
                }
            }
            if promoted { eval(pools) }
            // Bound staging independently of how many unique experts this
            // token batch routed. Evaluating each scatter before reading the
            // next slice lets the prior raw buffers be released immediately.
            var lo = 0
            while lo < demand.count {
                let hi = min(lo + ExpertStore.defaultLoadBatch, demand.count)
                let selection = Array(demand[lo ..< hi])
                let tIO = Date()
                let batch = store.readBatch(selection.map { missKeys[$0] })
                ioSeconds += -tIO.timeIntervalSinceNow
                let tScatter = Date()
                let idx = MLXArray(selection.map { slotIdx[$0] })
                for p in 0 ..< 9 {
                    pools[p][idx] = batch[p]
                }
                eval(pools)
                scatterSeconds += -tScatter.timeIntervalSinceNow
                lo = hi
            }
            recordsFetched += demand.count
            fillSeconds += -tMiss.timeIntervalSinceNow
            for (j, i) in missPos.enumerated() { result[i] = Int(slotIdx[j]) }
        }
        return result
    }

    public func unpinAll() {
        for i in 0 ..< slots where pinned[i] { pinned[i] = false }
    }

    /// Where prefill time actually goes. Split out because "prefill is slow"
    /// is not actionable: reading the records, scattering them into the pool,
    /// and the compute over them are three different problems with three
    /// different fixes — and measuring the split is what showed the chunk size
    /// was the lever and read-ahead was not.
    public private(set) var ioSeconds = 0.0
    public private(set) var scatterSeconds = 0.0
    public private(set) var fillSeconds = 0.0
    public private(set) var recordsFetched = 0

    public var hitRate: Double {
        let t = hits + misses
        return t == 0 ? 0 : Double(hits) / Double(t)
    }

    public func resetStats() {
        hits = 0
        misses = 0
        prefetchPromotions = 0
        ioSeconds = 0
        scatterSeconds = 0
        fillSeconds = 0
        recordsFetched = 0
    }
}
