// DeepSeek-V4-Flash forward pass over the landed seam: `DS4Config` (GGUF
// metadata), `DS4Weights`/`DS4LayerWeights` (resident trunk) and
// `DS4ExpertStore` (MXFP4 routed experts). This file owns the orchestration:
// mHC pre/post, CSA attention with the compressed-KV stream and indexer,
// hash/top-6 MoE, and the output head. Numerics follow the ds4 C engine.

import Foundation
import MLX
import MLXNN

/// Router normalization floor from the reference engine (2^-14).
enum DS4Constants {
    static let routerNormMin: Float = 6.103515625e-5
    static let indexerQatScale: Float = 0.08838834764831845
}

/// Forward-pass trace (SLOTSTREAM_DS4_TRACE=1): per-layer activation and
/// routing stats on stderr, for bisecting nonsense decodes against the real
/// checkpoint. Off by default; every read syncs, so it is diagnostics only.
enum DS4Trace {
    static let on = ProcessInfo.processInfo.environment["SLOTSTREAM_DS4_TRACE"] == "1"

    static func log(_ s: @autoclosure () -> String) {
        guard on else { return }
        FileHandle.standardError.write((s() + "\n").data(using: .utf8)!)
    }

    /// Scalar RMS of the whole tensor (syncs).
    static func rms(_ x: MLXArray) -> Float {
        sqrt(x.asType(.float32).square().mean().item(Float.self))
    }

    static func finite(_ x: MLXArray) -> Bool {
        isNaN(x.asType(.float32)).sum().item(Int.self) == 0
    }
}

public final class DS4Model {
    public let cfg: DS4Config
    public let weights: DS4Weights
    public let experts: DS4ExpertStore
    /// The shared slot pool (SlotPool). Serves the decode step: one token
    /// routes `expertUsedCount` experts per layer, so the pool caches the hot
    /// records across tokens. Prefill passes never touch it (see
    /// `routedExperts`).
    public let pool: SlotPool
    /// Rope params per layer: dense base for ratio-0 layers, compressed base
    /// otherwise (ds4.c:11893-11937).
    private(set) var ropeParams: [DS4Math.RopeParams]

    public init(cfg: DS4Config, weights: DS4Weights, experts: DS4ExpertStore,
                pool: SlotPool) throws {
        self.cfg = cfg
        self.weights = weights
        self.experts = experts
        self.pool = pool
        precondition(cfg.compressRatios.count >= cfg.blockCount,
                     "compressRatios must cover every block")
        ropeParams = (0..<cfg.blockCount).map { l in
            let compressed = (cfg.compressRatio(at: l) ?? 0) != 0
            let base = compressed ? cfg.compressRopeFreqBase : cfg.ropeFreqBase
            let freqScale: Float = compressed ? 1.0 / cfg.ropeScalingFactor : 1.0
            let ext: Float = compressed && cfg.ropeScalingFactor > 1 ? 1 : 0
            return DS4Math.RopeParams.make(
                base: base, freqScale: freqScale, extFactor: ext,
                origCtx: cfg.ropeOriginalContextLength, betaFast: cfg.yarnBetaFast,
                betaSlow: cfg.yarnBetaSlow, nRot: cfg.ropeDimensionCount)
        }
    }

    public func makeState() -> DS4State { DS4State(cfg: cfg) }

    /// Forward `ids` starting at the state's current position; returns logits
    /// [T, vocab] and advances the state. T == 1 is the decode step; larger T
    /// is prefill (projections batched, compressor/indexer/attention per
    /// token). Throws when routed-expert reads fail.
    public func forward(_ ids: [Int], state: DS4State) throws -> MLXArray {
        precondition(!ids.isEmpty, "DS4Model.forward: empty ids")
        return head(try hiddenStates(ids, state: state))
    }

    /// The layer stack only: runs `ids` through all blocks and returns the
    /// post-FFN hidden state `[T, hc, embd]`, advancing the state. The head is
    /// per-row, so `forward`/`lastLogits` can slice before it and skip the
    /// [T, vocab] logits transient on every position but the one they need.
    public func hiddenStates(_ ids: [Int], state: DS4State) throws -> MLXArray {
        precondition(!ids.isEmpty, "DS4Model.hiddenStates: empty ids")
        let t = ids.count
        let pos0 = state.tokenCount
        let idArr = MLXArray(ids.map { Int32($0) })
        let embd = weights.embedding(idArr).asType(.float32).reshaped([t, cfg.embeddingLength])
        var hc = tiled(embd.reshaped([t, 1, cfg.embeddingLength]),
                       repetitions: [1, cfg.hyperConnectionCount, 1])
        let positions = MLXArray((0..<t).map { Int32(pos0 + $0) })

        for l in 0..<cfg.blockCount {
            hc = try attentionSublayer(l, hc: hc, cache: state.caches[l], state: state,
                                       positions: positions, pos0: pos0)
            hc = try ffnSublayer(l, hc: hc, ids: ids)
            if DS4Trace.on {
                DS4Trace.log("L\(l) out pos0=\(pos0) t=\(t) hc rms=\(DS4Trace.rms(hc)) finite=\(DS4Trace.finite(hc))")
            }
            let pt0 = Date()
            eval(hc)
            if Self.prof { Self.profStage["layer.eval", default: 0] += -pt0.timeIntervalSinceNow }
        }
        state.advance(t)
        if Self.prof, pos0 == 0 || t == 1 {
            let tot = Self.profStage.values.reduce(0, +)
            let rows = Self.profStage.sorted { $0.value > $1.value }.prefix(12).map { "\($0.key)=\(String(format: "%.3f", $0.value))" }
            FileHandle.standardError.write("PROF t=\(t) total=\(String(format: "%.2f", tot))s: \(rows.joined(separator: " "))\n".data(using: .utf8)!)
            Self.profStage = [:]
        }
        return hc
    }

    /// Logits for the last consumed position only.
    ///
    /// The head runs on the single sliced row: it is strictly per-position
    /// (RMS norm, sigmoid, weighted sum and the vocab matmul never mix rows),
    /// so this is the same value `forward` produces for that position, minus
    /// the vocab-sized transient for the other T-1 of them.
    public func lastLogits(_ ids: [Int], state: DS4State) throws -> MLXArray {
        let hidden = try hiddenStates(ids, state: state)
        return head(hidden[(hidden.dim(0) - 1)...])
    }

    // MARK: head

    /// HC collapse (output_hc_head_one, ds4.c:15616-15641): unweighted RMSNorm
    /// over the flat HC state, per-stream sigmoid weights, weighted sum,
    /// output RMSNorm, vocab projection.
    func head(_ hc: MLXArray) -> MLXArray {
        let t = hc.dim(0)
        let flat = DS4Math.rmsNormNoWeight(hc.reshaped([t, -1]), eps: cfg.rmsNormEpsilon)
        let pre = matmul(flat, weights.outputHcFn.asType(.float32).transposed()) // [T, 4]
        let wts = MLXNN.sigmoid(pre * weights.outputHcScale.asType(.float32)
            + weights.outputHcBase.asType(.float32)) + cfg.hyperConnectionEpsilon
        let embd = (hc * wts.expandedDimensions(axis: -1)).sum(axis: 1)
        let norm = DS4Math.rmsNorm(embd, weight: weights.outputNorm, eps: cfg.rmsNormEpsilon)
        return weights.lmHead(norm)
    }

    // MARK: attention sublayer

    static let prof: Bool = ProcessInfo.processInfo.environment["SLOTSTREAM_DS4_PROF"] == "1"
    static var profStage: [String: Double] = [:]
    static func profStep<T>(_ name: String, _ f: () throws -> T) rethrows -> T {
        guard prof else { return try f() }
        let t0 = Date()
        let r = try f()
        if let a = r as? MLXArray { eval(a) }
        profStage[name, default: 0] += -t0.timeIntervalSinceNow
        return r
    }

    func attentionSublayer(_ l: Int, hc: MLXArray, cache: DS4LayerCache, state: DS4State,
                           positions: MLXArray, pos0: Int) throws -> MLXArray {
        let w = weights.layer(l)
        let t = hc.dim(0)
        let ratio = cache.compressRatio
        let split = Self.profStep("attn.hcSplit\(t == 1 ? ".dec" : "")") { DS4Math.hcSplit(hc: hc, fn: w.hcAttnFn, scale: w.hcAttnScale,
                                    base: w.hcAttnBase, eps: cfg.rmsNormEpsilon,
                                    iters: cfg.sinkhornIterations) }
        let xSub = DS4Math.hcWeightedSum(pre: split.pre, hc: hc)
        let attnNorm = DS4Math.rmsNorm(xSub, weight: w.attnNorm, eps: cfg.rmsNormEpsilon)
        if DS4Trace.on {
            let combRowSums = split.comb.sum(axis: -1) // [T, dst] — each ~1
            DS4Trace.log("L\(l) attn hc.pre[\(split.pre.dim(0))] mean=\(split.pre.mean().item(Float.self)) "
                + "post mean=\(split.post.mean().item(Float.self)) "
                + "combRowSum mean=\(combRowSums.mean().item(Float.self)) min=\(combRowSums.min().item(Float.self)) max=\(combRowSums.max().item(Float.self)) "
                + "xSub rms=\(DS4Trace.rms(xSub)) attnNorm rms=\(DS4Trace.rms(attnNorm)) finite=\(DS4Trace.finite(attnNorm))")
        }

        let qr = w.attnQA(attnNorm) // [T, qLoraRank]
        let qrN = DS4Math.rmsNorm(qr, weight: w.attnQANorm, eps: cfg.rmsNormEpsilon)
        var q = Self.profStep("attn.q") { w.attnQB(qrN).reshaped([t, cfg.headCount, cfg.keyLength]) }
        q = DS4Math.perHeadRmsNorm(q, eps: cfg.rmsNormEpsilon)
        q = DS4Math.ropeTail(q, positions: positions, p: ropeParams[l], inverse: false)

        let kvRaw = w.attnKV(attnNorm) // [T, keyLength]
        let kvN = DS4Math.rmsNorm(kvRaw, weight: w.attnKVANorm, eps: cfg.rmsNormEpsilon)
        let kvR = DS4Math.ropeTail(kvN.reshaped([t, cfg.headCountKV, cfg.keyLength]),
                                   positions: positions, p: ropeParams[l], inverse: false)
            .reshaped([t, cfg.keyLength])
        let kvQ = Self.profStep("attn.kvfp8") { DS4Math.fp8RoundTrip(kvR, rotDim: cfg.ropeDimensionCount).asType(.float16) }
        cache.pushRaw(kvQ, firstPosition: pos0)

        var kvC: MLXArray?
        var scC: MLXArray?
        if ratio != 0 {
            guard let cKv = w.attnCompressorKV, let cGate = w.attnCompressorGate,
                let cApe = w.attnCompressorApe, w.attnCompressorNorm != nil,
                let width = w.compressorOutWidth else {
                fatalError("DS4 layer \(l) has ratio \(ratio) but missing compressor weights")
            }
            kvC = matmul(attnNorm, cKv.asType(.float32).transposed()) // [T, width]
            // ape is MLX [ratio, width]; row (pos % ratio) is this token's map.
            let cols = MLXArray((0..<t).map { Int32((pos0 + $0) % ratio) })
            scC = matmul(attnNorm, cGate.asType(.float32).transposed())
                + cApe.asType(.float32)[cols] // [T, width]
        }

        var icKvC: MLXArray?
        var icScC: MLXArray?
        if ratio == 4 {
            guard let icKv = w.indexerCompressorKV, let icGate = w.indexerCompressorGate,
                let icApe = w.indexerCompressorApe else {
                fatalError("DS4 layer \(l) has ratio 4 but missing indexer compressor weights")
            }
            let width = 2 * cfg.indexerKeyLength
            icKvC = matmul(attnNorm, icKv.asType(.float32).transposed())
            let cols = MLXArray((0..<t).map { Int32((pos0 + $0) % ratio) })
            icScC = matmul(attnNorm, icGate.asType(.float32).transposed())
                + icApe.asType(.float32)[cols] // [T, width]
        }

        var outs: [MLXArray] = []
        outs.reserveCapacity(t)
        for i in 0..<t {
            let pos = pos0 + i
            if ratio != 0 {
                if let pooled = DS4Model.compressorStep(state: &cache.compState, kvCur: kvC![i],
                                                        scCur: scC![i], pos: pos,
                                                        headDim: cfg.keyLength, ratio: ratio) {
                    let normed = DS4Math.rmsNormNoWeight(pooled, eps: cfg.rmsNormEpsilon)
                        * w.attnCompressorNorm!.asType(.float32)
                    let compPos = MLXArray([Int32(pos + 1 - ratio)])
                    let roped = DS4Math.ropeTail(normed.reshaped([1, 1, cfg.keyLength]),
                                                 positions: compPos, p: ropeParams[l],
                                                 inverse: false).reshaped([cfg.keyLength])
                    let (codes, scales) = DS4Math.fp8Encode(
                        roped[0..<cache.nopeDim].reshaped([1, cache.nopeDim]))
                    cache.appendCompRow(payload: codes, scales: scales.asType(.float16),
                                        ropeDims: roped[cache.nopeDim...].reshaped([1, cfg.ropeDimensionCount]))
                }
                if ratio == 4 {
                    if let pooled = DS4Model.compressorStep(state: &cache.indexState,
                                                            kvCur: icKvC![i], scCur: icScC![i],
                                                            pos: pos, headDim: cfg.indexerKeyLength,
                                                            ratio: ratio) {
                        let normed = DS4Math.rmsNormNoWeight(pooled, eps: cfg.rmsNormEpsilon)
                            * w.indexerCompressorNorm!.asType(.float32)
                        let compPos = MLXArray([Int32(pos + 1 - ratio)])
                        let roped = DS4Math.ropeTail(normed.reshaped([1, 1, cfg.indexerKeyLength]),
                                                     positions: compPos, p: ropeParams[l],
                                                     inverse: false).reshaped([cfg.indexerKeyLength])
                        cache.appendIndexRow(
                            DS4Math.indexerQAT(roped.reshaped([1, cfg.indexerKeyLength])))
                    }
                }
            }

            let rawWin = cache.rawWindow(pos)
            var compRows = cache.dequantCompRows(cache.compCount)
            var compMask: MLXArray?
            if ratio == 4, let iQB = w.indexerAttnQB, let iProj = w.indexerProj,
                cache.indexCount > 0, compRows != nil {
                (compRows, compMask) = indexerSelect(compRows: compRows!, iQB: iQB, iProj: iProj,
                                                     qrNorm: qrN[i], attnNorm: attnNorm[i],
                                                     cache: cache, pos: pos,
                                                     params: ropeParams[l])
            }
            let heads = DS4Math.attentionRows(q: q[i], rawK: rawWin, compK: compRows,
                                              compMask: compMask, sinks: w.attnSinks,
                                              kqScale: 1.0 / sqrt(Float(cfg.keyLength)))
            outs.append(heads.expandedDimensions(axis: 0))
            eval(heads)
            state.recordLayer(layer: l, token: i, DS4LayerSnapshot(
                compState: cache.compState, indexState: cache.indexState,
                compCount: cache.compCount, indexCount: cache.indexCount))
        }
        var heads = Self.profStep("attn.rows") { concatenated(outs, axis: 0) } // [T, H, D]
        if DS4Trace.on {
            DS4Trace.log("L\(l) attn out rows=\(heads.dim(0)) rawWin=\(cache.rawWindow(pos0 + t - 1).dim(0)) "
                + "compRows=\(cache.compCount) indexRows=\(cache.indexCount) heads rms=\(DS4Trace.rms(heads)) finite=\(DS4Trace.finite(heads))")
        }
        heads = Self.profStep("attn.ropeinv") { DS4Math.ropeTail(heads, positions: positions, p: ropeParams[l], inverse: true) }
        let low = Self.profStep("attn.outA") { groupedOutProj(heads, oA: w.attnOutputA) }
        let attnOut = Self.profStep("attn.outB") { w.attnOutputB(low) }
        if DS4Trace.on {
            DS4Trace.log("L\(l) attn attnOut rms=\(DS4Trace.rms(attnOut)) finite=\(DS4Trace.finite(attnOut))")
        }
        return DS4Math.hcPost(blockOut: attnOut, hc: hc, post: split.post, comb: split.comb)
    }

    // MARK: compressor

    /// One-token compressor update (ds4.c:14121-14216). Writes the token's
    /// projected row into the recurrent state and, on ratio boundaries, pools
    /// the window with a per-dimension softmax over the rows (prev group from
    /// plane 0, current group from plane 1 for ratio 4) and returns the pooled
    /// row. Ratio-4 state rotates after every pool.
    static func compressorStep(state: inout DS4CompState, kvCur: MLXArray, scCur: MLXArray,
                               pos: Int, headDim: Int, ratio: Int) -> MLXArray? {
        let width = (ratio == 4 ? 2 : 1) * headDim
        let rows = ratio == 4 ? 2 * ratio : ratio
        state.ensure(width: width, rows: rows)
        let posMod = pos % ratio
        let row = ratio == 4 ? ratio + posMod : posMod
        state.write(row: row, kvRow: kvCur, scoreRow: scCur, width: width)
        guard (pos + 1) % ratio == 0 else { return nil }

        let kv = state.kv!
        let sc = state.score!
        let k: MLXArray
        let s: MLXArray
        if ratio == 4 {
            k = concatenated([kv[0..<ratio, 0..<headDim],
                              kv[ratio..<(2 * ratio), headDim..<(2 * headDim)]], axis: 0)
            s = concatenated([sc[0..<ratio, 0..<headDim],
                              sc[ratio..<(2 * ratio), headDim..<(2 * headDim)]], axis: 0)
        } else {
            k = kv[0..<ratio]
            s = sc[0..<ratio]
        }
        let m = s.max(axis: 0) // [headDim]
        let mSafe = which(m .<= -Float.infinity / 2, MLXArray(Float(0)), m)
        let weights = exp(s - mSafe)
        let denom = weights.sum(axis: 0)
        let num = (weights * k).sum(axis: 0)
        let pooled = which(denom .> 0, num / denom, MLXArray(Float(0)))
        if DS4Trace.on {
            DS4Trace.log("compress pool pos=\(pos) ratio=\(ratio) headDim=\(headDim) "
                + "pooled rms=\(DS4Trace.rms(pooled)) finite=\(DS4Trace.finite(pooled)) denom min=\(denom.min().item(Float.self))")
        }

        if ratio == 4 {
            state.rotateRatio4()
        }
        return pooled
    }

    // MARK: indexer selection

    /// Top-k compressed-row selection for ratio-4 layers (ds4.c:14550-14610).
    /// Returns the rows in visit order: all of them (index order) when
    /// indexCount fits the budget, otherwise the selected rows in descending
    /// indexer-score order — ds4's decode attends in score order.
    /// Top-k compressed-row selection for ratio-4 layers (ds4.c:14550-14610).
    /// The indexer only CHOOSES which compressed-KV rows attend — it returns
    /// `compRows` itself (the 512-wide attention keys), all of them in visit
    /// order when indexCount fits the budget, otherwise the top-k rows in
    /// descending indexer-score order (ds4's decode attends in score order).
    /// Returning the 128-wide indexer rows here instead attended over garbage.
    func indexerSelect(compRows: MLXArray, iQB: MLXArray, iProj: MLXArray, qrNorm: MLXArray,
                       attnNorm: MLXArray, cache: DS4LayerCache, pos: Int,
                       params: DS4Math.RopeParams) -> (rows: MLXArray, mask: MLXArray?) {
        let c = cache.indexCount
        guard c > cfg.indexerTopK else { return (compRows, nil) }
        let ic = cache.indexRows![0..<c] // [C, 128] F32

        let iq = matmul(qrNorm.reshaped([1, cfg.qLoraRank]),
                        iQB.asType(.float32).transposed()) // [1, 8192]
        let heads = iq.reshaped([1, cfg.indexerHeadCount, cfg.indexerKeyLength])
        let roped = DS4Math.ropeTail(heads, positions: MLXArray([Int32(pos)]), p: params,
                                     inverse: false)
        let q = DS4Math.indexerQAT(roped).reshaped([cfg.indexerHeadCount, cfg.indexerKeyLength])

        let iw = matmul(attnNorm.reshaped([1, cfg.embeddingLength]),
                        iProj.asType(.float32).transposed())
            * (1.0 / sqrt(Float(cfg.indexerKeyLength * cfg.indexerHeadCount))) // [1, 64]

        let dots = matmul(q, ic.transposed()) // [64, C] F32
        let scores = (maximum(dots, 0) * iw.transposed()).sum(axis: 0) // [C]
        let order = argSort(-scores, axis: 0)[0..<cfg.indexerTopK].asType(.int32)
        return (compRows[order], nil)
    }

    // MARK: grouped output projection

    /// attn_output_a [8192, 4096] split into `outputGroupCount` groups of
    /// 1024 rows; group g maps heads 8g..8g+7 (ds4.c:12096-12112). One
    /// batched quantizedMM over the groups, then the shared output_b.
    /// attn_output_a [8192, 4096] split into `outputGroupCount` groups of
    /// 1024 rows; group g maps heads 8g..8g+7 (ds4.c:12096-12112). One
    /// batched quantizedMM over the groups; the caller applies the shared
    /// output_b projection to the joined result.
    func groupedOutProj(_ heads: MLXArray, oA: QLinear) -> MLXArray {
        let t = heads.dim(0)
        let groupDim = cfg.headCount * cfg.keyLength / cfg.outputGroupCount
        // QLinear stores the weight PACKED: for bits b the uint32 words hold
        // 32/b elements each, so [out, in] becomes [out, in * b / 32] —
        // [8192, 1024] for the 8-bit attn_output_a, not [8192, 4096]. The
        // 3D reshape must follow the packed layout; reshaping as unpacked
        // threw a silent element-count trap on the first real forward pass.
        precondition(oA.w.dim(0) == cfg.outputGroupCount * cfg.outputLoraRank
            && oA.w.dim(1) == groupDim * oA.bits / 32,
            "oA is [\(oA.w.shape)], expected [\(cfg.outputGroupCount * cfg.outputLoraRank), "
                + "\(groupDim * oA.bits / 32)] packed uint32")
        let x = heads.reshaped([t, cfg.outputGroupCount, groupDim]).transposed(1, 0, 2)
        let w3 = oA.w.reshaped(
            [cfg.outputGroupCount, cfg.outputLoraRank, groupDim * oA.bits / 32])
        guard let scales = oA.scales else {
            fatalError("DS4 attn_output_a must be quantized")
        }
        let s3 = scales.reshaped(
            [cfg.outputGroupCount, cfg.outputLoraRank, groupDim / oA.groupSize])
        let b3 = oA.biases?.reshaped(
            [cfg.outputGroupCount, cfg.outputLoraRank, groupDim / oA.groupSize])
        let low = quantizedMM(x, w3, scales: s3, biases: b3, transpose: true,
                              groupSize: oA.groupSize, bits: oA.bits) // [G, T, rank]
        return low.transposed(1, 0, 2).reshaped([t, cfg.outputGroupCount * cfg.outputLoraRank])
    }

    // MARK: FFN sublayer

    func ffnSublayer(_ l: Int, hc: MLXArray, ids: [Int]) throws -> MLXArray {
        let w = weights.layer(l)
        let t = hc.dim(0)
        let split = DS4Math.hcSplit(hc: hc, fn: w.hcFfnFn, scale: w.hcFfnScale,
                                    base: w.hcFfnBase, eps: cfg.rmsNormEpsilon,
                                    iters: cfg.sinkhornIterations)
        let xSub = DS4Math.hcWeightedSum(pre: split.pre, hc: hc)
        let norm = DS4Math.rmsNorm(xSub, weight: w.ffnNorm, eps: cfg.rmsNormEpsilon)

        let clamp = cfg.swigluClampExp[l] // the limit itself, per ds4.c:12170
        let (sel, wts) = Self.profStep("ffn.router\(t == 1 ? ".dec" : "")") { routerSelection(norm: norm, w: w, ids: ids) }
        // Routing decisions for the cache simulator, the same flat stream
        // Qwen's MoELayer records (RouterTrace.swift): per MoE call, header
        // layer/tokens/topK int32 then tokens·topK int16 ids. Prefill passes
        // record too; trace_convert.py drops multi-token calls by default,
        // which is right here as well — DS4 prefill never touches the pool.
        if RouterTrace.on {
            RouterTrace.record(layer: l, tokens: t, topK: cfg.expertUsedCount,
                               ids: sel.asArray(Int32.self))
        }
        if DS4Trace.on {
            let k = cfg.expertUsedCount
            let sel0 = Array(sel[0].asArray(Int32.self).prefix(k))
            let wts0 = Array(wts[0].asArray(Float.self).prefix(k))
            var extra = ""
            if w.ffnGateTid2Eid != nil { extra = " (hash tid2eid)" }
            DS4Trace.log("L\(l) router\(t == 1 ? " dec" : "")\(extra) sel0=\(sel0) wts0=\(wts0.map { String(format: "%.4f", $0) }) "
                + "norm rms=\(DS4Trace.rms(norm)) finite=\(DS4Trace.finite(norm))")
        }
        // Decode (T == 1) goes through the slot pool, mirroring Qwen's
        // MoELayer.cached: unpin the previous layer, map the routed experts
        // into slots, and gather over the pool. Larger passes stay off the
        // pool — a partial chunk can route thousands of unique experts and
        // `ensure` pins what it maps — so they read each token's experts
        // straight from the GGUF instead (the sweep is the follow-up here,
        // and like the Qwen sweep it will leave the pool untouched).
        if t == 1 {
            pool.unpinAll()
            let moe = try Self.profStep("ffn.moe.pool") { try routedExpertsViaPool(x: norm[0], sel: sel[0], routeW: wts[0],
                                               w: w, layer: l) }
            let g = Self.profStep("ffn.shg") { w.ffnGateShexp(norm) }
            let u = Self.profStep("ffn.shu") { w.ffnUpShexp(norm) }
            let mid = DS4Math.swigluClamped(gate: g.asType(.float32), up: u.asType(.float32),
                                            clamp: clamp)
            let shared = w.ffnDownShexp(mid)
            let ffnOut = moe.expandedDimensions(axis: 0) + shared
            return DS4Math.hcPost(blockOut: ffnOut, hc: hc, post: split.post, comb: split.comb)
        }
        var moeParts: [MLXArray] = []
        moeParts.reserveCapacity(t)
        for i in 0..<t {
            moeParts.append(try Self.profStep("ffn.moe.batch") { try routedExperts(x: norm[i], sel: sel[i], routeW: wts[i], w: w, layer: l) })
            eval(moeParts[i])
        }
        let moe = concatenated(moeParts.map { $0.expandedDimensions(axis: 0) }, axis: 0)
        let g = w.ffnGateShexp(norm)
        let u = w.ffnUpShexp(norm)
        let mid = DS4Math.swigluClamped(gate: g.asType(.float32), up: u.asType(.float32),
                                        clamp: clamp)
        let shared = w.ffnDownShexp(mid)
        let ffnOut = moe + shared
        return DS4Math.hcPost(blockOut: ffnOut, hc: hc, post: split.post, comb: split.comb)
    }

    /// Router probabilities sqrt(softplus(logit)) and the six selections with
    /// normalized, expertWeightsScale-scaled weights (ds4.c:12326-12431).
    func routerSelection(norm: MLXArray, w: DS4LayerWeights, ids: [Int])
        -> (sel: MLXArray, weights: MLXArray) {
        let logits = matmul(norm, w.ffnGateInp.asType(.float32).transposed()) // [T, 256]
        let probs = sqrt(DS4Math.softplus(logits))
        let sel: MLXArray
        if let tid = w.ffnGateTid2Eid {
            // Table row per token: selected[i] = tid2eid[token, i] in table
            // order (ds4.c:12307-12324; MLX layout [vocabSize, 6]).
            let idArr = MLXArray(ids.map { Int32($0) })
            sel = tid.asType(.int32)[idArr] // [T, 6]
        } else {
            var selection = probs
            if let bias = w.expProbsBias {
                selection = selection + bias.asType(.float32)
            }
            sel = DS4Math.topKDescending(selection, k: cfg.expertUsedCount)
        }
        var wts = takeAlong(probs, sel, axis: -1) // [T, 6]
        let sum = maximum(wts.sum(axis: -1, keepDims: true),
                          MLXArray(DS4Constants.routerNormMin))
        wts = wts / sum * MLXArray(cfg.expertWeightsScale)
        return (sel, wts)
    }

    /// One token's routed experts through the MXFP4 3D gather contract. Rows
    /// keep selection order and the stacked expert pieces follow it, so
    /// rhsIndices is 0..<k (sorted, as sortedIndices requires). The router
    /// weight is applied to the SwiGLU output before the down projection
    /// (ds4.c:12568-12583).
    func routedExperts(x: MLXArray, sel: MLXArray, routeW: MLXArray, w: DS4LayerWeights,
                       layer: Int) throws -> MLXArray {
        let k = cfg.expertUsedCount
        precondition(sel.dim(0) == k, "selection must have \(k) entries")
        let ids = sel.asArray(Int32.self).map { Int($0) }
        let pieces = try experts.readBatch(ids.map { ExpertKey(layer, $0) })
        precondition(pieces.count == DS4ExpertStore.pieceCount,
                     "expert record must have \(DS4ExpertStore.pieceCount) pieces")

        let xg = broadcast(x.reshaped([1, 1, cfg.embeddingLength]),
                           to: [k, 1, cfg.embeddingLength])
        let gw = pieces[0] // [k, ff, E/8] u32
        let gs = pieces[1] // [k, ff, E/32] u8
        let uw = pieces[2]
        let us = pieces[3]
        let dw = pieces[4] // [k, E, ff/8] u32
        let ds = pieces[5]
        let ff = cfg.expertFeedForwardLength
        precondition(gw.dim(1) == ff && gw.dim(2) == cfg.embeddingLength / 8,
            "gate w piece is [\(gw.shape[1...])], expected [\(ff), \(cfg.embeddingLength / 8)]")
        precondition(gs.dim(2) == cfg.embeddingLength / 32,
            "gate scales piece is [\(gs.shape[1...])], expected [\(ff), \(cfg.embeddingLength / 32)]")
        precondition(dw.dim(1) == cfg.embeddingLength && dw.dim(2) == ff / 8,
            "down w piece is [\(dw.shape[1...])], expected [\(cfg.embeddingLength), \(ff / 8)]")

        let ridx = MLXArray((0..<k).map { Int32($0) })
        let g = gatherQuantizedMM(xg, gw, scales: gs, biases: nil, rhsIndices: ridx,
                                  transpose: true, groupSize: 32, bits: 4, mode: .mxfp4,
                                  sortedIndices: true) // [k, 1, ff]
        let u = gatherQuantizedMM(xg, uw, scales: us, biases: nil, rhsIndices: ridx,
                                  transpose: true, groupSize: 32, bits: 4, mode: .mxfp4,
                                  sortedIndices: true)
        let mid = DS4Math.swigluClamped(gate: g, up: u, clamp: cfg.swigluClampExp[layer])
        let weighted = mid * routeW.reshaped([k, 1, 1])
        let d = gatherQuantizedMM(weighted, dw, scales: ds, biases: nil, rhsIndices: ridx,
                                  transpose: true, groupSize: 32, bits: 4, mode: .mxfp4,
                                  sortedIndices: true) // [k, 1, E]
        return d.sum(axis: 0).reshaped([cfg.embeddingLength])
    }

    /// The same computation as `routedExperts` for one token, with the
    /// selected experts mapped into the shared slot pool and gathered out of
    /// it (the decode path). The pool holds the same quantized bytes the
    /// checkpoint does — the golden-equivalence invariant — so the only
    /// difference from the staging path is where the weight rows live.
    ///
    /// `sortedIndices` is deliberately NOT set here, unlike the staging call
    /// above: that flag promises ascending rhs indices (the staging path's
    /// `0..<k` satisfies it), while pool slot indices are arbitrary. Passing
    /// it over unsorted indices would be a false hint to the grouped kernel;
    /// the unsorted form is the contract Qwen's `MoELayer.cached` decode
    /// gather has always run on.
    func routedExpertsViaPool(x: MLXArray, sel: MLXArray, routeW: MLXArray,
                              w: DS4LayerWeights, layer: Int) throws -> MLXArray {
        let k = cfg.expertUsedCount
        precondition(sel.dim(0) == k, "selection must have \(k) entries")
        let keys = sel.asArray(Int32.self).map { ExpertKey(layer, Int($0)) }
        let slotOf = pool.ensure(keys) // pins until the next layer's unpinAll
        let slotIdx = MLXArray(slotOf.map { Int32($0) })

        let xg = broadcast(x.reshaped([1, 1, cfg.embeddingLength]),
                           to: [k, 1, cfg.embeddingLength])
        let p = pool.pools // gate w, gate scales, up w, up scales, down w, down scales
        let g = gatherQuantizedMM(xg, p[0], scales: p[1], biases: nil, rhsIndices: slotIdx,
                                  transpose: true, groupSize: 32, bits: 4, mode: .mxfp4)
        let u = gatherQuantizedMM(xg, p[2], scales: p[3], biases: nil, rhsIndices: slotIdx,
                                  transpose: true, groupSize: 32, bits: 4, mode: .mxfp4)
        let mid = DS4Math.swigluClamped(gate: g, up: u, clamp: cfg.swigluClampExp[layer])
        let weighted = mid * routeW.reshaped([k, 1, 1])
        let d = gatherQuantizedMM(weighted, p[4], scales: p[5], biases: nil, rhsIndices: slotIdx,
                                  transpose: true, groupSize: 32, bits: 4, mode: .mxfp4)
        return d.sum(axis: 0).reshaped([cfg.embeddingLength])
    }
}
