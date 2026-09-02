// Prefill + decode loop with sampling, stop tokens, and streaming callbacks.

import Foundation
import MLX

public struct SampleParams {
    public var temperature: Float = 0.7
    public var topP: Float = 0.8
    public var topK: Int = 20
    public var minP: Float = 0
    public var presencePenalty: Float = 1.5
    public var seed: UInt64? = nil
    public var maxTokens = 512
    /// Text sequences that end generation (Ollama `options.stop`, OpenAI `stop`).
    public var stop: [String] = []

    public init() {}

    /// Clamp every knob into the range the sampler is defined on.
    ///
    /// Values outside it used to produce silent garbage rather than an error:
    /// a `top_p` of 0 or a `min_p` above 1 filters out every candidate, and the
    /// old `probs / probs.sum()` then divided 0 by 0, so the sampler emitted
    /// token 0 forever. A negative `num_predict` (Ollama's "until EOS") indexed
    /// a reversed Range and trapped, killing the process.
    public func sanitized() -> SampleParams {
        var p = self
        if !p.temperature.isFinite { p.temperature = 0 }
        p.temperature = max(0, p.temperature)
        if !p.topP.isFinite || p.topP <= 0 || p.topP > 1 { p.topP = 1 }
        if !p.minP.isFinite { p.minP = 0 }
        p.minP = min(max(0, p.minP), 1)
        if !p.presencePenalty.isFinite { p.presencePenalty = 0 }
        p.topK = max(0, p.topK)
        // <= 0 means "as many as allowed" for Ollama (-1) and OpenAI clients.
        if p.maxTokens <= 0 { p.maxTokens = SampleParams.maxTokenCeiling }
        p.maxTokens = min(p.maxTokens, SampleParams.maxTokenCeiling)
        p.stop = p.stop.filter { !$0.isEmpty }
        return p
    }

    /// Upper bound on a single response. Decode is the slow axis here, so an
    /// unbounded "until EOS" request needs a ceiling that is generous but finite.
    public static let maxTokenCeiling = 262_144

    public static var instruct: SampleParams { SampleParams() }
    public static var thinking: SampleParams {
        var p = SampleParams()
        p.temperature = 1.0
        p.topP = 0.95
        p.presencePenalty = 0
        return p
    }
    public static var greedy: SampleParams {
        var p = SampleParams()
        p.temperature = 0
        p.presencePenalty = 0
        return p
    }
}

public struct GenStats {
    /// Every token in the prompt, whether or not it had to be recomputed.
    /// This is what the Ollama/OpenAI surfaces report as prompt_eval_count.
    public var promptTokens = 0
    /// Prompt tokens actually pushed through the model this request. Equal to
    /// `promptTokens` on a cold prompt; `promptTokens - reusedPrefixTokens`
    /// when the conversation prefix cache matched.
    public var prefillTokens = 0
    /// Prompt tokens served from the retained state of a previous request.
    public var reusedPrefixTokens = 0
    public var prefillSeconds = 0.0
    public var decodeTokens = 0
    public var decodeSeconds = 0.0
    public var expertHitRate = 0.0
    public var ngramRowHits = 0
    public var ngramRowMisses = 0
    /// Speculative decode (MTP): drafts proposed, drafts accepted, and verify
    /// passes run. Zero when the draft head is disabled.
    public var draftedTokens = 0
    public var acceptedDrafts = 0
    public var verifyPasses = 0
    public var draftAcceptRate: Double {
        draftedTokens > 0 ? Double(acceptedDrafts) / Double(draftedTokens) : 0
    }
    /// Whole-process lifetime RSS high-water, including MLX, Swift, mmap pages,
    /// allocator cache, and I/O staging. This is the number memory promises use.
    public var peakMemoryGB = 0.0
    /// MLX-only high-water retained as a diagnostic, never as the RAM gate.
    public var mlxPeakMemoryGB = 0.0
    /// Prefill time split: reading expert records, scattering them into the
    /// pool, and how many records were fetched. Everything else is compute.
    public var prefillIOSeconds = 0.0
    public var prefillScatterSeconds = 0.0
    public var prefillRecords = 0
    /// "stop" (EOS, a stop sequence, or a cancelled stream) or "length".
    public var finishReason = "stop"
    /// The end-of-sequence token that ended generation, when that is why it
    /// ended. Nil for stop sequences, client disconnects, and max tokens —
    /// no single token ends the run there. `stopTokenHex` is filled by
    /// Engine.generate (which owns the tokenizer) as the hex bytes of the
    /// token's text, special-token markers included.
    public var stopTokenId: Int? = nil
    public var stopTokenHex: String? = nil

    public var prefillTPS: Double { prefillSeconds > 0 ? Double(prefillTokens) / prefillSeconds : 0 }
    public var prefixHit: Bool { reusedPrefixTokens > 0 }
    public var decodeTPS: Double { decodeSeconds > 0 ? Double(decodeTokens) / decodeSeconds : 0 }
}

/// Token sampling, split out from the decode loop so it can be exercised on
/// synthetic logits with no checkpoint loaded (`slotstream sampler-golden`)
/// and compared against the numpy reference in `Tools/sampler_ref.py`.
///
/// Order matches HuggingFace's processor chain: presence penalty on raw
/// logits, then temperature, then top-k, then top-p, then min-p.
public struct Sampler {
    public var rngState: UInt64 = 0x9E37_79B9_7F4A_7C15

    public init(seed: UInt64? = nil) {
        if let s = seed { rngState = s == 0 ? 0xDEAD_BEEF : s }
    }

    public mutating func next(
        _ logits: MLXArray, params: SampleParams, generated: Set<Int>
    ) -> Int {
        var l = logits.reshaped([-1]).asType(.float32)
        if params.presencePenalty != 0 && !generated.isEmpty {
            // subtract penalty on already-generated tokens
            let ids = MLXArray(generated.sorted().map { Int32($0) })
            let current = take(l, ids, axis: 0)
            l = putAlong(l, ids, values: current - params.presencePenalty, axis: 0)
        }
        if params.temperature <= 0 {
            return argMax(l).item(Int.self)
        }
        l = l / params.temperature
        if params.topK > 0 && params.topK < l.dim(0) {
            let kth = takeAlong(
                l, argPartition(-l, kth: params.topK - 1)[..<params.topK], axis: 0
            ).min()
            l = which(l .< kth, MLXArray(-Float.infinity), l)
        }
        var probs = softmax(l, axis: -1)
        if params.topP < 1 {
            let order = argSort(-probs)
            let sorted = take(probs, order, axis: 0)
            let cum = cumsum(sorted, axis: 0)
            let keepSorted = (cum - sorted) .< params.topP  // keep until cumulative prob (exclusive) reaches topP
            var keep = MLXArray.zeros([probs.dim(0)], dtype: .bool)
            keep = putAlong(keep, order, values: keepSorted, axis: 0)
            probs = which(keep, probs, MLXArray(Float(0)))
        }
        if params.minP > 0 {
            let cutoff = probs.max() * params.minP
            probs = which(probs .< cutoff, MLXArray(Float(0)), probs)
        }
        // gumbel-free categorical: inverse CDF with a splitmix stream.
        // The draw is scaled by the unnormalized total instead of normalizing
        // the probabilities: it avoids a 0/0 when a filter empties the
        // candidate set, and since u < 1 it also guarantees u*total < total,
        // so the pick can never run off the end of the CDF onto a
        // zero-probability token the way a bare `cdf .< u` could.
        rngState = Splitmix.mix(rngState &+ 1)
        let u = Float(Double(rngState >> 11) / Double(1 << 53))
        let cdf = cumsum(probs, axis: 0)
        let total = cdf[probs.dim(0) - 1].item(Float.self)
        guard total.isFinite, total > 0 else {
            // Nothing survived filtering (or the logits were NaN): fall back to
            // the most likely token rather than emitting token 0 forever.
            return argMax(logits.reshaped([-1]).asType(.float32)).item(Int.self)
        }
        let pick = (cdf .< MLXArray(u * total)).sum().item(Int.self)
        return min(pick, probs.dim(0) - 1)
    }
}

/// Windowed decode meter. Each "round" is a batch of emitted tokens: one
/// per plain-loop iteration, or one verify-pass burst in the MTP path.
/// The live decode line uses the last <= `window` emitted tokens instead
/// of the whole-run average, so the rate the user sees matches what the
/// generator is actually doing right now rather than the average across a
/// long first-pass.
fileprivate final class DecodeMeter {
    let window: Int
    private var timestamps: [TimeInterval] = []
    private var cumEmitted: [Int] = []      // emitted count at end of each round
    private var cumDrafted: [Int] = []      // cumulative drafted tokens
    private var cumAccepted: [Int] = []     // cumulative accepted drafts
    private let t0: TimeInterval

    init(window: Int, start: Date) {
        self.window = max(1, window)
        self.t0 = start.timeIntervalSinceReferenceDate
    }

    func record(emitted: Int, drafted: Int = 0, accepted: Int = 0) {
        let now = Date().timeIntervalSinceReferenceDate
        timestamps.append(now)
        cumEmitted.append((cumEmitted.last ?? 0) + emitted)
        cumDrafted.append((cumDrafted.last ?? 0) + drafted)
        cumAccepted.append((cumAccepted.last ?? 0) + accepted)
    }

    /// (tps, draftedInWindow, acceptedInWindow, emittedInWindow, mtpOn)
    func snapshot() -> (tps: Double, drafted: Int, accepted: Int, emitted: Int, mtp: Bool) {
        let n = cumEmitted.count
        guard n > 0 else { return (0, 0, 0, 0, false) }
        // Walk back from the newest round, accumulating emitted until adding
        // the next round would push the window past `window`. Always include
        // the newest round so the meter is never silent.
        var e = 0
        var d = 0
        var a = 0
        var firstIdx = n - 1
        for i in stride(from: n - 1, through: 0, by: -1) {
            let delta = cumEmitted[i] - (i > 0 ? cumEmitted[i - 1] : 0)
            if e > 0 && e + delta > window { break }
            e += delta
            d += i > 0 ? cumDrafted[i] - cumDrafted[i - 1] : cumDrafted[i]
            a += i > 0 ? cumAccepted[i] - cumAccepted[i - 1] : cumAccepted[i]
            firstIdx = i
            if e >= window { break }
        }
        let span: TimeInterval
        if firstIdx == 0 {
            span = timestamps[n - 1] - t0
        } else {
            span = timestamps[n - 1] - timestamps[firstIdx - 1]
        }
        let tps = span > 0 ? Double(e) / span : 0
        return (tps, d, a, e, cumDrafted.last ?? 0 > 0)
    }
}

public final class Generator {
    public let model: Qwen4ExpModel
    /// Tokens per prefill pass. Bigger is faster on long prompts: a chunk
    /// activates nearly every expert of every layer, so the expert stream is
    /// re-read roughly once per chunk and halving the chunk count halves the
    /// bytes moved. It costs transient activation memory, which is why it is a
    /// knob rather than "as large as the prompt". Measured in MEASUREMENTS.md.
    public var prefillChunk = PrefillTuning.chunk
    /// Draft tokens per speculative round when the MTP head is enabled.
    /// The measured accept curve picks the default; SLOTSTREAM_DRAFT_DEPTH
    /// overrides for experiments.
    public var draftDepth: Int = {
        if let s = ProcessInfo.processInfo.environment["SLOTSTREAM_DRAFT_DEPTH"],
            let n = Int(s), n >= 1, n <= 16 { return n }
        return 4
    }()
    /// Gate for the speculative path — `mtp-check` compares speculative
    /// against plain decode on the same loaded model by flipping this.
    public var speculationEnabled = true
    var sampler = Sampler()
    var rngState: UInt64 {
        get { sampler.rngState }
        set { sampler.rngState = newValue }
    }

    public init(model: Qwen4ExpModel) {
        self.model = model
    }

    func sample(_ logits: MLXArray, params: SampleParams, generated: Set<Int>) -> Int {
        sampler.next(logits, params: params, generated: generated)
    }

    /// Runs prefill + decode; calls `onToken` for each generated token id.
    /// Returns (tokenIds, stats). `stop` checked between tokens (cancellation).
    /// `cache`, when given, is consulted for a state this prompt extends and
    /// receives the state back at the end, holding exactly the ids it consumed.
    public func generate(
        promptIds: [Int], params: SampleParams, eosIds: Set<Int>,
        cache: PrefixCache? = nil, visionEmbeds: MLXArray? = nil,
        shouldContinue: (() -> Bool)? = nil,
        onToken: ((Int) -> Bool)? = nil
    ) -> ([Int], GenStats) {
        let params = params.sanitized()
        if let s = params.seed { rngState = s == 0 ? 0xDEAD_BEEF : s }
        var stats = GenStats()
        // An empty prompt would leave `logits` at its placeholder value and make
        // the sampler invent a first token from nothing. Callers reject this at
        // the API boundary; this is the backstop.
        guard !promptIds.isEmpty else { return ([], stats) }
        // Vision prompts are not prefix-cacheable (embeddings vary per image)
        let isVision = visionEmbeds != nil
        let hit = isVision ? nil : cache?.take(
            matching: promptIds, reserveTokens: promptIds.count + params.maxTokens)
        // Disk cache: try longest chunk-aligned prefix on disk before falling back to RAM cache
        var diskState: Qwen4ExpModel.State? = nil
        var diskHitLen: Int? = nil
        var diskParentKey: String? = nil
        var diskParentTokenCount = 0
        if !isVision {
            // Embedding extractor: pull the embedding rows for chunk `d`
            // (tokens [d*prefillChunk, (d+1)*prefillChunk)) on demand. The
            // chain walk bails at the first miss so we never compute past
            // the depth we actually consume.
            let m = model
            let pc = prefillChunk
            let embed: (Int) -> [Float]? = { d in
                let lo = d * pc
                let hi = lo + pc
                guard hi <= promptIds.count else { return nil }
                let ids = MLXArray(promptIds[lo..<hi].map { Int32($0) }, [1, hi - lo])
                let rows = m.resident.embed(ids).asType(.float32)
                eval(rows)
                return rows.reshaped([rows.dim(1) * rows.dim(2)]).asArray(Float.self)
            }
            if let l = DiskCache.longestPrefixHit(chunk: prefillChunk, embed: embed) {
                for d in 0..<(l / pc) {
                    guard let embeddings = embed(d) else { break }
                    diskParentKey = ChunkIndex.makeKey(
                        parentSha: diskParentKey, embeddings: embeddings)
                }
                diskParentTokenCount = l
            }

            // A decoded conversation endpoint is a variable-length child of
            // the deepest fixed boundary. Test the longest possible endpoint
            // first, and only accept one whose content-derived key matches.
            var terminalKey: String? = nil
            var terminalLen: Int? = nil
            for candidate in ChunkIndex.shared.childEndpoints(
                parentSha: diskParentKey, parentTokenCount: diskParentTokenCount,
                before: promptIds.count)
            {
                let lo = diskParentTokenCount
                let hi = candidate.tokenCount
                let ids = MLXArray(promptIds[lo..<hi].map { Int32($0) }, [1, hi - lo])
                let rows = m.resident.embed(ids).asType(.float32)
                eval(rows)
                let embeddings = rows.reshaped([rows.dim(1) * rows.dim(2)]).asArray(Float.self)
                if ChunkIndex.makeKey(
                    parentSha: diskParentKey, embeddings: embeddings) == candidate.key
                {
                    terminalKey = candidate.key
                    terminalLen = hi
                    break
                }
            }

            let fixedLen = diskParentTokenCount
            let loadLen = terminalLen ?? fixedLen
            if loadLen > 0 {
                let depth = fixedLen / pc
                let curReused = hit?.reused ?? 0
                if loadLen > curReused {
                    var usedLen = loadLen
                    var ds = DiskCache.loadState(
                        for: embed, depth: depth, terminalKey: terminalKey,
                        tokenIds: Array(promptIds[0..<loadLen]),
                        template: model.makeState())
                    // A stale or corrupt terminal must not hide a valid fixed
                    // parent that was already proven by the chain walk.
                    if ds == nil, terminalKey != nil, fixedLen > curReused {
                        usedLen = fixedLen
                        ds = DiskCache.loadState(
                            for: embed, depth: depth,
                            tokenIds: Array(promptIds[0..<fixedLen]),
                            template: model.makeState())
                    }
                    if let ds {
                        diskState = ds
                        diskHitLen = usedLen
                        FileHandle.standardError.write("kvcache disk hit: \(usedLen)/\(promptIds.count) tokens\n".data(using: .utf8)!)
                    }
                }
            }
        }
        let state: Qwen4ExpModel.State
        let reused: Int
        if let ds = diskState, let l = diskHitLen {
            state = ds
            reused = l
            // Return the RAM hit state to the pool if we used disk instead
            if let h = hit { cache?.store(state: h.state, tokens: Array(promptIds[0..<h.reused])) }
        } else {
            state = hit?.state ?? model.makeState()
            reused = hit?.reused ?? 0
        }
        stats.promptTokens = promptIds.count
        stats.reusedPrefixTokens = reused
        MLX.Memory.peakMemory = 0
        // Zero before prefill, not only after: otherwise these carry the
        // previous request's decode phase into this request's prefill split.
        model.pool.resetStats()
        model.ngram.resetStats()

        // ---- prefill in chunks (only the tokens the state has not consumed)
        // With the MTP draft head enabled, every chunk also flows through the
        // head so its attention cache covers the whole prompt: the entry for
        // token i fuses the previous position's multi stream with token i's
        // embedding, keeping the invariant mtp.offset == tokenCount - 1.
        // A state handed back by the cache that a plain-path request built
        // has no draft cache to extend; finish that request plain rather than
        // speculating over a misaligned head (unreachable in serve, where the
        // mode is fixed per process; the A/B tools flip it per request).
        let stateKnowsMTP = reused == 0
            || (state.mtp != nil && state.mtp!.offset == max(0, state.tokenCount - 1))
        // Vision disables speculative decode (needs vision-aware verify)
        let mtpHead: MTPHead? = isVision ? nil : (speculationEnabled && stateKnowsMTP ? model.mtpHead : nil)
        if mtpHead != nil && state.mtp == nil { state.mtp = MTPState() }
        // Vision helpers: map placeholder positions to rows
        let visionPositions: [Int] = isVision ? promptIds.enumerated().filter { $0.element == model.cfg.imageTokenId }.map { $0.offset } : []
        let visionFlat: MLXArray? = {
            guard let v = visionEmbeds else { return nil }
            if v.ndim == 3 { return v.reshaped([v.dim(1), model.cfg.hiddenSize]) }
            return v
        }()
        var t0 = Date()
        let prefillTotal = promptIds.count - reused
        if prefillTotal > 0 {
            let vInfo = isVision ? " (vision \(visionFlat?.dim(0) ?? 0) tokens)" : ""
            FileHandle.standardError.write("prefill start: \(prefillTotal) new / \(promptIds.count) total\(vInfo), chunk \(prefillChunk)\n".data(using: .utf8)!)
        }
        var logits: MLXArray = MLXArray(0)
        var i = reused
        while i < promptIds.count {
            if let keepGoing = shouldContinue, !keepGoing() {
                stats.finishReason = "stop"
                stats.prefillTokens = i - reused
                stats.prefillSeconds = -t0.timeIntervalSinceNow
                stats.peakMemoryGB = ProcessMemory.peakResidentGB
                stats.mlxPeakMemoryGB = Double(MLX.Memory.peakMemory) / 1e9
                return ([], stats)
            }
            let hi = min(i + prefillChunk, promptIds.count)
            let chunk = Array(promptIds[i ..< hi])
            let pct = promptIds.count > 0 ? Int(Double(hi) * 100 / Double(promptIds.count)) : 100
            FileHandle.standardError.write(String(format: "prefill %d/%d (%d%%)  chunk %d..%d\n", hi, promptIds.count, pct, i, hi).data(using: .utf8)!)
            // Slice vision rows for this chunk
            let chunkVision: MLXArray? = {
                guard let vf = visionFlat, !visionPositions.isEmpty else { return nil }
                var startIdx: Int? = nil, endIdx: Int? = nil
                for (idx, pos) in visionPositions.enumerated() where pos >= i && pos < hi {
                    if startIdx == nil { startIdx = idx }
                    endIdx = idx
                }
                guard let s = startIdx, let e = endIdx else { return nil }
                let n = e - s + 1
                if n == vf.dim(0) { return vf }
                // slice rows s..<e+1
                return vf[s..<(e+1), 0...]
            }()
            if let head = mtpHead {
                let (mixed, multi) = model.hiddenStatesWithMulti(chunk, state: state, visionEmbeds: chunkVision)
                state.lastMulti = head.consume(
                    chunk: chunk, chunkMulti: multi, prevMulti: state.lastMulti,
                    resident: model.resident, rope: model.rope, state: state.mtp!)
                if hi == promptIds.count {
                    logits = model.lmHead(mixed[0..., (mixed.dim(1) - 1)..., 0...])
                    eval(logits)
                } else {
                    eval(mixed)
                }
            } else if hi == promptIds.count {
                logits = model.lastLogits(chunk, state: state, visionEmbeds: chunkVision)
                eval(logits)
            } else {
                let h = model.hiddenStates(chunk, state: state, visionEmbeds: chunkVision)
                eval(h)
            }
            // Persist a chunk-aligned prefix for disk reuse. Done after the
            // chunk compute so the state it captures is exactly what a future
            // hit would reload. log() inside saveAsync tells us if/when it
            // actually wrote — silent failure here is not acceptable.
            if !isVision, DiskCache.enabled, hi % prefillChunk == 0, hi < promptIds.count {
                let depth = hi / prefillChunk
                // Per-chunk embedding hash (not cumulative). The chain walk in
                // ChunkIndex produces `makeKey(parent: chain[d-1], embeddings: embed(d))`,
                // so the save side has to match that shape — parent_sha comes
                // from walking depths 0..<depth, the chunk embeddings are
                // exactly the rows for tokens [depth-1 .. depth].
                let chunkLo = (depth - 1) * prefillChunk
                let chunkHi = chunkLo + prefillChunk
                let chunkIds = MLXArray(
                    promptIds[chunkLo..<chunkHi].map { Int32($0) },
                    [1, prefillChunk])
                let chunkRows = model.resident.embed(chunkIds).asType(.float32)
                eval(chunkRows)
                let chunkEmbeds = chunkRows
                    .reshaped([chunkRows.dim(1) * chunkRows.dim(2)])
                    .asArray(Float.self)
                var parentSha: String? = nil
                for d in 0..<(depth - 1) {
                    let lo = d * prefillChunk
                    let hi2 = lo + prefillChunk
                    let ids2 = MLXArray(
                        promptIds[lo..<hi2].map { Int32($0) }, [1, prefillChunk])
                    let r2 = model.resident.embed(ids2).asType(.float32)
                    eval(r2)
                    let e2 = r2.reshaped([r2.dim(1) * r2.dim(2)]).asArray(Float.self)
                    parentSha = ChunkIndex.makeKey(parentSha: parentSha, embeddings: e2)
                }
                let key = ChunkIndex.makeKey(parentSha: parentSha, embeddings: chunkEmbeds)
                DiskCache.saveAsync(
                    state: state, tokenIds: Array(promptIds[0..<hi]),
                    key: key, parentSha: parentSha, depth: depth,
                    embeddings: chunkEmbeds, parentTokenCount: chunkLo)
                diskParentKey = key
                diskParentTokenCount = hi
            }
            i = hi
        }
        stats.prefillTokens = promptIds.count - reused
        stats.prefillSeconds = -t0.timeIntervalSinceNow
        stats.prefillIOSeconds = model.pool.ioSeconds
        stats.prefillScatterSeconds = model.pool.scatterSeconds
        stats.prefillRecords = model.pool.recordsFetched
        model.pool.resetStats()
        model.ngram.resetStats()

        // ---- decode
        var out: [Int] = []
        var generated = Set<Int>()
        var reason = "length"
        // Exactly the ids `state` has consumed, tracked rather than inferred:
        // a token is sampled before it is fed, so both break paths below leave
        // the last one unconsumed and it must not be claimed.
        var consumed = promptIds
        t0 = Date()
        if let head = mtpHead, speculationEnabled, let mtpState = state.mtp {
            speculativeDecode(
                head: head, mtpState: mtpState, state: state, logits: logits,
                params: params, eosIds: eosIds, shouldContinue: shouldContinue,
                onToken: onToken, out: &out, generated: &generated,
                reason: &reason, consumed: &consumed, stats: &stats)
        } else {
            var lastLog = Date()
            for _ in 0 ..< max(0, params.maxTokens) {
                if let keepGoing = shouldContinue, !keepGoing() { reason = "stop"; break }
                let tok = sample(logits, params: params, generated: generated)
                if eosIds.contains(tok) {
                    reason = "stop"; stats.stopTokenId = tok; break
                }
                out.append(tok)
                generated.insert(tok)
                // Throttled decode progress: every 16 tokens or 1s
                if out.count == 1 || out.count % 16 == 0 || Date().timeIntervalSince(lastLog) > 1.0 {
                    let elapsed = -t0.timeIntervalSinceNow
                    let tps = elapsed > 0 ? Double(out.count) / elapsed : 0
                    FileHandle.standardError.write(String(format: "decode %d/%d  %.1f tok/s  elapsed %.1fs\n", out.count, params.maxTokens, tps, elapsed).data(using: .utf8)!)
                    lastLog = Date()
                }
                // The callback stops the run for a stop sequence or a gone client.
                if let cb = onToken, !cb(tok) { reason = "stop"; break }
                logits = model.lastLogits([tok], state: state)
                consumed.append(tok)
                eval(logits)
            }
            // Final decode summary if not already logged
            if out.count > 0 {
                let elapsed = -t0.timeIntervalSinceNow
                FileHandle.standardError.write(String(format: "decode done: %d tokens in %.1fs (%.1f tok/s) reason=%@\n", out.count, elapsed, elapsed > 0 ? Double(out.count)/elapsed : 0, reason).data(using: .utf8)!)
            }
        }
        if !isVision {
            cache?.store(state: state, tokens: consumed)
            // Conversation endpoints are rarely aligned to the prefill pass
            // size. Save the terminal delta from the deepest fixed boundary.
            if DiskCache.enabled, consumed.count > diskParentTokenCount {
                let lo = diskParentTokenCount
                let hi = consumed.count
                let ids = MLXArray(consumed[lo..<hi].map { Int32($0) }, [1, hi - lo])
                let rows = model.resident.embed(ids).asType(.float32)
                eval(rows)
                let embeddings = rows.reshaped([rows.dim(1) * rows.dim(2)]).asArray(Float.self)
                let key = ChunkIndex.makeKey(
                    parentSha: diskParentKey, embeddings: embeddings)
                DiskCache.saveAsync(
                    state: state, tokenIds: consumed, key: key,
                    parentSha: diskParentKey,
                    depth: diskParentTokenCount / prefillChunk + 1,
                    embeddings: embeddings,
                    parentTokenCount: diskParentTokenCount)
            }
        }
        stats.finishReason = reason
        stats.decodeTokens = out.count
        stats.decodeSeconds = -t0.timeIntervalSinceNow
        stats.expertHitRate = model.pool.hitRate
        let expertLookups = model.pool.hits + model.pool.misses
        if expertLookups > 0 {
            let residentPct = 100.0 * Double(model.pool.hits) / Double(expertLookups)
            let prefetchPct = 100.0 * Double(model.pool.prefetchPromotions) / Double(expertLookups)
            FileHandle.standardError.write(String(
                format: "decode experts: %.1f%% resident (%d/%d), %.1f%% prefetched (%d), %d demand loads\n",
                residentPct, model.pool.hits, expertLookups, prefetchPct,
                model.pool.prefetchPromotions, model.pool.recordsFetched
            ).data(using: .utf8)!)
        }
        stats.ngramRowHits = model.ngram.rowHits
        stats.ngramRowMisses = model.ngram.rowMisses
        stats.mlxPeakMemoryGB = Double(MLX.Memory.peakMemory) / 1e9
        stats.peakMemoryGB = ProcessMemory.peakResidentGB
        return (out, stats)
    }
}

extension Generator {
    /// Self-speculative decode with the MTP draft head. One round:
    ///
    ///   1. draft `draftDepth` tokens greedily by chaining the head
    ///      (each step fuses the previous multi stream with the previous
    ///      token's embedding — "scheme A"),
    ///   2. verify them in ONE batched main-model pass (decode is
    ///      kernel-launch-bound at plateau cache sizes, so a k+1-token pass
    ///      costs roughly one token's launches),
    ///   3. sample sequentially from the verified logits with the plain
    ///      loop's exact semantics — same rng draw order, same presence
    ///      penalty evolution, drawing ONLY for tokens the plain loop would
    ///      have sampled, so the sampler stream never desyncs,
    ///   4. reconcile: the verify pass consumed all k+1 tokens; if some were
    ///      rejected, roll the state back (zero-copy checkpoint — recurrent
    ///      arrays are replaced, never mutated; KV rolls back by offset) and
    ///      re-run just the kept tokens. The GDN state cannot be rewound, so
    ///      this rebuild pass is the price of a rejection.
    ///
    /// Every emitted token's logits still come from the main model, so this
    /// changes WHAT computes the logits (batched passes instead of
    /// single-token passes), not the sampling rule. Batch shape changes move
    /// logits within the same floating-point envelope as prefill re-chunking
    /// (see MEASUREMENTS on the prefix cache); `mtp-check` gates on that.
    func speculativeDecode(
        head: MTPHead, mtpState: MTPState, state: Qwen4ExpModel.State,
        logits: MLXArray, params: SampleParams, eosIds: Set<Int>,
        shouldContinue: (() -> Bool)?, onToken: ((Int) -> Bool)?,
        out: inout [Int], generated: inout Set<Int>, reason: inout String,
        consumed: inout [Int], stats: inout GenStats
    ) {
        // The first token comes off the prefill logits exactly like the
        // plain loop's first iteration.
        var pending: Int? = nil
        if params.maxTokens > 0 {
            if let keepGoing = shouldContinue, !keepGoing() { reason = "stop"; return }
            let tok = sample(logits, params: params, generated: generated)
            if eosIds.contains(tok) {
                reason = "stop"; stats.stopTokenId = tok; return
            }
            out.append(tok)
            generated.insert(tok)
            if let cb = onToken, !cb(tok) { reason = "stop"; return }
            pending = tok
        }

        // Throttled decode progress: mirrors the plain-decode loop's log so
        // MTP and plain runs read the same way. The MTP line also surfaces
        // the live draft acceptance rate (accepted / drafted) and the verify
        // pass count — the metrics the speculative path actually moves on.
        // Both lines report a windowed tok/s over the last <= 50 emitted
        // tokens (instead of the whole-run average), so the rate tracks what
        // the generator is doing right now.
        let t0 = Date()
        let meter = DecodeMeter(window: 50, start: t0)
        if out.count > 0 { meter.record(emitted: 1) }   // the first token, if any
        var lastLog = t0
        func logProgress() {
            let s = meter.snapshot()
            let elapsed = -t0.timeIntervalSinceNow
            if s.mtp {
                let acc = s.drafted > 0 ? 100.0 * Double(s.accepted) / Double(s.drafted) : 0
                FileHandle.standardError.write(String(
                    format: "decode %d/%d  %.1f tok/s (last %d)  mtp %d/%d accepted (%.0f%%), %d verify passes  elapsed %.1fs\n",
                    out.count, params.maxTokens, s.tps, s.emitted,
                    s.accepted, s.drafted, acc, stats.verifyPasses, elapsed
                ).data(using: .utf8)!)
            } else {
                FileHandle.standardError.write(String(
                    format: "decode %d/%d  %.1f tok/s (last %d)  elapsed %.1fs\n",
                    out.count, params.maxTokens, s.tps, s.emitted, elapsed
                ).data(using: .utf8)!)
            }
        }
        logProgress()

        while let p = pending, out.count < params.maxTokens {
            if let keepGoing = shouldContinue, !keepGoing() { reason = "stop"; break }
            let ck = state.checkpoint()

            // ---- draft (greedy chain; provisional MTP cache entries)
            var drafts: [Int] = []
            var dMulti = state.lastMulti!
            var dTok = p
            for _ in 0 ..< max(1, draftDepth) {
                let e = model.resident.embed(MLXArray([Int32(dTok)], [1, 1])).asType(.bfloat16)
                let (s, m) = head(embedded: e, hiddenMulti: dMulti, rope: model.rope, state: mtpState)
                let dl = model.lmHead(s)
                dTok = argMax(dl.reshaped([-1]).asType(.float32)).item(Int.self)
                drafts.append(dTok)
                dMulti = m
            }
            stats.draftedTokens += drafts.count

            // ---- one batched verify pass over pending + drafts
            let verifyIds = [p] + drafts
            let (vLogits, vMulti) = model.allLogitsWithMulti(verifyIds, state: state)
            eval(vLogits, vMulti)
            stats.verifyPasses += 1

            // ---- sequential acceptance
            var good = 0  // accepted drafts == generation tokens consumed beyond p
            var nextPending: Int? = nil
            for i in 0 ... drafts.count {
                if out.count >= params.maxTokens { break }  // reason stays "length"
                let tok = sample(
                    vLogits[0..., i ..< (i + 1), 0...], params: params, generated: generated)
                if eosIds.contains(tok) {
                    reason = "stop"; stats.stopTokenId = tok; break
                }
                out.append(tok)
                generated.insert(tok)
                if let cb = onToken, !cb(tok) { reason = "stop"; break }
                if i < drafts.count && tok == drafts[i] {
                    good += 1
                    continue
                }
                nextPending = tok  // the rejection correction, or the bonus token
                break
            }
            stats.acceptedDrafts += good
            meter.record(emitted: good + (nextPending != nil ? 1 : 0),
                         drafted: drafts.count, accepted: good)

            // ---- reconcile the state with what was actually kept
            let keep = [p] + Array(drafts[0 ..< good])
            let passMulti: MLXArray
            if keep.count != verifyIds.count {
                state.restore(ck)
                let (_, rMulti) = model.hiddenStatesWithMulti(keep, state: state)
                eval(rMulti)
                passMulti = rMulti
            } else {
                passMulti = vMulti
            }
            mtpState.trim(to: ck.mtpOffset)
            state.lastMulti = head.consume(
                chunk: keep, chunkMulti: passMulti, prevMulti: ck.lastMulti,
                resident: model.resident, rope: model.rope, state: mtpState)
            consumed.append(contentsOf: keep)
            // The draft cache holds one entry per consumed token except the
            // first. A drift here silently degrades every later draft, so
            // fail loud instead.
            precondition(
                mtpState.offset == state.tokenCount - 1,
                "mtp cache misaligned: \(mtpState.offset) entries at \(state.tokenCount) tokens")
            // Throttled progress: every 16 tokens or 1s, like the plain loop.
            if out.count % 16 == 0 || Date().timeIntervalSince(lastLog) > 1.0 {
                logProgress()
                lastLog = Date()
            }
            pending = nextPending
            if reason == "stop" { break }
        }
        // Final decode summary (the MTP path doesn't take the plain loop's
        // "decode done" branch, so write it here).
        if out.count > 0 {
            let elapsed = -t0.timeIntervalSinceNow
            let s = meter.snapshot()
            let tps = elapsed > 0 ? Double(out.count) / elapsed : 0
            let acc = s.mtp && s.drafted > 0 ? 100.0 * Double(s.accepted) / Double(s.drafted) : 0
            if s.mtp {
                FileHandle.standardError.write(String(
                    format: "decode done: %d tokens in %.1fs (%.1f tok/s, %.1f over last %d)  mtp %d/%d accepted (%.0f%%), %d verify passes  reason=%@\n",
                    out.count, elapsed, tps, s.tps, s.emitted,
                    s.accepted, s.drafted, acc, stats.verifyPasses, reason
                ).data(using: .utf8)!)
            } else {
                FileHandle.standardError.write(String(
                    format: "decode done: %d tokens in %.1fs (%.1f tok/s, %.1f over last %d)  reason=%@\n",
                    out.count, elapsed, tps, s.tps, s.emitted, reason
                ).data(using: .utf8)!)
            }
        }
    }
}

/// Prefill chunking. Overridable so the size can be measured and so a small
/// machine can trade prefill speed for transient memory.
public enum PrefillTuning {
    public static var chunk: Int {
        if let s = ProcessInfo.processInfo.environment["SLOTSTREAM_PREFILL_CHUNK"],
            let n = Int(s), n > 0
        {
            return min(n, 4096)
        }
        return 256
    }
}
