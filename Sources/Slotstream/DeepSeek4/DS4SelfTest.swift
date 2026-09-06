// Synthetic self-tests for the DS4 math kernels. Every check compares the
// MLX implementation against a scalar Swift reference transcribed line-for-
// line from the ds4 C engine (the ds4.c line ranges are noted per reference).
// Run with DS4SelfTest.runAll(); uses tiny tensors only.

import Foundation
import MLX

enum DS4Ref {
    // MARK: ds4.c:11430-11461 (hc_pre: rmsnorm + F16 fn matvec) and 11332-11426

    static func rmsNormNoWeight(_ x: [Float], eps: Float) -> [Float] {
        var ss = 0.0
        for v in x { ss += Double(v) * Double(v) }
        let scale = 1.0 / sqrtf(Float(ss / Double(x.count)) + eps)
        return x.map { $0 * scale }
    }

    static func hcSplit(hc: [Float], hcDim: Int, fn: [Float], scale: [Float], base: [Float],
                        iters: Int = 20) -> (pre: [Float], post: [Float], comb: [Float]) {
        let flat = rmsNormNoWeight(hc, eps: 1e-6)
        var mix = [Float](repeating: 0, count: 24)
        for j in 0..<24 {
            var acc: Float = 0
            for i in 0..<hcDim { acc += flat[i] * fn[i * 24 + j] }
            mix[j] = acc
        }
        return sinkhorn(mix: mix, scale: scale, base: base, iters: iters)
    }

    // MARK: ds4.c:11332-11426 (hc_split_sinkhorn_one) + hc_post addressing

    static func sinkhorn(mix: [Float], scale: [Float], base: [Float], iters: Int = 20)
        -> (pre: [Float], post: [Float], comb: [Float]) {
        var out = [Float](repeating: 0, count: 24)
        for i in 0..<4 {
            out[i] = 1.0 / (1.0 + exp(-(mix[i] * scale[0] + base[i]))) + 1e-6
        }
        for i in 0..<4 {
            out[4 + i] = 2.0 / (1.0 + exp(-(mix[4 + i] * scale[1] + base[4 + i])))
        }
        var c = [Float](repeating: 0, count: 16)
        for dst in 0..<4 {
            var rowMax = -Float.infinity
            for src in 0..<4 {
                let idx = src + dst * 4
                let v = mix[8 + idx] * scale[2] + base[8 + idx]
                c[idx] = v
                if v > rowMax { rowMax = v }
            }
            var rowSum: Float = 0
            for src in 0..<4 {
                c[src + dst * 4] = exp(c[src + dst * 4] - rowMax)
                rowSum += c[src + dst * 4]
            }
            for src in 0..<4 {
                c[src + dst * 4] = c[src + dst * 4] / rowSum + 1e-6
            }
        }
        for src in 0..<4 {
            var sum: Float = 0
            for dst in 0..<4 { sum += c[src + dst * 4] }
            for dst in 0..<4 { c[src + dst * 4] *= 1.0 / (sum + 1e-6) }
        }
        for _ in 1..<iters {
            for dst in 0..<4 {
                var sum: Float = 0
                for src in 0..<4 { sum += c[src + dst * 4] }
                for src in 0..<4 { c[src + dst * 4] *= 1.0 / (sum + 1e-6) }
            }
            for src in 0..<4 {
                var sum: Float = 0
                for dst in 0..<4 { sum += c[src + dst * 4] }
                for dst in 0..<4 { c[src + dst * 4] *= 1.0 / (sum + 1e-6) }
            }
        }
        for i in 0..<16 { out[8 + i] = c[i] }
        // hc_post_one applies comb[dst + src*4] = c[dst + src*4].
        var applied = [Float](repeating: 0, count: 16)
        for dst in 0..<4 {
            for src in 0..<4 { applied[dst * 4 + src] = c[dst + src * 4] }
        }
        return (Array(out[0..<4]), Array(out[4..<8]), applied)
    }

    // MARK: ds4.c:14066-14117 (compressor_pool_decode_state) + 14121-14216 machine

    static func compressorMachine(kvSeq: [[Float]], scSeq: [[Float]], headDim: Int, ratio: Int)
        -> [[Float]] {
        let width = (ratio == 4 ? 2 : 1) * headDim
        let rows = ratio == 4 ? 2 * ratio : ratio
        var stateKv = [Float](repeating: 0, count: rows * width)
        var stateScore = [Float](repeating: -Float.infinity, count: rows * width)
        var pooled: [[Float]] = []
        for pos in 0..<kvSeq.count {
            let posMod = pos % ratio
            let row = ratio == 4 ? ratio + posMod : posMod
            for j in 0..<width {
                stateKv[row * width + j] = kvSeq[pos][j]
                stateScore[row * width + j] = scSeq[pos][j]
            }
            guard (pos + 1) % ratio == 0 else { continue }
            var out = [Float](repeating: 0, count: headDim)
            for j in 0..<headDim {
                var maxScore = -Float.infinity
                if ratio == 4 {
                    for r in 0..<ratio {
                        maxScore = max(maxScore, stateScore[r * width + j])
                        maxScore = max(maxScore, stateScore[(ratio + r) * width + headDim + j])
                    }
                } else {
                    for r in 0..<ratio { maxScore = max(maxScore, stateScore[r * width + j]) }
                }
                if maxScore <= -Float.infinity / 2 { out[j] = 0; continue }
                var denom: Float = 0
                var sum: Float = 0
                if ratio == 4 {
                    for r in 0..<ratio {
                        let wp = exp(stateScore[r * width + j] - maxScore)
                        let wc = exp(stateScore[(ratio + r) * width + headDim + j] - maxScore)
                        denom += wp + wc
                        sum += wp * stateKv[r * width + j]
                        sum += wc * stateKv[(ratio + r) * width + headDim + j]
                    }
                } else {
                    for r in 0..<ratio {
                        let w = exp(stateScore[r * width + j] - maxScore)
                        denom += w
                        sum += w * stateKv[r * width + j]
                    }
                }
                out[j] = denom > 0 ? sum / denom : 0
            }
            pooled.append(out)
            if ratio == 4 {
                var nk = stateKv
                var ns = stateScore
                for r in 0..<ratio {
                    for j in 0..<width {
                        nk[r * width + j] = stateKv[(ratio + r) * width + j]
                        ns[r * width + j] = stateScore[(ratio + r) * width + j]
                    }
                }
                for r in 0..<ratio {
                    for j in 0..<width {
                        nk[(ratio + r) * width + j] = nk[r * width + j]
                        ns[(ratio + r) * width + j] = ns[r * width + j]
                    }
                }
                stateKv = nk
                stateScore = ns
            }
        }
        return pooled
    }

    // MARK: ds4.c:3713-3777 (E2M1 grid, FP4 round trip, Hadamard128)

    static func e2m1Dequant(_ x: Float) -> Float {
        let values: [Float] = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        let sign: Float = x < 0 ? -1 : 1
        let ax = min(abs(x), 6.0)
        var best = 0
        var bestDiff = abs(ax - values[0])
        for i in 1..<8 {
            let diff = abs(ax - values[i])
            if diff < bestDiff || (diff == bestDiff && (i & 1) == 0 && (best & 1) != 0) {
                best = i
                bestDiff = diff
            }
        }
        return sign * values[best]
    }

    static func hadamardFP4RoundTrip(_ x: inout [Float]) {
        var stride = 1
        while stride < 128 {
            var baseIdx = 0
            while baseIdx < 128 {
                var i = 0
                while i < stride {
                    let a = x[baseIdx + i]
                    let b = x[baseIdx + stride + i]
                    x[baseIdx + i] = a + b
                    x[baseIdx + stride + i] = a - b
                    i += 1
                }
                baseIdx += 2 * stride
            }
            stride <<= 1
        }
        for i in 0..<128 { x[i] *= 0.08838834764831845 }
        var off = 0
        while off < 128 {
            var amax: Float = 0
            for i in 0..<32 {
                let av = abs(x[off + i])
                if av > amax { amax = av }
            }
            if amax < 7.052966104933725e-38 { amax = 7.052966104933725e-38 }
            let scale = scalbn(1.0 as Float, Int(ceilf(log2f(amax / 6.0))))
            for i in 0..<32 {
                var v = x[off + i] / scale
                if v > 6.0 { v = 6.0 }
                if v < -6.0 { v = -6.0 }
                x[off + i] = e2m1Dequant(v) * scale
            }
            off += 32
        }
    }

    // MARK: ds4.c:3640-3711 (E4M3FN dequant + compressed-KV row quantize)

    static func e4m3Value(_ i: Int) -> Float {
        var expScale = [Float](repeating: 0, count: 16)
        for e in 1...15 { expScale[e] = scalbn(1.0 as Float, e - 7) }
        let e = (i >> 3) & 0x0F
        let m = i & 0x07
        return e == 0 ? Float(m) * 0.001953125 : (1.0 + Float(m) * 0.125) * expScale[e]
    }

    static func e4m3Dequant(_ x: Float) -> Float {
        let sign: Float = x < 0 ? -1 : 1
        let ax = min(abs(x), 448.0)
        var lo = 0
        var hi = 126
        while lo < hi {
            let mid = (lo + hi + 1) >> 1
            if e4m3Value(mid) <= ax { lo = mid } else { hi = mid - 1 }
        }
        var best = lo
        if best < 126 {
            let bestDiff = abs(ax - e4m3Value(best))
            let nextDiff = abs(ax - e4m3Value(best + 1))
            if nextDiff < bestDiff || (nextDiff == bestDiff && ((best + 1) & 1) == 0 && (best & 1) != 0) {
                best += 1
            }
        }
        return sign * e4m3Value(best)
    }

    /// ds4.c:3693 fp8 row quantize for a 448-wide non-RoPE prefix.
    static func fp8QuantRow(_ x: inout [Float]) {
        var off = 0
        while off < 448 {
            var amax: Float = 0
            for i in 0..<64 {
                let av = abs(x[off + i])
                if av > amax { amax = av }
            }
            if amax < 1.0e-4 { amax = 1.0e-4 }
            let scale = scalbn(1.0 as Float, Int(ceilf(log2f(amax / 448.0))))
            for i in 0..<64 {
                var v = x[off + i] / scale
                if v > 448.0 { v = 448.0 }
                if v < -448.0 { v = -448.0 }
                x[off + i] = e4m3Dequant(v) * scale
            }
            off += 64
        }
    }

    // MARK: ds4.c:11823-11937 (YaRN ramp/corr + rope_tail)

    static func corrDim(_ nDims: Float, _ ctx: Int, _ beta: Float, _ base: Float) -> Float {
        nDims * logf(Float(ctx) / (beta * 2.0 * Float.pi)) / (2.0 * logf(base))
    }

    static func yarnRamp(_ low: Float, _ high: Float, _ i0: Int) -> Float {
        let y = (Float(i0 / 2) - low) / max(0.001, high - low)
        return 1.0 - min(1.0, max(0.0, y))
    }

    static func ropeTail(x: inout [Float], nHead: Int, headDim: Int, nRot: Int, pos: Int,
                         origCtx: Int, base: Float, freqScale: Float, extFactor: Float,
                         attnFactor: Float, betaFast: Float, betaSlow: Float, inverse: Bool) {
        let nNope = headDim - nRot
        let thetaScale = powf(base, -2.0 / Float(nRot))
        let sinSign: Float = inverse ? -1.0 : 1.0
        var corr = [Float](repeating: 0, count: 2)
        if extFactor != 0 {
            corr[0] = max(0.0, floorf(corrDim(Float(nRot), origCtx, betaFast, base)))
            corr[1] = min(Float(nRot - 1), ceilf(corrDim(Float(nRot), origCtx, betaSlow, base)))
        }
        for h in 0..<nHead {
            let off = h * headDim + nNope
            var thetaExtrap = Float(pos)
            var i = 0
            while i < nRot {
                let thetaInterp = freqScale * thetaExtrap
                var theta = thetaInterp
                var mscale = attnFactor
                if extFactor != 0 {
                    let rampMix = yarnRamp(corr[0], corr[1], i) * extFactor
                    theta = thetaInterp * (1.0 - rampMix) + thetaExtrap * rampMix
                    mscale *= 1.0 + 0.1 * logf(1.0 / freqScale)
                }
                let c = cosf(theta) * mscale
                let s = sinSign * sinf(theta) * mscale
                let x0 = x[off + i]
                let x1 = x[off + i + 1]
                x[off + i] = x0 * c - x1 * s
                x[off + i + 1] = x0 * s + x1 * c
                thetaExtrap *= thetaScale
                i += 2
            }
        }
    }

    // MARK: ds4.c:12370-12382 (topk_desc)

    static func topKDesc(_ score: [Float], k: Int) -> [Int] {
        var idx = [Int](repeating: -1, count: k)
        for i in 0..<score.count {
            for j in 0..<k {
                if idx[j] < 0 || score[i] > score[idx[j]] {
                    var m = k - 1
                    while m > j {
                        idx[m] = idx[m - 1]
                        m -= 1
                    }
                    idx[j] = i
                    break
                }
            }
        }
        return idx
    }
}

public enum DS4SelfTest {
    struct Rng {
        var state: UInt64
        init(_ seed: UInt64) { state = seed &* 6364136223846793005 &+ 1442695040888963407 }
        mutating func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float((state >> 40) & 0xFFFFFF) / Float(0xFFFFFF) * 2 - 1
        }
        mutating func magnitude() -> Float {
            scalbn(1.0 as Float, -30 + Int((next() + 1) * 30)) // 2^-30 ..< 1, log-uniform
        }
    }

    static func compare(_ a: MLXArray, _ b: [Float]) -> (maxAbs: Float, maxRel: Float, exact: Double) {
        let av = a.asType(.float32).asArray(Float.self)
        precondition(av.count == b.count, "compare: \(av.count) vs \(b.count) elements")
        var maxAbs: Float = 0
        var maxRel: Float = 0
        var exact = 0
        for i in 0..<av.count {
            let d = abs(av[i] - b[i])
            maxAbs = max(maxAbs, d)
            maxRel = max(maxRel, d / max(abs(b[i]), 1e-30))
            if av[i] == b[i] { exact += 1 }
        }
        return (maxAbs, maxRel, Double(exact) / Double(av.count))
    }

    /// 1. mHC split (rmsnorm -> fn matvec -> sinkhorn, comb in hc_post
    /// addressing): hand-computed zeros case + random end-to-end vs ds4.c.
    static func testSinkhorn() -> Bool {
        var ok = true
        // Hand case: all-zero mix and base, unit scales.
        let hand = DS4Ref.sinkhorn(mix: [Float](repeating: 0, count: 24),
                                   scale: [1, 1, 1], base: [Float](repeating: 0, count: 24))
        if !check(abs(hand.pre[0] - 0.500001) < 1e-6, "sinkhorn hand pre = 0.5+eps") { ok = false }
        if !check(abs(hand.post[0] - 1.0) < 1e-6, "sinkhorn hand post = 1.0") { ok = false }
        if !check(abs(hand.comb[0] - 0.25) < 1e-4, "sinkhorn hand comb = 0.25") { ok = false }

        // Random end-to-end through the real hcSplit (1 token). hcSplit takes
        // fn as [24, 4·embd] (MLX orientation), the scalar ref as [4·embd, 24].
        var rng = Rng(42)
        let embd = 4096
        let hcDim = 4 * embd
        let hc = (0..<hcDim).map { _ in rng.next() * 0.1 }
        let fn = (0..<hcDim * 24).map { _ in rng.next() * 0.02 }
        let fnT = (0..<24 * hcDim).map { j -> Float in
            let row = j / hcDim, col = j % hcDim
            return fn[col * 24 + row]
        }
        let scale = [Float(0.7), Float(1.3), Float(0.9)]
        let base = (0..<24).map { _ in rng.next() * 0.5 }
        let ref = DS4Ref.hcSplit(hc: hc, hcDim: hcDim, fn: fn, scale: scale, base: base)
        let split = DS4Math.hcSplit(hc: MLXArray(hc).reshaped([1, 4, embd]),
                                    fn: MLXArray(fnT).reshaped([24, hcDim]),
                                    scale: MLXArray(scale), base: MLXArray(base),
                                    eps: 1e-6, iters: 20)
        let preC = compare(split.pre, ref.pre)
        let postC = compare(split.post, ref.post)
        let combC = compare(split.comb.reshaped([16]), ref.comb)
        if !check(preC.maxRel < 1e-4, "hcSplit pre (rel \(preC.maxRel))") { ok = false }
        if !check(postC.maxRel < 1e-4, "hcSplit post (rel \(postC.maxRel))") { ok = false }
        if !check(combC.maxRel < 1e-4, "hcSplit comb [dst][src] (rel \(combC.maxRel))") { ok = false }
        return ok
    }

    /// 2. Compressor softmax-pool state machine: 3 groups of ratio 4 on a toy
    /// headDim, MLX vs the ds4.c:14066/14121 transcription (prev plane 0,
    /// current plane 1, rotation after each pool).
    static func testCompressor() -> Bool {
        var ok = true
        var rng = Rng(7)
        let headDim = 8
        let ratio = 4
        let width = 2 * headDim
        var kvSeq: [[Float]] = []
        var scSeq: [[Float]] = []
        for _ in 0..<12 {
            kvSeq.append((0..<width).map { _ in rng.next() })
            scSeq.append((0..<width).map { _ in rng.next() * 3 })
        }
        let ref = DS4Ref.compressorMachine(kvSeq: kvSeq, scSeq: scSeq, headDim: headDim, ratio: ratio)
        var state = DS4CompState()
        var got: [[Float]] = []
        for pos in 0..<12 {
            if let pooled = DS4Model.compressorStep(state: &state, kvCur: MLXArray(kvSeq[pos]),
                                                    scCur: MLXArray(scSeq[pos]), pos: pos,
                                                    headDim: headDim, ratio: ratio) {
                got.append(pooled.asArray(Float.self))
            }
        }
        if !check(got.count == ref.count, "compressor pool count \(got.count) == \(ref.count)") { ok = false }
        for (i, r) in ref.enumerated() where i < got.count {
            let c = compare(MLXArray(got[i]), r)
            if !check(c.maxRel < 1e-5, "compressor pool \(i) (rel \(c.maxRel))") { ok = false }
        }
        return ok
    }

    /// 3. Hadamard128 + FP4 round trip vs ds4.c:3713-3777, and top-k ordering
    /// preservation through Q(iq)·Q(k).
    static func testHadamardFP4() -> Bool {
        var ok = true
        var rng = Rng(11)
        let x = (0..<128).map { _ in rng.next() * 3 }
        var ref = x
        DS4Ref.hadamardFP4RoundTrip(&ref)
        let got = DS4Math.indexerQAT(MLXArray(x).reshaped([1, 128]))
        let c = compare(got, ref)
        if !check(c.maxRel < 0.35, "hadamard+fp4 round trip (rel \(c.maxRel), exact \(c.exact))") { ok = false }

        // Top-k ordering: 1 query, 8 candidates; scores before vs after both
        // sides go through the QAT round trip.
        let q = (0..<128).map { _ in rng.next() * 2 }
        let ks = (0..<8).map { _ in (0..<128).map { _ in rng.next() * 2 } }
        func dot(_ a: [Float], _ b: [Float]) -> Float {
            zip(a, b).map { $0 * $1 }.reduce(0, +)
        }
        let fullScores = ks.map { dot(q, $0) }
        let qQ = DS4Math.indexerQAT(MLXArray(q).reshaped([1, 128])).asArray(Float.self)
        let qatScores = ks.map { dot(qQ, DS4Math.indexerQAT(MLXArray($0).reshaped([1, 128])).asArray(Float.self)) }
        let fullTop = DS4Ref.topKDesc(fullScores, k: 4)
        let qatTop = DS4Ref.topKDesc(qatScores, k: 4)
        // The QAT round trip preserves WHICH rows are selected; the order of
        // near-tied candidates can flip because per-row QAT noise is a few
        // percent of the score.
        if !check(Set(fullTop) == Set(qatTop),
                  "qat top-4 selection preserved: qat \(qatTop) vs full \(fullTop)") { ok = false }
        return ok
    }

    /// 4. E4M3 quantize/dequant round trip vs ds4.c:3663-3711: per-chunk error
    /// <= 2^-3 of the chunk amax, values near-exact.
    static func testE4M3() -> Bool {
        var ok = true
        var rng = Rng(3)
        var vals = [Float](repeating: 0, count: 448)
        for i in 0..<448 {
            vals[i] = rng.magnitude() * (rng.next() > 0 ? 1 : -1)
        }
        vals[0] = 448
        vals[1] = -448
        vals[2] = 256
        vals[3] = 2.0
        vals[4] = 0.001953125
        vals[5] = -0.0009765625
        vals[6] = 1e-30
        var refRow = vals
        DS4Ref.fp8QuantRow(&refRow)
        let x = MLXArray(vals).reshaped([1, 448])
        let (codes, scales) = DS4Math.fp8Encode(x)
        let got = DS4Math.fp8Decode(codes, scales: scales).reshaped([448])
        let gc = got.asArray(Float.self)
        // Per-chunk relative-to-amax error.
        var worstChunk: Float = 0
        var exact = 0
        for chunk in 0..<7 {
            var amax: Float = 0
            for i in 0..<64 { amax = max(amax, abs(vals[chunk * 64 + i])) }
            amax = max(amax, 1e-4)
            for i in 0..<64 {
                let d = abs(gc[chunk * 64 + i] - refRow[chunk * 64 + i])
                worstChunk = max(worstChunk, d / amax)
                if gc[chunk * 64 + i] == refRow[chunk * 64 + i] { exact += 1 }
            }
        }
        if !check(worstChunk <= 0.125, "e4m3 chunk error \(worstChunk) <= 2^-3 of amax") { ok = false }
        if !check(Double(exact) > 440.0, "e4m3 near-exact (\(exact)/448)") { ok = false }
        return ok
    }

    /// 5. rope_tail: dense and compressed params at positions across the
    /// YaRN range, MLX vs ds4.c:11823-11937, plus inverse round trip.
    static func testRopeTail() -> Bool {
        var ok = true
        var rng = Rng(19)
        let nHead = 2
        let headDim = 512
        let nRot = 64
        let positions = [0, 5, 70000]
        var base = [Float](repeating: 0, count: positions.count * nHead * headDim)
        for i in 0..<base.count { base[i] = rng.next() }

        for params in [(base: Float(10000), fs: Float(1.0), ext: Float(0)),
                       (base: Float(160000), fs: Float(1.0 / 16.0), ext: Float(1))] {
            let attn: Float = params.ext != 0 ? 1.0 / (1.0 + 0.1 * logf(1.0 / params.fs)) : 1.0
            let p = DS4Math.RopeParams.make(base: params.base, freqScale: params.fs,
                                            extFactor: params.ext, origCtx: 65536,
                                            betaFast: 32, betaSlow: 1, nRot: nRot)
            var worst: Float = 0
            var worstRound: Float = 0
            for (t, pos) in positions.enumerated() {
                let lo = t * nHead * headDim
                let row = Array(base[lo..<(lo + nHead * headDim)])
                var ref = row
                DS4Ref.ropeTail(x: &ref, nHead: nHead, headDim: headDim, nRot: nRot, pos: pos,
                                origCtx: 65536, base: params.base, freqScale: params.fs,
                                extFactor: params.ext, attnFactor: attn,
                                betaFast: 32, betaSlow: 1, inverse: false)
                let xArr = MLXArray(row).reshaped([1, nHead, headDim])
                let got = DS4Math.ropeTail(xArr, positions: MLXArray([Int32(pos)]), p: p, inverse: false)
                let d = compare(got, ref)
                worst = max(worst, d.maxAbs)
                let back = DS4Math.ropeTail(got, positions: MLXArray([Int32(pos)]), p: p, inverse: true)
                let dr = compare(back, row)
                worstRound = max(worstRound, dr.maxRel)
            }
            // sin/cos of arguments up to ~70000 rad differ by a few ULP between
            // Metal and libm; the theta chain itself is bit-matched.
            let tol: Float = params.ext != 0 ? 0.05 : 0.002
            if !check(worst < tol, "ropeTail base \(params.base) (maxAbs \(worst))") { ok = false }
            if !check(worstRound < 0.05, "ropeTail inverse base \(params.base) (rel \(worstRound))") { ok = false }
        }
        return ok
    }

    /// 6. Top-6 descending with ties to the lower index, vs ds4.c:12370.
    static func testTopK() -> Bool {
        var ok = true
        let cases: [[Float]] = [
            [5, 5, 1, 3, 2, 2, 2],
            [2, 2, 2, 9, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1, -1],
            [0.5, -0.5, 0.5, 3, 3, 1, 2],
        ]
        for s in cases {
            let ref = DS4Ref.topKDesc(s, k: 6)
            let got = DS4Math.topKDescending(MLXArray(s).reshaped([1, s.count]), k: 6)
                .reshaped([6]).asArray(Int32.self).map { Int($0) }
            if !check(got == ref, "topK \(s) -> got \(got) want \(ref)") { ok = false }
        }
        return ok
    }

    static func check(_ cond: Bool, _ name: String) -> Bool {
        print(cond ? "  ok   \(name)" : "  FAIL \(name)")
        return cond
    }

    /// Runs every self-test; returns true when all pass.
    @discardableResult
    public static func runAll() -> Bool {
        print("DS4SelfTest:")
        var ok = true
        if !testSinkhorn() { ok = false }
        if !testCompressor() { ok = false }
        if !testHadamardFP4() { ok = false }
        if !testE4M3() { ok = false }
        if !testRopeTail() { ok = false }
        if !testTopK() { ok = false }
        print(ok ? "DS4SelfTest: PASS" : "DS4SelfTest: FAIL")
        return ok
    }
}
