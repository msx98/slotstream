// DeepSeek-V4-Flash math kernels, transcribed from the ds4 C engine (CPU
// reference ds4.c:11330-15300, Metal kernels metal/*.metal). All activations
// are F32; attention Q/K/V dots are F16; compressed rows are E4M3-encoded with
// power-of-two chunk scales; indexer rows go through the Hadamard/FP4 QAT
// round trip.

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - scalars and norms

public enum DS4Math {
    public static let negInf = -Float.infinity

    public static func rmsNormNoWeight(_ x: MLXArray, eps: Float) -> MLXArray {
        let f = x.asType(.float32)
        return f * rsqrt(f.square().mean(axis: -1, keepDims: true) + eps)
    }

    public static func rmsNorm(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: weight.asType(.float32), eps: eps)
    }

    /// Per-head unweighted RMSNorm over the last dimension split into heads
    /// (ds4.c:8386 head_rms_norm_inplace).
    public static func perHeadRmsNorm(_ x: MLXArray, eps: Float) -> MLXArray {
        rmsNormNoWeight(x, eps: eps)
    }

    /// ds4.c:12164 softplus_stable.
    public static func softplus(_ x: MLXArray) -> MLXArray {
        which(x .> 20, x, which(x .< -20, exp(x), log1p(exp(x))))
    }

    /// ds4.c:12170 swiglu: gate clamped above only, up clamped both sides.
    public static func swigluClamped(gate: MLXArray, up: MLXArray, clamp limit: Float) -> MLXArray {
        let g = minimum(gate, MLXArray(limit))
        let u = minimum(maximum(up, MLXArray(-limit)), MLXArray(limit))
        return MLXNN.silu(g) * u
    }

    // MARK: round-half-even and power-of-two scales

    /// Round-half-even on non-negative values (exact: floor, subtract, compare).
    public static func roundHalfEven(_ x: MLXArray) -> MLXArray {
        let f = floor(x)
        let r = x - f
        let up = f + 1
        let fOdd = (floor(f / 2) * 2 .!= f)
        return which(r .> 0.5, up, which(r .< 0.5, f, which(fOdd, up, f)))
    }

    /// Exact 2^e for integer e (LUT + take; comparisons and ·2 stay exact).
    public static let expLUT: MLXArray = {
        var v = [Float]()
        v.reserveCapacity(189)
        for e in -140...48 { v.append(scalbn(Float(1), e)) }
        return MLXArray(v)
    }()
    public static let expLUTOffset = 140

    private static func takePow2(_ e: MLXArray) -> MLXArray {
        let k = minimum(maximum(e + MLXArray(Int32(expLUTOffset)), MLXArray(0)),
                        MLXArray(Int32(expLUT.count - 1)))
        return expLUT[k]
    }

    /// Smallest power of two >= r, r > 0 (ds4.c:3703/3760 scale formula).
    /// The exponent comes from log2 (±1 ulp); the exact-power corrections via
    /// ·2 / ÷2 and comparison make the result exact regardless.
    public static func pow2Ceil(_ r: MLXArray) -> MLXArray {
        let k = ceil(log2(r)).asType(.int32)
        var s = takePow2(k)
        s = which(s .< r, s * 2, s)
        s = which(s / 2 .>= r, s / 2, s)
        return s
    }

    // MARK: E4M3FN (compressed KV payload)

    /// Signed E4M3FN value table [256] (ds4.c:3640-3661); code 127 is unused.
    public static let e4m3LUT: MLXArray = {
        var expScale = [Float](repeating: 0, count: 16)
        for e in 1...15 { expScale[e] = scalbn(Float(1), e - 7) }
        var v = [Float](repeating: 0, count: 256)
        for b in 0..<256 {
            let sign: Float = b >= 128 ? -1 : 1
            let i = b & 0x7F
            guard i <= 126 else { continue } // 127 stays 0; never produced
            let e = (i >> 3) & 0x0F
            let m = i & 0x07
            let mag: Float = e == 0 ? Float(m) * 0.001953125 : (1 + Float(m) * 0.125) * expScale[e]
            v[b] = sign * mag
        }
        return MLXArray(v)
    }()

    /// Encode [..., 448] F32 into E4M3FN codes plus per-64-chunk F32 scales
    /// (amax floor 1e-4, scale 2^ceil(log2(amax/448)), clamp ±448; grid
    /// nearest with ties to the even grid code, ds4.c:3663-3711).
    public static func fp8Encode(_ x: MLXArray) -> (codes: MLXArray, scales: MLXArray) {
        precondition(x.dim(-1) % 64 == 0, "fp8Encode needs 64-aligned rows")
        let lead = Array(x.shape.dropLast())
        let chunks = x.reshaped(lead + [-1, 64])
        let amax = maximum(chunks.abs().max(axis: -1, keepDims: true), MLXArray(Float(1e-4)))
        let scale = pow2Ceil(amax / 448) // [..., chunks, 1]
        let v = minimum(maximum(chunks / scale, MLXArray(-Float(448))), MLXArray(Float(448)))
        let sign = v .< 0
        let ax = minimum(v.abs(), MLXArray(Float(448)))
        let axSafe = maximum(ax, MLXArray(Float(1e-30)))

        var g = floor(log2(axSafe)).asType(.int32)
        g = which(ax .< takePow2(g), g - 1, g)
        g = which(ax .>= takePow2(g + 1), g + 1, g)
        let gClamped = minimum(maximum(g, MLXArray(-9)), MLXArray(8))

        let isSub = g .< -6
        let subCode = roundHalfEven(ax * 512) // [0, 8]; 8 == the normal code at 2^-6

        let p2 = takePow2(gClamped)        // 2^g
        let step = takePow2(gClamped - 3)  // 2^(g-3)
        var m = roundHalfEven((ax - p2) / step)
        let carry = m .>= 8
        let g2 = which(carry, gClamped + 1, gClamped)
        m = which(carry, MLXArray(Float(0)), m)
        let normCode = (g2 + 7) * 8 + m.asType(.int32)

        var code = which(isSub, subCode.asType(.int32), normCode)
        code = minimum(maximum(code, MLXArray(0)), MLXArray(126))
        let signed = code + which(sign, MLXArray(Int32(128)), MLXArray(Int32(0)))
        let codes = signed.asType(.uint8).reshaped(lead + [x.dim(-1)])
        let scales = scale.reshaped(lead + [-1])
        return (codes, scales)
    }

    /// Dequantize codes [..., 448] uint8 with scales [..., chunks] to F32.
    public static func fp8Decode(_ codes: MLXArray, scales: MLXArray) -> MLXArray {
        let vals = e4m3LUT[codes.asType(.int32)]
        let lead = Array(vals.shape.dropLast())
        let v = vals.reshaped(lead + [-1, 64])
        let sc = scales.asType(.float32).reshaped(scales.shape + [1])
        return (v * sc).reshaped(lead + [codes.dim(-1)])
    }

    /// E4M3 round trip on the non-RoPE dims of a [..., headDim] row batch;
    /// RoPE dims pass through (ds4.c:3693, applied before the F16 store).
    public static func fp8RoundTrip(_ x: MLXArray, rotDim: Int) -> MLXArray {
        let lead = Array(x.shape.dropLast())
        let nope = x[.ellipsis, 0..<(x.dim(-1) - rotDim)]
        let ropeDims = x[.ellipsis, (x.dim(-1) - rotDim)...]
        let (codes, scales) = fp8Encode(nope)
        let rt = fp8Decode(codes, scales: scales).asType(.float16)
        return concatenated([rt, ropeDims.asType(.float16)], axis: -1).reshaped(lead + [x.dim(-1)])
    }

    // MARK: E2M1 + Hadamard (indexer QAT)

    public static let e2m1Grid: MLXArray = MLXArray([Float(0), 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    public static let e2m1EvenMask: MLXArray = MLXArray([true, false, true, false, true, false, true, false])

    /// FP4 activation round trip, per-32 amax (floor 7.05e-38), scale
    /// 2^ceil(log2(amax/6)), clamp ±6, nearest grid value with ties to the
    /// even grid index (ds4.c:3713-3767).
    public static func fp4RoundTrip(_ x: MLXArray) -> MLXArray {
        precondition(x.dim(-1) % 32 == 0, "fp4RoundTrip needs 32-aligned rows")
        let lead = Array(x.shape.dropLast())
        let r = x.reshaped(lead + [-1, 32])
        let amax = maximum(r.abs().max(axis: -1, keepDims: true),
                           MLXArray(Float(7.052966104933725e-38)))
        let scale = pow2Ceil(amax / 6)
        let v = minimum(maximum(r / scale, MLXArray(-Float(6))), MLXArray(Float(6)))
        // Nearest grid value of |v| (grid is non-negative); sign reapplied after.
        let ax = v.abs()
        let diff = (ax.expandedDimensions(axis: -1) - e2m1Grid).abs() // [..., 32, 8]
        let best = diff.min(axis: -1, keepDims: true)
        let tie = diff .== best
        let idx = MLXArray((0..<8).map { Int32($0) }) // [8] broadcasts
        let big = MLXArray(Int32(8))
        let minAny = which(tie, idx, big).min(axis: -1)
        let minEven = which(tie .&& e2m1EvenMask, idx, big).min(axis: -1)
        let pick = which(minEven .< 8, minEven, minAny) // [..., 32] i32
        let q = e2m1Grid[pick]
        let sgn = which(v .< 0, MLXArray(-Float(1)), MLXArray(Float(1)))
        return (q * sgn * scale).reshaped(lead + [x.dim(-1)])
    }

    /// 128-wide Hadamard butterfly (stride 1..64, a±b), scaled by 1/sqrt(128)
    /// (ds4.c:3735; identical order in metal/dsv4_kv.metal:180-191).
    public static func hadamard128(_ x: MLXArray) -> MLXArray {
        precondition(x.dim(-1) == 128, "hadamard128 needs 128-wide rows")
        let lead = Array(x.shape.dropLast())
        var v = x
        var stride = 1
        while stride < 128 {
            let g = v.reshaped(lead + [-1, 2, stride])
            let a = g[.ellipsis, 0, 0...]
            let b = g[.ellipsis, 1, 0...]
            v = concatenated([a + b, a - b], axis: -1).reshaped(lead + [128])
            stride <<= 1
        }
        return v * MLXArray(Float(0.08838834764831845))
    }

    /// Indexer quantization-aware training round trip (ds4.c:3774).
    public static func indexerQAT(_ x: MLXArray) -> MLXArray {
        fp4RoundTrip(hadamard128(x))
    }

    // MARK: RoPE (tail-only, adjacent pairs)

    /// Per-layer RoPE constants, precomputed on the CPU with the same Float32
    /// operations as ds4.c:11823-11937.
    public struct RopeParams {
        public let thetaScale: Float
        public let freqScale: Float
        public let ramp: [Float] // per pair-dim, pre-multiplied by extFactor (0 when none)
        public let extFactor: Float
        public let mscale: Float
        public let nRot: Int

        public static func make(base: Float, freqScale: Float, extFactor: Float,
                         origCtx: Int, betaFast: Float, betaSlow: Float, nRot: Int) -> RopeParams {
            let thetaScale = powf(base, -2.0 / Float(nRot))
            var ramp = [Float](repeating: 0, count: nRot / 2)
            var attn: Float = 1.0
            if extFactor != 0 {
                let low = max(0.0, floorf(corrDim(Float(nRot), origCtx, betaFast, base)))
                let high = min(Float(nRot - 1), ceilf(corrDim(Float(nRot), origCtx, betaSlow, base)))
                for i in 0..<(nRot / 2) {
                    let y = (Float(i) - low) / max(Float(0.001), high - low)
                    ramp[i] = (1.0 - min(1.0, max(0.0, y))) * extFactor
                }
                attn = 1.0 / (1.0 + 0.1 * logf(1.0 / freqScale))
            }
            let mscale: Float = extFactor != 0 ? attn * (1 + 0.1 * logf(1.0 / freqScale)) : attn
            return RopeParams(thetaScale: thetaScale, freqScale: freqScale, ramp: ramp,
                              extFactor: extFactor, mscale: mscale, nRot: nRot)
        }

        private static func corrDim(_ nDims: Float, _ ctx: Int, _ beta: Float, _ base: Float) -> Float {
            nDims * logf(Float(ctx) / (beta * 2.0 * Float.pi)) / (2.0 * logf(base))
        }
    }

    /// Rotate only the last `nRot` dims of every head, adjacent pairs,
    /// extrapolated chain pos·thetaScale^j mixed with the YaRN ramp
    /// (ds4.c:11842-11890). `x` is [T, H, D] F32; positions [T] Int32.
    public static func ropeTail(_ x: MLXArray, positions: MLXArray, p: RopeParams, inverse: Bool) -> MLXArray {
        let t = x.dim(0)
        let h = x.dim(1)
        let d = x.dim(2)
        let nRot = p.nRot
        let nNope = d - nRot
        let half = nRot / 2

        let posCol = positions.asType(.float32).reshaped([t, 1])
        let rest = MLXArray.full([t, half - 1], values: MLXArray(p.thetaScale))
        let chain = cumprod(concatenated([posCol, rest], axis: 1), axis: 1) // [t, half]
        var theta: MLXArray
        if p.extFactor != 0 {
            // Same float-op sequence as the C loop: interp·(1-ramp) + extrap·ramp.
            let r = MLXArray(p.ramp).reshaped([1, half])
            theta = (chain * p.freqScale) * (1 - r) + chain * r
        } else {
            theta = chain * p.freqScale
        }
        let c = cos(theta) * p.mscale
        var s = sin(theta) * p.mscale
        if inverse { s = -s }
        let cb = c.reshaped([t, 1, half])
        let sb = s.reshaped([t, 1, half])

        let tail = x[.ellipsis, nNope...].reshaped([t, h, half, 2])
        let a = tail[.ellipsis, 0]
        let b = tail[.ellipsis, 1]
        let a2 = a * cb - b * sb
        let b2 = a * sb + b * cb
        let rotated = concatenated([a2.expandedDimensions(axis: -1), b2.expandedDimensions(axis: -1)], axis: -1)
            .reshaped([t, h, nRot])
        return concatenated([x[.ellipsis, 0..<nNope], rotated], axis: -1)
    }

    // MARK: mHC split / post

    /// HC pre for one sublayer: unweighted RMSNorm over the flat 4·4096 HC
    /// state, F16 control projection to 24, sigmoid/2·sigmoid splits, and the
    /// 20-iteration Sinkhorn combine matrix (ds4.c:11332-11426, metal
    /// dsv4_hc.metal:113-282). `fn` is the seam's F16 [24, 4·embd] control
    /// matrix. Returns pre [T,4], post [T,4] and comb [T, dst, src].
    public static func hcSplit(hc: MLXArray, fn: MLXArray, scale: MLXArray, base: MLXArray,
                        eps: Float, iters: Int) -> (pre: MLXArray, post: MLXArray, comb: MLXArray) {
        let t = hc.dim(0)
        let flat = hc.reshaped([t, -1])
        let nrm = rmsNormNoWeight(flat, eps: eps)
        let mix = matmul(nrm, fn.asType(.float32).transposed()) // [t, 24]
        let sc = scale.asType(.float32)
        let bs = base.asType(.float32)
        let pre = MLXNN.sigmoid(mix[0..., 0..<4] * sc[0] + bs[0..<4]) + eps
        let post = 2 * MLXNN.sigmoid(mix[0..., 4..<8] * sc[1] + bs[4..<8])

        var m = (mix[0..., 8..<24] * sc[2] + bs[8..<24]).reshaped([t, 4, 4]) // [dst][src]
        // First row softmax: exp(v - rowmax) / rowsum (no eps) + eps elementwise.
        m = exp(m - m.max(axis: -1, keepDims: true))
        m = m / m.sum(axis: -1, keepDims: true) + eps
        m = m / (m.sum(axis: -2, keepDims: true) + eps)
        for _ in 1..<iters {
            m = m / (m.sum(axis: -1, keepDims: true) + eps)
            m = m / (m.sum(axis: -2, keepDims: true) + eps)
        }
        // hc_post applies comb[dst + src*4] which addresses M[src][dst].
        return (pre, post, m.transposed(0, 2, 1))
    }

    /// Reduce the four HC streams into the sublayer input (ds4.c:11413).
    public static func hcWeightedSum(pre: MLXArray, hc: MLXArray) -> MLXArray {
        (hc * pre.expandedDimensions(axis: -1)).sum(axis: 1) // [T, 4096]
    }

    /// HC post: out[dst] = block·post[dst] + Σ_src comb[dst][src]·hc[src]
    /// (ds4.c:11512-11532).
    public static func hcPost(blockOut: MLXArray, hc: MLXArray, post: MLXArray, comb: MLXArray) -> MLXArray {
        blockOut.expandedDimensions(axis: 1) * post.expandedDimensions(axis: -1)
            + matmul(comb, hc)
    }

    // MARK: top-k and attention

    /// Descending top-k with ties resolved to the lower index
    /// (ds4.c:12370 topk_desc). scores [T, N] → indices [T, k] Int32.
    public static func topKDescending(_ scores: MLXArray, k: Int) -> MLXArray {
        let n = scores.dim(1)
        let iota = MLXArray((0..<n).map { Int32($0) }).reshaped([1, n])
        var rem = scores
        var idxs: [MLXArray] = []
        for _ in 0..<k {
            let isMax = rem .== rem.max(axis: -1, keepDims: true)
            let idx = which(isMax, iota, MLXArray(Int32(n))).min(axis: -1) // [T]
            idxs.append(idx)
            rem = putAlong(rem, idx.expandedDimensions(axis: 1),
                           values: MLXArray(-Float.infinity), axis: -1)
        }
        return stacked(idxs, axis: 1)
    }

    /// Sink-aware attention of one token's heads over raw + compressed rows
    /// (ds4.c:14307-14367). The sink logit joins the denominator last with
    /// value zero; masked compressed rows score -inf and contribute nothing.
    /// `q` [H, D] F32, `rawK` [R, D] F16, `compK` [C, D] F16 already in visit
    /// order. Returns [H, D] F32.
    public static func attentionRows(q: MLXArray, rawK: MLXArray, compK: MLXArray?,
                              compMask: MLXArray?, sinks: MLXArray, kqScale: Float) -> MLXArray {
        let qh = q.asType(.float16)
        var scoreParts: [MLXArray] = [matmul(qh, rawK.transposed()).asType(.float32) * kqScale]
        var valueParts: [MLXArray] = [rawK]
        if let c = compK {
            var s = matmul(qh, c.transposed()).asType(.float32) * kqScale
            if let mask = compMask {
                s = which(mask.reshaped([1, -1]), s, MLXArray(negInf))
            }
            scoreParts.append(s)
            valueParts.append(c)
        }
        let scores = concatenated(scoreParts, axis: 1) // [H, R+C]
        let values = concatenated(valueParts, axis: 0) // [R+C, D] F16
        let sinksCol = sinks.asType(.float32).reshaped([-1, 1])
        let m = maximum(scores.max(axis: 1, keepDims: true), sinksCol)
        let p = exp(scores - m)
        let denom = p.sum(axis: 1, keepDims: true) + exp(sinksCol - m)
        return matmul(p.asType(.float16), values).asType(.float32) / denom
    }
}
