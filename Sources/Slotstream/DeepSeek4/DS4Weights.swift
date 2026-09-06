// DeepSeek-V4-Flash resident trunk, loaded straight from the GGUF via pread.
//
// Everything except the routed experts lives here: token_embd (F16), the Q8_0
// attention/shared-expert linears and lm head (repacked into MLX's affine
// layout), the F16 compressor/indexer/router tensors, the hyper-connection
// tensors and all norms — 8.80 GB on the reference checkpoint (147.17 GB of
// routed experts stay on SSD; see DS4ExpertStore). Tensor bytes are never
// mmapped: each tensor is pread once into a 16 KiB-aligned buffer that MLX
// then owns (no further copies), matching Checkpoint.swift's F_NOCACHE
// discipline.
//
// Orientation: GGUF dims are ne[] (fastest axis first); the numpy shape is
// the reverse. Every matmul-shaped tensor here is stored [out, in]
// (ne = {in, out}), so a linear in→out lands as w [out, in] and QLinear's
// quantizedMM(transpose: true) applies directly. The two exceptions, loaded
// as-is and documented on their fields, are the *_ape position maps
// (ne = {width, ratio} → [ratio, width]) and the hc fn matrices
// (ne = {4·embd, width} → [width, 4·embd]).
//
// Q8_0 → MLX affine repack (group 32, bits 8): MLX reads the packed bytes
// UNSIGNED (dequant = scale·u8 + bias; see the quantized.h metal kernel and
// vendored mlx's own GGUF loader gguf_quants.cpp, extract_q8_0_data), while
// GGUF Q8_0 stores signed int8. Each payload byte is flipped into unsigned
// (q ^ 0x80 == q + 128 mod 256), four bytes per uint32 in element order, and
// the group bias carries −128·scale, so dequant stays scale·q. The packed
// weight is [out, in/4] uint32 (4 int8 per uint32) — NOT the [out, in/8]
// that applies to bits 4.

import Foundation
import MLX

// MARK: - Pread I/O

/// File-descriptor + pread plumbing shared by the DS4 loaders. Handles run
/// F_NOCACHE / F_RDAHEAD-off like Checkpoint.swift: model bytes are streamed
/// once and must not evict anything the system cached on purpose.
enum DS4IO {
    static func openFD(_ path: String) throws -> Int32 {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw ModelError("cannot open GGUF \(path): \(String(cString: strerror(errno)))")
        }
        _ = fcntl(fd, F_NOCACHE, 1)
        _ = fcntl(fd, F_RDAHEAD, 0)
        return fd
    }

    static func preadFully(
        _ fd: Int32, _ dst: UnsafeMutableRawPointer, fileOffset: UInt64, count: Int
    ) throws {
        guard fileOffset <= UInt64(Int64.max) else {
            throw ModelError("GGUF read offset \(fileOffset) overflows off_t")
        }
        var done = 0
        while done < count {
            let got = Foundation.pread(
                fd, dst + done, count - done, off_t(fileOffset) + off_t(done))
            guard got > 0 else {
                throw ModelError(
                    "pread failed at GGUF offset \(fileOffset + UInt64(done)) "
                        + "(\(done)/\(count) bytes): \(String(cString: strerror(errno)))")
            }
            done += got
        }
    }

    /// pread `count` bytes into a fresh 16 KiB-aligned malloc'd buffer. The
    /// caller hands ownership to MLXArray(rawPointer:), whose finalizer frees.
    static func readAligned(
        _ fd: Int32, fileOffset: UInt64, count: Int, what: String
    ) throws -> UnsafeMutableRawPointer {
        let p = try aligned(count, what: what)
        do {
            try preadFully(fd, p, fileOffset: fileOffset, count: count)
        } catch {
            free(p)
            throw error
        }
        return p
    }

    static func aligned(_ bytes: Int, what: String) throws -> UnsafeMutableRawPointer {
        var p: UnsafeMutableRawPointer? = nil
        let rc = posix_memalign(&p, 16384, max(bytes, 1))
        guard rc == 0, let p else {
            throw ModelError("out of memory staging \(what) (\(bytes) B)")
        }
        return p
    }
}

// MARK: - Q8_0 repack

enum DS4Quant {
    static let blockSize = GGUFQuant.blockSize(.q8_0)!    // 32
    static let bytesPerBlock = GGUFQuant.bytesPerBlock(.q8_0)!  // 34
    static let groupSize = blockSize
    static let q8Bits = 8

    /// Repack one GGUF Q8_0 tensor (logical [out, in], ne = {in, out}, 34 B
    /// per 32 elems) into the affine layout MLX expects: w uint32 [out, in/4],
    /// scales f16 [out, in/32], biases f16 [out, in/32] = −128·scale. See the
    /// file doc comment for why the bytes flip and the bias is not zero.
    static func repackQ8(
        _ raw: UnsafeRawPointer, out: Int, in_: Int, what: String
    ) throws -> (w: UnsafeMutableRawPointer, scales: UnsafeMutableRawPointer, biases: UnsafeMutableRawPointer) {
        guard in_ % groupSize == 0 else {
            throw ModelError("\(what): input dim \(in_) is not a multiple of \(groupSize)")
        }
        let blocks = in_ / groupSize
        let w = try DS4IO.aligned(out * in_, what: "\(what) packed w")
        let scales = try DS4IO.aligned(out * blocks * 2, what: "\(what) scales")
        let biases = try DS4IO.aligned(out * blocks * 2, what: "\(what) biases")
        let src = raw.assumingMemoryBound(to: UInt8.self)
        let wWords = w.assumingMemoryBound(to: UInt32.self)
        let sWords = scales.assumingMemoryBound(to: UInt16.self)
        let bWords = biases.assumingMemoryBound(to: UInt16.self)
        for r in 0 ..< out {
            let row = src + r * blocks * bytesPerBlock
            for b in 0 ..< blocks {
                let blk = row + b * bytesPerBlock
                let scaleBits = UInt16(blk[0]) | UInt16(blk[1]) << 8
                sWords[r * blocks + b] = scaleBits
                // −128·scale with a single f16 rounding, matching mlx's
                // float16 assignment in extract_q8_0_data.
                bWords[r * blocks + b] = (Float16(-128) * Float16(bitPattern: scaleBits)).bitPattern
                // Payload: 32 bytes → 8 uint32 words, little-endian so byte
                // order == element order, each byte flipped q ^ 0x80.
                let dst = r * (in_ / 4) + b * 8
                for j in 0 ..< 8 {
                    let o = blk + 2 + 4 * j
                    var word = UInt32(o[0]) | UInt32(o[1]) << 8
                        | UInt32(o[2]) << 16 | UInt32(o[3]) << 24
                    word ^= 0x8080_8080
                    wWords[dst + j] = word
                }
            }
        }
        return (w, scales, biases)
    }
}

// MARK: - Resident trunk

/// The resident DeepSeek-V4-Flash trunk. See the file doc comment for layout
/// and conversion contracts.
public final class DS4Weights {
    public let config: DS4Config
    /// F16 [vocabSize, embd] — row per token, numpy shape [129280, 4096].
    public let tokenEmbd: MLXArray
    /// lm head, Q8_0 affine packed: [vocabSize, embd] = [129280, 4096].
    public let output: QLinear
    public let outputNorm: MLXArray  // f32 [4096]
    /// Output-head hyper-connection tensors, as stored: base f32 [4],
    /// fn f16 [4, 4·embd] (ne = {16384, 4}), scale f32 [1].
    public let outputHcBase: MLXArray
    public let outputHcFn: MLXArray
    public let outputHcScale: MLXArray

    private let layerList: [DS4LayerWeights]

    public func layer(_ l: Int) -> DS4LayerWeights {
        precondition(
            l >= 0 && l < layerList.count,
            "layer \(l) out of range 0..<\(layerList.count)")
        return layerList[l]
    }

    public init(ggufPath: String, cfg: DS4Config) throws {
        // ~8.8 GB multi-GB trunk: take the same one-model-process guard as
        // ResidentWeights. It is process-wide and idempotent, so a caller
        // that already holds it is unaffected.
        try ModelProcessGuard.acquire()
        let gguf = try GGUFFile(path: ggufPath)
        // Re-runs the geometry checks against this file's tensor directory.
        try cfg.validate(gguf: gguf)
        let fd = try DS4IO.openFD(ggufPath)
        defer { close(fd) }

        config = cfg
        let E = cfg.embeddingLength
        tokenEmbd = try DS4Weights.loadPlain(
            "token_embd.weight", .f16, [E, cfg.vocabSize], gguf, fd)
        output = try DS4Weights.loadQ8(
            "output.weight", in_: E, out: cfg.vocabSize, gguf, fd)
        outputNorm = try DS4Weights.loadPlain("output_norm.weight", .f32, [E], gguf, fd)
        outputHcBase = try DS4Weights.loadPlain("output_hc_base.weight", .f32, [4], gguf, fd)
        outputHcFn = try DS4Weights.loadPlain(
            "output_hc_fn.weight", .f16, [4 * E, 4], gguf, fd)
        outputHcScale = try DS4Weights.loadPlain("output_hc_scale.weight", .f32, [1], gguf, fd)

        var ls: [DS4LayerWeights] = []
        ls.reserveCapacity(cfg.blockCount)
        for l in 0 ..< cfg.blockCount {
            ls.append(try DS4LayerWeights.load(gguf: gguf, fd: fd, cfg: cfg, index: l))
        }
        layerList = ls
    }

    /// Dequantized embedding rows: [T] int32 token ids → [T, embd] f16.
    public func embedding(_ ids: MLXArray) -> MLXArray {
        take(tokenEmbd, ids, axis: 0)
    }

    /// lm head: [T, embd] → [T, vocabSize] (Q8_0 affine quantizedMM).
    public func lmHead(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            x, output.w, scales: output.scales!, biases: output.biases,
            transpose: true, groupSize: output.groupSize, bits: output.bits)
    }

    // MARK: tensor loading helpers (module-internal; also used by selfCheck)

    static func requireTensor(
        _ name: String, _ type: GGUFTensorType, _ ne: [Int], _ gguf: GGUFFile
    ) throws -> GGUFTensorInfo {
        guard let t = gguf.tensor(named: name) else {
            throw ModelError("GGUF is missing tensor \(name) — check --model")
        }
        guard t.type == type else {
            throw ModelError("tensor \(name) is \(t.type), expected \(type) — check --model")
        }
        guard t.dims == ne else {
            throw ModelError(
                "tensor \(name) has ne \(t.dims), expected \(ne) — check --model")
        }
        guard let bytes = t.byteSize, bytes > 0 else {
            throw ModelError("tensor \(name) has no known byte size — check --model")
        }
        return t
    }

    /// Load a plain f16/f32/i32 tensor as-is: pread into an aligned buffer,
    /// hand ownership to MLX. `ne` is file order; the MLX shape is reversed.
    static func loadPlain(
        _ name: String, _ type: GGUFTensorType, _ ne: [Int], _ gguf: GGUFFile, _ fd: Int32
    ) throws -> MLXArray {
        let t = try requireTensor(name, type, ne, gguf)
        let p = try DS4IO.readAligned(
            fd, fileOffset: gguf.dataSectionOffset + t.dataOffset,
            count: t.byteSize!, what: name)
        let dtype: DType
        switch type {
        case .f16: dtype = .float16
        case .f32: dtype = .float32
        case .i32: dtype = .int32
        default: fatalError("loadPlain only takes f16/f32/i32")
        }
        return MLXArray(rawPointer: p, ne.reversed(), dtype: dtype) { free(p) }
    }

    /// Load a Q8_0 [out, in] linear (ne = {in, out}) repacked for MLX.
    static func loadQ8(
        _ name: String, in_: Int, out: Int, _ gguf: GGUFFile, _ fd: Int32
    ) throws -> QLinear {
        let t = try requireTensor(name, .q8_0, [in_, out], gguf)
        let raw = try DS4IO.readAligned(
            fd, fileOffset: gguf.dataSectionOffset + t.dataOffset,
            count: t.byteSize!, what: name)
        let (wp, sp, bp) = try DS4Quant.repackQ8(raw, out: out, in_: in_, what: name)
        free(raw)
        return QLinear(
            w: MLXArray(rawPointer: wp, [out, in_ / 4], dtype: .uint32) { free(wp) },
            scales: MLXArray(rawPointer: sp, [out, in_ / DS4Quant.groupSize], dtype: .float16) {
                free(sp)
            },
            biases: MLXArray(rawPointer: bp, [out, in_ / DS4Quant.groupSize], dtype: .float16) {
                free(bp)
            },
            groupSize: DS4Quant.groupSize, bits: DS4Quant.q8Bits)
    }
}

// MARK: - Per-layer weights

/// One decoder block's resident weights. All matmul weights are [out, in]
/// (ne = {in, out}); the exceptions are documented on their fields. Tensors
/// that exist only on some layers are optionals, required exactly when the
/// layer's compress ratio / index demands them (missing required tensors and
/// unexpected shapes throw at load).
public final class DS4LayerWeights {
    public let index: Int
    /// attention/kv compress ratio for this block (0 = none), from
    /// deepseek4.attention.compress_ratios.
    public let compressRatio: Int

    // Norms and sinks, F32 as stored.
    public let attnNorm: MLXArray       // [4096]
    public let ffnNorm: MLXArray        // [4096]
    public let attnQANorm: MLXArray     // [1024] (q_a lora norm)
    public let attnKVANorm: MLXArray    // [512]
    public let attnSinks: MLXArray      // [64], one sink logit per head

    // Attention projections, Q8_0 affine packed (group 32, bits 8), [out, in].
    public let attnQA: QLinear        // 4096 → 1024 (q lora down-proj)
    public let attnQB: QLinear        // 1024 → 32768 (64 heads × 512)
    public let attnKV: QLinear        // 4096 → 512 (compressed kv latent)
    public let attnOutputA: QLinear   // 4096 → 8192
    public let attnOutputB: QLinear   // 8192 → 4096

    // Shared expert, Q8_0, [out, in].
    public let ffnGateShexp: QLinear  // 4096 → 2048
    public let ffnUpShexp: QLinear    // 4096 → 2048
    public let ffnDownShexp: QLinear  // 2048 → 4096

    // MoE router, F16 [256, 4096] standard [out, in]; resident on every layer.
    public let ffnGateInp: MLXArray
    /// Token-id → expert-id table, I32 [129280, 6]; hash layers 0–2 only.
    public let ffnGateTid2Eid: MLXArray?
    /// Per-expert routing bias, F32 [256]; layers 3+ only.
    public let expProbsBias: MLXArray?

    // Attention kv compressor, layers with compressRatio > 0. F16 in the
    // standard [out, in] orientation: kv/gate map 4096 → `compressorOutWidth`
    // (1024 at ratio 4, 512 at ratio 128 in the reference file).
    public let attnCompressorKV: MLXArray?
    public let attnCompressorGate: MLXArray?
    /// Position map, NOT a matmul weight: stored ne = {width, ratio} →
    /// MLX [ratio, width]. axis0 indexes the compressed position (the ratio),
    /// axis1 the compressed slot (the width).
    public let attnCompressorApe: MLXArray?
    public let attnCompressorNorm: MLXArray?  // f32 [512]
    public let compressorOutWidth: Int?

    // Sparse-attention indexer, compressRatio == 4 layers only (even layers
    // 2–42). F16 standard [out, in]: compressor kv/gate 4096 → 256, q_b
    // 1024 → 8192 (64 indexer heads × 128), proj 4096 → 64 (one score per
    // head). indexer.k_norm.{weight,bias} and indexer.ape are absent from the
    // reference checkpoint; the optionals stay so a future export can add
    // them without a format break.
    public let indexerCompressorKV: MLXArray?
    public let indexerCompressorGate: MLXArray?
    /// ne = {256, 4} → MLX [4, 256]; position map like the attn one.
    public let indexerCompressorApe: MLXArray?
    public let indexerCompressorNorm: MLXArray?  // f32 [128]
    public let indexerAttnQB: MLXArray?          // [8192, 1024]
    public let indexerProj: MLXArray?            // [64, 4096]
    public let indexerKNorm: MLXArray?
    public let indexerKNormBias: MLXArray?
    public let indexerApe: MLXArray?

    // Hyper-connection tensors, every layer. fn is stored ne = {4·embd,
    // width} → MLX [width, 4·embd]: axis0 is the width dimension of the
    // reference file (24 for attn/ffn, 4 for the output head), axis1 the
    // concatenated hidden state (4 streams × 4096). base/scale are F32.
    public let hcAttnFn: MLXArray     // f16 [24, 16384]
    public let hcAttnBase: MLXArray   // f32 [24]
    public let hcAttnScale: MLXArray  // f32 [3]
    public let hcFfnFn: MLXArray      // f16 [24, 16384]
    public let hcFfnBase: MLXArray    // f32 [24]
    public let hcFfnScale: MLXArray   // f32 [3]

    static func load(
        gguf: GGUFFile, fd: Int32, cfg: DS4Config, index: Int
    ) throws -> DS4LayerWeights {
        let blk = "blk.\(index)."
        let E = cfg.embeddingLength
        let ratio = cfg.compressRatio(at: index) ?? 0

        // Plain F32/F16/I32 helpers bound to this call.
        func plain(_ name: String, _ type: GGUFTensorType, _ ne: [Int]) throws -> MLXArray {
            try DS4Weights.loadPlain(blk + name, type, ne, gguf, fd)
        }
        func q8(_ name: String, in_: Int, out: Int) throws -> QLinear {
            try DS4Weights.loadQ8(blk + name, in_: in_, out: out, gguf, fd)
        }
        func optionalPlain(_ name: String, _ type: GGUFTensorType, _ ne: [Int]) throws -> MLXArray? {
            guard gguf.tensor(named: blk + name) != nil else { return nil }
            return try DS4Weights.loadPlain(blk + name, type, ne, gguf, fd)
        }

        // Q8 attention + shared expert, validated against the Flash geometry.
        let attnQA = try q8("attn_q_a.weight", in_: E, out: cfg.qLoraRank)
        let attnQB = try q8(
            "attn_q_b.weight", in_: cfg.qLoraRank, out: cfg.headCount * cfg.keyLength)
        let attnKV = try q8("attn_kv.weight", in_: E, out: cfg.keyLength)
        // 8192 = 2 × hidden in the reference file (64 heads × 128 value
        // width); not derivable from metadata, so it is pinned literally.
        let attnOutputA = try q8("attn_output_a.weight", in_: E, out: 2 * E)
        let attnOutputB = try q8("attn_output_b.weight", in_: 2 * E, out: E)
        let ffn = cfg.expertFeedForwardLength
        let ffnGateShexp = try q8("ffn_gate_shexp.weight", in_: E, out: ffn)
        let ffnUpShexp = try q8("ffn_up_shexp.weight", in_: E, out: ffn)
        let ffnDownShexp = try q8("ffn_down_shexp.weight", in_: ffn, out: E)

        // Router + routing tables.
        let ffnGateInp = try plain("ffn_gate_inp.weight", .f16, [E, cfg.expertCount])
        let hash = cfg.hashLayerCount
        let tid2eid: MLXArray? =
            index < hash ? try plain("ffn_gate_tid2eid.weight", .i32, [cfg.expertUsedCount, cfg.vocabSize]) : nil
        let expProbs: MLXArray? = index >= hash ? try plain("exp_probs_b.bias", .f32, [cfg.expertCount]) : nil

        // Attention kv compressor (ratio > 0 layers: 2–42).
        var cKV: MLXArray? = nil, cGate: MLXArray? = nil, cApe: MLXArray? = nil
        var cNorm: MLXArray? = nil, cWidth: Int? = nil
        if ratio > 0 {
            // The compressor width is not in the metadata (1024 at ratio 4,
            // 512 at ratio 128 in the reference file); take it from the file
            // and cross-check gate/ape against it.
            guard let kvInfo = gguf.tensor(named: blk + "attn_compressor_kv.weight"),
                kvInfo.type == .f16, kvInfo.dims.count == 2, kvInfo.dims[0] == E,
                kvInfo.dims[1] > 0
            else {
                throw ModelError(
                    "layer \(index) needs a 2-D F16 \(blk)attn_compressor_kv.weight "
                        + "with ne [\(E), width] — check --model")
            }
            let width = kvInfo.dims[1]
            cWidth = width
            cKV = try plain("attn_compressor_kv.weight", .f16, [E, width])
            cGate = try plain("attn_compressor_gate.weight", .f16, [E, width])
            cApe = try plain("attn_compressor_ape.weight", .f16, [width, ratio])
            cNorm = try plain("attn_compressor_norm.weight", .f32, [cfg.keyLength])
        }

        // Sparse-attention indexer (ratio == 4 layers only).
        var iKV: MLXArray? = nil, iGate: MLXArray? = nil, iApe: MLXArray? = nil
        var iNorm: MLXArray? = nil, iQB: MLXArray? = nil, iProj: MLXArray? = nil
        var iKNorm: MLXArray? = nil, iKNormBias: MLXArray? = nil, iTopApe: MLXArray? = nil
        if ratio == 4 {
            // 256 compressor width: fixed by the reference export, not in the
            // metadata (indexer heads 64 × 128 live in q_b instead).
            let width = 256
            iKV = try plain("indexer_compressor_kv.weight", .f16, [E, width])
            iGate = try plain("indexer_compressor_gate.weight", .f16, [E, width])
            iApe = try plain("indexer_compressor_ape.weight", .f16, [width, ratio])
            iNorm = try plain("indexer_compressor_norm.weight", .f32, [128])
            iQB = try plain(
                "indexer.attn_q_b.weight", .f16,
                [cfg.qLoraRank, cfg.indexerHeadCount * cfg.indexerKeyLength])
            iProj = try plain("indexer.proj.weight", .f16, [E, cfg.indexerHeadCount])
            iKNorm = try optionalPlain("indexer.k_norm.weight", .f32, [128])
            iKNormBias = try optionalPlain("indexer.k_norm.bias", .f32, [128])
            // indexer.ape (the indexer attention's own position map) is a
            // DIFFERENT tensor from indexer_compressor_ape; it is absent from
            // the reference checkpoint. Assigning it over iApe turned every
            // required compressor ape nil and killed the first real forward
            // pass at layer 2.
            iTopApe = try optionalPlain("indexer.ape.weight", .f16, [128, ratio])
        }

        // Hyper connections (every layer).
        let hcAttnFn = try plain("hc_attn_fn.weight", .f16, [4 * E, 24])
        let hcAttnBase = try plain("hc_attn_base.weight", .f32, [24])
        let hcAttnScale = try plain("hc_attn_scale.weight", .f32, [3])
        let hcFfnFn = try plain("hc_ffn_fn.weight", .f16, [4 * E, 24])
        let hcFfnBase = try plain("hc_ffn_base.weight", .f32, [24])
        let hcFfnScale = try plain("hc_ffn_scale.weight", .f32, [3])

        return DS4LayerWeights(
            index: index, compressRatio: ratio,
            attnNorm: try plain("attn_norm.weight", .f32, [E]),
            ffnNorm: try plain("ffn_norm.weight", .f32, [E]),
            attnQANorm: try plain("attn_q_a_norm.weight", .f32, [cfg.qLoraRank]),
            attnKVANorm: try plain("attn_kv_a_norm.weight", .f32, [cfg.keyLength]),
            attnSinks: try plain("attn_sinks.weight", .f32, [cfg.headCount]),
            attnQA: attnQA, attnQB: attnQB, attnKV: attnKV,
            attnOutputA: attnOutputA, attnOutputB: attnOutputB,
            ffnGateShexp: ffnGateShexp, ffnUpShexp: ffnUpShexp, ffnDownShexp: ffnDownShexp,
            ffnGateInp: ffnGateInp, ffnGateTid2Eid: tid2eid, expProbsBias: expProbs,
            attnCompressorKV: cKV, attnCompressorGate: cGate, attnCompressorApe: cApe,
            attnCompressorNorm: cNorm, compressorOutWidth: cWidth,
            indexerCompressorKV: iKV, indexerCompressorGate: iGate,
            indexerCompressorApe: iApe, indexerCompressorNorm: iNorm,
            indexerAttnQB: iQB, indexerProj: iProj,
            indexerKNorm: iKNorm, indexerKNormBias: iKNormBias, indexerApe: iTopApe,
            hcAttnFn: hcAttnFn, hcAttnBase: hcAttnBase, hcAttnScale: hcAttnScale,
            hcFfnFn: hcFfnFn, hcFfnBase: hcFfnBase, hcFfnScale: hcFfnScale)
    }

    init(
        index: Int, compressRatio: Int,
        attnNorm: MLXArray, ffnNorm: MLXArray, attnQANorm: MLXArray, attnKVANorm: MLXArray,
        attnSinks: MLXArray,
        attnQA: QLinear, attnQB: QLinear, attnKV: QLinear, attnOutputA: QLinear,
        attnOutputB: QLinear,
        ffnGateShexp: QLinear, ffnUpShexp: QLinear, ffnDownShexp: QLinear,
        ffnGateInp: MLXArray, ffnGateTid2Eid: MLXArray?, expProbsBias: MLXArray?,
        attnCompressorKV: MLXArray?, attnCompressorGate: MLXArray?, attnCompressorApe: MLXArray?,
        attnCompressorNorm: MLXArray?, compressorOutWidth: Int?,
        indexerCompressorKV: MLXArray?, indexerCompressorGate: MLXArray?,
        indexerCompressorApe: MLXArray?, indexerCompressorNorm: MLXArray?,
        indexerAttnQB: MLXArray?, indexerProj: MLXArray?,
        indexerKNorm: MLXArray?, indexerKNormBias: MLXArray?, indexerApe: MLXArray?,
        hcAttnFn: MLXArray, hcAttnBase: MLXArray, hcAttnScale: MLXArray,
        hcFfnFn: MLXArray, hcFfnBase: MLXArray, hcFfnScale: MLXArray
    ) {
        self.index = index
        self.compressRatio = compressRatio
        self.attnNorm = attnNorm
        self.ffnNorm = ffnNorm
        self.attnQANorm = attnQANorm
        self.attnKVANorm = attnKVANorm
        self.attnSinks = attnSinks
        self.attnQA = attnQA
        self.attnQB = attnQB
        self.attnKV = attnKV
        self.attnOutputA = attnOutputA
        self.attnOutputB = attnOutputB
        self.ffnGateShexp = ffnGateShexp
        self.ffnUpShexp = ffnUpShexp
        self.ffnDownShexp = ffnDownShexp
        self.ffnGateInp = ffnGateInp
        self.ffnGateTid2Eid = ffnGateTid2Eid
        self.expProbsBias = expProbsBias
        self.attnCompressorKV = attnCompressorKV
        self.attnCompressorGate = attnCompressorGate
        self.attnCompressorApe = attnCompressorApe
        self.attnCompressorNorm = attnCompressorNorm
        self.compressorOutWidth = compressorOutWidth
        self.indexerCompressorKV = indexerCompressorKV
        self.indexerCompressorGate = indexerCompressorGate
        self.indexerCompressorApe = indexerCompressorApe
        self.indexerCompressorNorm = indexerCompressorNorm
        self.indexerAttnQB = indexerAttnQB
        self.indexerProj = indexerProj
        self.indexerKNorm = indexerKNorm
        self.indexerKNormBias = indexerKNormBias
        self.indexerApe = indexerApe
        self.hcAttnFn = hcAttnFn
        self.hcAttnBase = hcAttnBase
        self.hcAttnScale = hcAttnScale
        self.hcFfnFn = hcFfnFn
        self.hcFfnBase = hcFfnBase
        self.hcFfnScale = hcFfnScale
    }
}

// MARK: - Self check

extension DS4Weights {
    /// Verify the DS4 load path against real checkpoint bytes WITHOUT loading
    /// the trunk — reads a few tens of MB of tensor data and some transient
    /// dequant buffers, so it is safe to run next to a model process. Throws
    /// a `ModelError` naming the check on the first mismatch.
    ///
    /// [1] Q8_0, blk.0.attn_q_a: repack + MLX dequantize vs the raw
    ///     int8·scale values parsed independently from the GGUF bytes, then
    ///     quantizedMM vs a manual f64 matmul (proves the [out, in]
    ///     orientation end to end).
    /// [2] F16, blk.2.attn_compressor_kv: byte-exact against the file, and a
    ///     matmul vs a manual f64 reference.
    /// [3] MXFP4, expert 0 of blk.5 via DS4ExpertStore.readBatch: the six
    ///     pieces dequantized with MLX vs a manual E8M0×E2M1 dequant of the
    ///     same rows pread straight from the file (both paths read the same
    ///     bytes, so any orientation or nibble-order bug shows up here), plus
    ///     one gatherQuantizedMM over the 3-D pieces — the seam SlotPool will
    ///     call.
    public static func selfCheck(ggufPath: String) throws -> String {
        let gguf = try GGUFFile(path: ggufPath)
        let cfg = try DS4Config(gguf: gguf)
        let fd = try DS4IO.openFD(ggufPath)
        defer { close(fd) }

        var report: [String] = []
        report.append("DS4 selfCheck: \((ggufPath as NSString).lastPathComponent)")
        var trunk = 0
        var experts = 0
        for t in gguf.tensors {
            let b = t.byteSize ?? 0
            if t.name.contains("_exps") { experts += b } else { trunk += b }
        }
        report.append(String(
            format: "dir totals: trunk %.2f GB, routed experts %.2f GB",
            Double(trunk) / 1e9, Double(experts) / 1e9))

        // 1. Q8_0 affine repack — blk.0.attn_q_a [1024, 4096].
        do {
            let info = try requireTensor("blk.0.attn_q_a.weight", .q8_0, [4096, 1024], gguf)
            let raw = try DS4IO.readAligned(
                fd, fileOffset: gguf.dataSectionOffset + info.dataOffset,
                count: info.byteSize!, what: "selfcheck attn_q_a")
            defer { free(raw) }
            let (wp, sp, bp) = try DS4Quant.repackQ8(raw, out: 1024, in_: 4096, what: "attn_q_a")
            let w = MLXArray(rawPointer: wp, [1024, 4096 / 4], dtype: .uint32) { free(wp) }
            let s = MLXArray(rawPointer: sp, [1024, 4096 / 32], dtype: .float16) { free(sp) }
            let b = MLXArray(rawPointer: bp, [1024, 4096 / 32], dtype: .float16) { free(bp) }

            let deq = dequantized(
                w[0 ..< 8], scales: s[0 ..< 8], biases: b[0 ..< 8],
                groupSize: 32, bits: 8, dtype: .float32
            ).asArray(Float32.self)
            // The affine dequantize kernel evaluates at f16 (the scales'
            // dtype) even when f32 output is requested — verified with a
            // synthetic tensor: scale·u8 rounds to f16 before the bias add.
            // So the exact f64 reference carries a per-element rounding
            // noise of up to ~2^-11 × 255·scale; bound each element by
            // 0.25·scale (structural bugs — wrong byte order, missing XOR,
            // transposed rows — differ by O(|q|)·scale, 1000x beyond this).
            let parsed = manualQ8Rows(raw, rows: 0 ..< 8, in_: 4096)
            let manual = parsed.values.flatMap { $0 }
            var tol = [Double](repeating: 0, count: manual.count)
            for r in 0 ..< 8 {
                for b in 0 ..< parsed.scales[r].count {
                    let t = 0.25 * parsed.scales[r][b] + 1e-7
                    for j in 0 ..< 32 { tol[r * 4096 + b * 32 + j] = t }
                }
            }
            let d1 = try compareElementwise(
                deq.map(Double.init), manual, tol: tol, what: "attn_q_a dequant")

            let x = checkX(4, 4096)
            let y = quantizedMM(
                x, w, scales: s, biases: b, transpose: true, groupSize: 32, bits: 8
            ).asArray(Float32.self)
            let xd = manualX(4, 4096)
            let wFull = allQ8Values(raw, rows: 0 ..< 1024, in_: 4096)
            var y2 = [Double](repeating: 0, count: 4 * 1024)
            for i in 0 ..< 4 {
                for o in 0 ..< 1024 {
                    var acc = 0.0
                    let row = wFull[o]
                    for k in 0 ..< 4096 { acc += xd[i * 4096 + k] * row[k] }
                    y2[i * 1024 + o] = acc
                }
            }
            let d2 = try compare(
                y.map(Double.init), y2, what: "attn_q_a matmul", relTol: 5e-3)
            report.append(String(
                format: "[q8]   blk.0.attn_q_a: dequant max|Δ| %.3g, matmul rel max|Δ| %.3g", d1, d2))
        }

        // 2. F16 compressor — blk.2.attn_compressor_kv [1024, 4096].
        do {
            let info = try requireTensor(
                "blk.2.attn_compressor_kv.weight", .f16, [4096, 1024], gguf)
            let raw = try DS4IO.readAligned(
                fd, fileOffset: gguf.dataSectionOffset + info.dataOffset,
                count: info.byteSize!, what: "selfcheck compressor_kv")
            defer { free(raw) }
            let arr = MLXArray(
                rawPointer: raw, [1024, 4096], dtype: .float16) {}
            // f16 → Double is exact, so value equality == byte equality.
            let rawHalfs = raw.assumingMemoryBound(to: UInt16.self)
            let mlxHalfs = arr.asArray(Float16.self)
            var diffs = 0
            for i in 0 ..< mlxHalfs.count
            where Double(mlxHalfs[i]) != Double(Float16(bitPattern: rawHalfs[i])) {
                diffs += 1
            }
            guard diffs == 0 else {
                throw ModelError(
                    "selfCheck compressor_kv: \(diffs)/\(mlxHalfs.count) f16 values differ from file")
            }
            let x = checkX(4, 4096)
            let y = matmul(x, arr[0 ..< 8].transposed()).asArray(Float32.self)
            let xd = manualX(4, 4096)
            var y2 = [Double](repeating: 0, count: 4 * 8)
            for i in 0 ..< 4 {
                for o in 0 ..< 8 {
                    var acc = 0.0
                    for k in 0 ..< 4096 {
                        acc += xd[i * 4096 + k] * Double(Float16(bitPattern: rawHalfs[o * 4096 + k]))
                    }
                    y2[i * 8 + o] = acc
                }
            }
            let d = try compare(
                y.map(Double.init), y2, what: "compressor_kv matmul", relTol: 1e-3)
            report.append(String(
                format: "[f16]  blk.2.attn_compressor_kv: bytes exact, matmul rel max|Δ| %.3g", d))
        }

        // 3. MXFP4 experts — expert 0 of blk.5, through DS4ExpertStore.
        do {
            let store = try DS4ExpertStore(ggufPath: ggufPath, cfg: cfg)
            let ff = cfg.expertFeedForwardLength
            let E = cfg.embeddingLength
            let nExp = cfg.expertCount
            let expect: [(rows: Int, cols: Int)] = [
                (ff, E / 8), (ff, E / 32), (ff, E / 8), (ff, E / 32), (E, ff / 8), (E, ff / 32),
            ]
            let shapesOK = store.poolShapes.count == expect.count
                && zip(store.poolShapes, expect).allSatisfy { $0 == $1 }
            guard shapesOK else {
                throw ModelError("selfCheck poolShapes \(store.poolShapes), expected \(expect)")
            }
            let expectRecord = 3 * ff * (E / GGUFQuant.blockSize(.mxfp4)!) * GGUFQuant.bytesPerBlock(.mxfp4)!
            guard store.recordBytes == expectRecord else {
                throw ModelError(
                    "selfCheck recordBytes \(store.recordBytes), expected \(expectRecord)")
            }

            let pieces = try store.readBatch([ExpertKey(5, 0)])
            guard pieces.count == 6 else {
                throw ModelError("selfCheck readBatch returned \(pieces.count) pieces, expected 6")
            }
            let specs: [(name: String, rows: Int, k: Int)] = [
                ("blk.5.ffn_gate_exps.weight", ff, E),
                ("blk.5.ffn_up_exps.weight", ff, E),
                ("blk.5.ffn_down_exps.weight", E, ff),
            ]
            var worst = 0.0
            var gateManualFlat: [Double] = []
            for (p, spec) in specs.enumerated() {
                let info = try requireTensor(spec.name, .mxfp4, [spec.k, spec.rows, nExp], gguf)
                let expertBytes = info.byteSize! / nExp
                let raw = try DS4IO.readAligned(
                    fd, fileOffset: gguf.dataSectionOffset + info.dataOffset,
                    count: expertBytes, what: "selfcheck \(spec.name)")
                defer { free(raw) }

                let wArr = pieces[2 * p][0]  // [rows, k/8] uint32
                let sArr = pieces[2 * p + 1][0]  // [rows, k/32] uint8
                guard wArr.shape == [spec.rows, spec.k / 8], sArr.shape == [spec.rows, spec.k / 32] else {
                    throw ModelError(
                        "selfCheck \(spec.name): pieces \(wArr.shape)/\(sArr.shape), "
                            + "expected [\(spec.rows), \(spec.k / 8)]/[\(spec.rows), \(spec.k / 32)]")
                }
                let deq = dequantized(
                    wArr, scales: sArr, biases: nil,
                    groupSize: 32, bits: 4, mode: .mxfp4, dtype: .float32
                )
                let deqRows = deq[0 ..< 8].asArray(Float32.self).map(Double.init)
                let manual = manualMXFP4Rows(
                    raw, rows: 0 ..< 8, k: spec.k,
                    rowBytes: (spec.k / GGUFQuant.blockSize(.mxfp4)!) * GGUFQuant.bytesPerBlock(.mxfp4)!
                ).flatMap { $0 }
                // E8M0 scales are exact powers of two and E2M1 values have
                // 1-2 mantissa bits, so the kernel's product is exact at any
                // working precision — the comparison must be exact.
                let tol = manual.map { 1e-9 * max(1, abs($0)) }
                worst = try max(
                    worst,
                    compareElementwise(deqRows, manual, tol: tol, what: "\(spec.name) dequant"))
                if p == 0 {
                    gateManualFlat = manualMXFP4Rows(
                        raw, rows: 0 ..< ff, k: spec.k,
                        rowBytes: (spec.k / GGUFQuant.blockSize(.mxfp4)!) * GGUFQuant.bytesPerBlock(.mxfp4)!
                    ).flatMap { $0 }
                }
            }

            // Pool seam: grouped mxfp4 gather over the 3-D pieces.
            let x3 = checkX(8, E).expandedDimensions(axes: [1])  // [8, 1, E]
            let idx = MLXArray([Int32](repeating: 0, count: 8))
            let out = gatherQuantizedMM(
                x3, pieces[0], scales: pieces[1], biases: nil,
                rhsIndices: idx, transpose: true, groupSize: 32, bits: 4, mode: .mxfp4
            ).squeezed(axis: 1).asArray(Float32.self)  // [8, ff]
            let xd = manualX(8, E)
            var out2 = [Double](repeating: 0, count: 8 * ff)
            for i in 0 ..< 8 {
                for n in 0 ..< ff {
                    var acc = 0.0
                    let row = Array(gateManualFlat[n * E ..< (n + 1) * E])
                    for k in 0 ..< E { acc += xd[i * E + k] * row[k] }
                    out2[i * ff + n] = acc
                }
            }
            let dg = try compare(
                out.map(Double.init), out2, what: "gate gatherQuantizedMM", relTol: 1e-3)
            report.append(String(
                format:
                    "[mx4]  blk.5 expert 0: gate/up/down dequant max|Δ| %.3g, gatherQuantizedMM rel max|Δ| %.3g",
                worst, dg))
            report.append(String(
                format: "[mx4]  recordBytes %d, poolShapes %@",
                store.recordBytes, String(describing: store.poolShapes)))
        }

        report.append("PASS")
        return report.joined(separator: "\n")
    }

    // MARK: self-check references (manual, independent of the load path)

    /// Dequant rows of a raw GGUF Q8_0 tensor: scale·int8, f64. Also returns
    /// the per-block f16 scales so callers can bound kernel rounding noise.
    private static func manualQ8Rows(
        _ raw: UnsafeRawPointer, rows: Range<Int>, in_: Int
    ) -> (values: [[Double]], scales: [[Double]]) {
        let blocks = in_ / DS4Quant.blockSize
        let bpb = DS4Quant.bytesPerBlock
        var vals: [[Double]] = []
        var scales: [[Double]] = []
        vals.reserveCapacity(rows.count)
        scales.reserveCapacity(rows.count)
        for r in rows {
            let row = (raw + r * blocks * bpb).assumingMemoryBound(to: UInt8.self)
            var v = [Double](repeating: 0, count: in_)
            var s = [Double](repeating: 0, count: blocks)
            for b in 0 ..< blocks {
                let blk = row + b * bpb
                let bits = UInt16(blk[0]) | UInt16(blk[1]) << 8
                let scale = Double(Float16(bitPattern: bits))
                s[b] = scale
                for j in 0 ..< DS4Quant.blockSize {
                    v[b * DS4Quant.blockSize + j] =
                        scale * Double(Int8(bitPattern: blk[2 + j]))
                }
            }
            vals.append(v)
            scales.append(s)
        }
        return (vals, scales)
    }

    /// Exact-f64 dequant of whole rows (for the manual matmul reference).
    private static func allQ8Values(
        _ raw: UnsafeRawPointer, rows: Range<Int>, in_: Int
    ) -> [[Double]] {
        manualQ8Rows(raw, rows: rows, in_: in_).values
    }

    /// Dequant rows of a raw MXFP4 expert span: 2^(e8m0−127)·E2M1, f64.
    /// Nibbles 8–15 are the negative halves of the E2M1 table.
    private static func manualMXFP4Rows(
        _ raw: UnsafeRawPointer, rows: Range<Int>, k: Int, rowBytes: Int
    ) -> [[Double]] {
        let blocks = k / GGUFQuant.blockSize(.mxfp4)!
        let bpb = GGUFQuant.bytesPerBlock(.mxfp4)!
        let lut: [Double] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
        var out: [[Double]] = []
        out.reserveCapacity(rows.count)
        for r in rows {
            let row = (raw + r * rowBytes).assumingMemoryBound(to: UInt8.self)
            var vals = [Double]()
            vals.reserveCapacity(k)
            for b in 0 ..< blocks {
                let blk = row + b * bpb
                let scale = pow(2, Double(blk[0]) - 127)
                for j in 0 ..< 16 {
                    let n = blk[1 + j]
                    let lo = n & 0xF, hi = n >> 4
                    vals.append((lo & 0x8 == 0 ? 1 : -1) * scale * lut[Int(lo & 0x7)])
                    vals.append((hi & 0x8 == 0 ? 1 : -1) * scale * lut[Int(hi & 0x7)])
                }
            }
            out.append(vals)
        }
        return out
    }

    /// Deterministic check input as f16 (MLX side) and exact f16→f64 (manual).
    private static func checkValue(_ i: Int, _ k: Int) -> Float16 {
        Float16(cos(Double((i * 3 + k * 11) % 41)) * 0.7)
    }

    private static func checkX(_ rows: Int, _ cols: Int) -> MLXArray {
        var v = [Float16]()
        v.reserveCapacity(rows * cols)
        for i in 0 ..< rows {
            for k in 0 ..< cols { v.append(checkValue(i, k)) }
        }
        return MLXArray(v, [rows, cols])
    }

    private static func manualX(_ rows: Int, _ cols: Int) -> [Double] {
        var v = [Double]()
        v.reserveCapacity(rows * cols)
        for i in 0 ..< rows {
            for k in 0 ..< cols { v.append(Double(checkValue(i, k))) }
        }
        return v
    }

    /// Max |Δ| between two flat arrays, throwing if the relative error (vs
    /// the reference max magnitude) exceeds `relTol`.
    private static func compare(
        _ a: [Double], _ b: [Double], what: String, relTol: Double
    ) throws -> Double {
        precondition(a.count == b.count, "\(what): \(a.count) vs \(b.count) elements")
        var maxDiff = 0.0
        var maxRef = 0.0
        for i in 0 ..< a.count {
            maxDiff = max(maxDiff, abs(a[i] - b[i]))
            maxRef = max(maxRef, abs(b[i]))
        }
        let rel = maxRef > 0 ? maxDiff / maxRef : maxDiff
        guard rel <= relTol else {
            throw ModelError(
                String(
                    format: "selfCheck %@: max |Δ| %.6g (rel %.3g) exceeds tolerance %.3g",
                    what, maxDiff, rel, relTol))
        }
        return maxDiff
    }

    /// Elementwise variant for dequant comparisons: every element carries its
    /// own absolute tolerance, so f16 kernel rounding can be accepted while
    /// any structural (layout/orientation/scale) mismatch still fails hard.
    private static func compareElementwise(
        _ a: [Double], _ b: [Double], tol: [Double], what: String
    ) throws -> Double {
        precondition(a.count == b.count, "\(what): \(a.count) vs \(b.count) elements")
        precondition(a.count == tol.count, "\(what): \(a.count) vs \(tol.count) tolerances")
        var maxDiff = 0.0
        var worstRatio = 0.0
        var worst = 0
        for i in 0 ..< a.count {
            let d = abs(a[i] - b[i])
            if d > maxDiff { maxDiff = d }
            let ratio = d / tol[i]
            if ratio > worstRatio { worstRatio = ratio; worst = i }
        }
        guard worstRatio <= 1 else {
            throw ModelError(
                String(
                    format:
                        "selfCheck %@: element %d off by %.6g (ref %.6g, tol %.3g) — repack is wrong",
                    what, worst, a[worst] - b[worst], b[worst], tol[worst]))
        }
        return maxDiff
    }
}
