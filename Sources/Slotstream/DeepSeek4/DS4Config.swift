// DeepSeek-V4-Flash GGUF config: parses the `deepseek4.*` metadata keys and
// validates the exact Flash geometry before anything downstream can trust it.
// Validation mirrors Checkpoint.swift: reject anything that is not this one
// architecture with a clear `--model`-style message instead of crashing later.

import Foundation

public struct DS4Config: Sendable {
    public let blockCount: Int
    public let contextLength: Int
    public let embeddingLength: Int
    public let vocabSize: Int
    public let headCount: Int
    public let headCountKV: Int
    public let keyLength: Int
    public let valueLength: Int
    public let qLoraRank: Int
    public let outputLoraRank: Int
    public let outputGroupCount: Int
    public let slidingWindow: Int
    public let rmsNormEpsilon: Float
    /// Per-layer attention compress ratios, exactly as stored in the file.
    /// The reference checkpoint has 46 entries for 43 blocks (the tail looks
    /// like the 3 hash layers); index only through `compressRatio(at:)`.
    public let compressRatios: [Int]
    public let indexerHeadCount: Int
    public let indexerKeyLength: Int
    public let indexerTopK: Int
    public let ropeDimensionCount: Int
    public let ropeFreqBase: Float
    public let ropeScalingType: String
    public let ropeScalingFactor: Float
    public let ropeOriginalContextLength: Int
    public let yarnBetaFast: Float
    public let yarnBetaSlow: Float
    public let compressRopeFreqBase: Float
    public let expertCount: Int
    public let expertUsedCount: Int
    public let expertFeedForwardLength: Int
    public let expertSharedCount: Int
    public let expertWeightsScale: Float
    public let expertWeightsNorm: Bool
    public let expertGatingFunc: Int
    public let hashLayerCount: Int
    public let nextNPredictLayers: Int
    public let hyperConnectionCount: Int
    public let sinkhornIterations: Int
    public let hyperConnectionEpsilon: Float
    /// Per-block swiglu clamp exponent (all 10.0 in the reference file).
    public let swigluClampExp: [Float]
    public let checkpointVariant: String
    public let visionSidecarRequired: Bool

    /// The `deepseek4.` metadata prefix; keys are read as prefix + suffix.
    public static let prefix = "deepseek4"
    /// The only checkpoint variant this build supports.
    public static let supportedVariant = "vision-exp"

    public init(gguf: GGUFFile) throws {
        let arch = gguf.kv("general.architecture").flatMap(\.stringValue)
        guard arch == Self.prefix else {
            throw ModelError(
                "GGUF architecture is \(arch ?? "missing"), not \(Self.prefix) — check --model")
        }
        func key(_ suffix: String) -> String { "\(Self.prefix).\(suffix)" }
        func int(_ suffix: String) throws -> Int {
            guard let v = gguf.kv(key(suffix))?.intValue else {
                throw ModelError("GGUF metadata is missing \(key(suffix)) — check --model")
            }
            return v
        }
        func float(_ suffix: String) throws -> Float {
            guard let v = gguf.kv(key(suffix))?.doubleValue else {
                throw ModelError("GGUF metadata is missing \(key(suffix)) — check --model")
            }
            return Float(v)
        }
        func str(_ suffix: String) throws -> String {
            guard let v = gguf.kv(key(suffix))?.stringValue else {
                throw ModelError("GGUF metadata is missing \(key(suffix)) — check --model")
            }
            return v
        }
        func bool(_ suffix: String) throws -> Bool {
            guard let v = gguf.kv(key(suffix))?.boolValue else {
                throw ModelError("GGUF metadata is missing \(key(suffix)) — check --model")
            }
            return v
        }
        func intArray(_ suffix: String) throws -> [Int] {
            guard let arr = gguf.kv(key(suffix))?.arrayValue else {
                throw ModelError("GGUF metadata is missing \(key(suffix)) array — check --model")
            }
            guard let vals = arr.map(\.intValue) as? [Int] else {
                throw ModelError("\(key(suffix)) must be an integer array")
            }
            return vals
        }
        func floatArray(_ suffix: String) throws -> [Float] {
            guard let arr = gguf.kv(key(suffix))?.arrayValue else {
                throw ModelError("GGUF metadata is missing \(key(suffix)) array — check --model")
            }
            guard let vals = arr.map(\.doubleValue) as? [Double] else {
                throw ModelError("\(key(suffix)) must be a float array")
            }
            return vals.map(Float.init)
        }

        blockCount = try int("block_count")
        contextLength = try int("context_length")
        embeddingLength = try int("embedding_length")
        vocabSize = try int("vocab_size")
        headCount = try int("attention.head_count")
        headCountKV = try int("attention.head_count_kv")
        keyLength = try int("attention.key_length")
        valueLength = try int("attention.value_length")
        qLoraRank = try int("attention.q_lora_rank")
        outputLoraRank = try int("attention.output_lora_rank")
        outputGroupCount = try int("attention.output_group_count")
        slidingWindow = try int("attention.sliding_window")
        rmsNormEpsilon = try float("attention.layer_norm_rms_epsilon")
        compressRatios = try intArray("attention.compress_ratios")
        indexerHeadCount = try int("attention.indexer.head_count")
        indexerKeyLength = try int("attention.indexer.key_length")
        indexerTopK = try int("attention.indexer.top_k")
        ropeDimensionCount = try int("rope.dimension_count")
        ropeFreqBase = try float("rope.freq_base")
        ropeScalingType = try str("rope.scaling.type")
        ropeScalingFactor = try float("rope.scaling.factor")
        ropeOriginalContextLength = try int("rope.scaling.original_context_length")
        yarnBetaFast = try float("rope.scaling.yarn_beta_fast")
        yarnBetaSlow = try float("rope.scaling.yarn_beta_slow")
        compressRopeFreqBase = try float("attention.compress_rope_freq_base")
        expertCount = try int("expert_count")
        expertUsedCount = try int("expert_used_count")
        expertFeedForwardLength = try int("expert_feed_forward_length")
        expertSharedCount = try int("expert_shared_count")
        expertWeightsScale = try float("expert_weights_scale")
        expertWeightsNorm = try bool("expert_weights_norm")
        expertGatingFunc = try int("expert_gating_func")
        hashLayerCount = try int("hash_layer_count")
        nextNPredictLayers = try int("nextn_predict_layers")
        hyperConnectionCount = try int("hyper_connection.count")
        sinkhornIterations = try int("hyper_connection.sinkhorn_iterations")
        hyperConnectionEpsilon = try float("hyper_connection.epsilon")
        swigluClampExp = try floatArray("swiglu_clamp_exp")
        checkpointVariant = try str("checkpoint_variant")
        visionSidecarRequired = try bool("vision.sidecar_required")

        try validate(gguf: gguf)
    }

    /// `attention.compress_ratios` has one entry per compressed layer; for the
    /// reference checkpoint it holds 46 values for 43 blocks, so out-of-range
    /// layers have no recorded ratio and read nil.
    public func compressRatio(at layer: Int) -> Int? {
        compressRatios.indices.contains(layer) ? compressRatios[layer] : nil
    }

    /// Reject anything that is not the exact DeepSeek-V4-Flash (vision-exp)
    /// shape before any range construction, array indexing, or allocation can
    /// trap. Silently accepting a near-match only turns a useful `--model`
    /// error into a much later crash.
    public func validate(gguf: GGUFFile? = nil) throws {
        func bad(_ detail: String) throws -> Never {
            throw ModelError("unsupported or invalid DeepSeek-V4 GGUF (\(detail)) — check --model")
        }
        guard checkpointVariant == Self.supportedVariant else {
            try bad("checkpoint_variant '\(checkpointVariant)' is not the supported "
                + "'\(Self.supportedVariant)' variant; Pro and other variants are not supported")
        }
        guard embeddingLength == 4096, vocabSize == 129_280,
            headCount == 64, headCountKV == 1, keyLength == 512, valueLength == 512,
            qLoraRank == 1024, outputLoraRank == 1024, outputGroupCount == 8,
            slidingWindow == 128,
            ropeDimensionCount == 64, ropeScalingType == "yarn",
            ropeScalingFactor == 16, ropeOriginalContextLength == 65_536,
            yarnBetaFast == 32, yarnBetaSlow == 1, compressRopeFreqBase == 160_000,
            expertCount == 256, expertUsedCount == 6, expertFeedForwardLength == 2048,
            expertSharedCount == 1, expertWeightsScale == 1.5, expertWeightsNorm,
            hashLayerCount == 3,
            hyperConnectionCount == 4, sinkhornIterations == 20,
            indexerHeadCount == 64, indexerKeyLength == 128, indexerTopK == 512
        else {
            try bad("checkpoint geometry does not match DeepSeek-V4-Flash (vision-exp)")
        }
        guard blockCount > 0, contextLength > 0,
            expertUsedCount <= expertCount,
            headCount > 0, headCountKV > 0, headCount % headCountKV == 0,
            keyLength > 0, valueLength > 0, qLoraRank > 0, outputLoraRank > 0,
            outputGroupCount > 0, slidingWindow > 0, ropeDimensionCount > 0,
            expertFeedForwardLength > 0, expertSharedCount > 0,
            hyperConnectionCount > 0, sinkhornIterations > 0,
            indexerHeadCount > 0, indexerKeyLength > 0, indexerTopK > 0,
            rmsNormEpsilon.isFinite, rmsNormEpsilon > 0,
            ropeFreqBase.isFinite, ropeFreqBase > 0,
            hyperConnectionEpsilon.isFinite, hyperConnectionEpsilon > 0,
            expertWeightsScale.isFinite, expertWeightsScale > 0
        else {
            try bad("non-positive, non-finite, or unsupported dimensions")
        }
        guard !swigluClampExp.isEmpty, swigluClampExp.count == blockCount,
            swigluClampExp.allSatisfy({ $0.isFinite && $0 >= 0 })
        else {
            try bad("swiglu_clamp_exp must have one finite non-negative entry per block")
        }
        guard !compressRatios.isEmpty,
            compressRatios.allSatisfy({ $0 >= 0 })
        else {
            try bad("compress_ratios must be non-negative")
        }
        guard let gguf else { return }

        // Tokenizer keys the export path needs must be present.
        for required in ["tokenizer.ggml.tokens", "tokenizer.ggml.merges",
                         "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"] {
            guard gguf.kv(required) != nil else {
                try bad("GGUF is missing \(required)")
            }
        }

        // Spot-check the tensor directory against the validated geometry so a
        // mismatched checkpoint cannot pass on metadata alone.
        func dims(of name: String) throws -> [Int] {
            guard let t = gguf.tensor(named: name) else {
                try bad("GGUF is missing tensor \(name)")
            }
            return t.dims
        }
        let embd = try dims(of: "token_embd.weight")
        guard embd == [embeddingLength, vocabSize] else {
            try bad("token_embd.weight is \(embd), expected [\(embeddingLength), \(vocabSize)]")
        }
        let gate = try dims(of: "blk.0.ffn_gate_exps.weight")
        guard gate.count == 3, gate[2] == expertCount else {
            try bad("blk.0.ffn_gate_exps.weight is \(gate), expected [·, ·, \(expertCount)]")
        }
        let tid2eid = try dims(of: "blk.0.ffn_gate_tid2eid.weight")
        guard tid2eid == [expertUsedCount, vocabSize] else {
            try bad("blk.0.ffn_gate_tid2eid.weight is \(tid2eid), expected [\(expertUsedCount), \(vocabSize)]")
        }
    }
}
