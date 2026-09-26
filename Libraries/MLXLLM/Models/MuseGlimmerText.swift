//
//  MuseGlimmerText.swift
//  mlx-swift-lm
//
// Text-tower port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/muse_glimmer.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct MuseGlimmerTextLLMConfiguration: Decodable, Sendable {
    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let postNormEps: Float
    public let attentionBias: Bool
    public let slidingWindow: Int
    public let qkScaleFactor: Float
    public let outputMultiplier: Float
    public let finalLogitSoftcapping: Float
    public let tieWordEmbeddings: Bool
    public let layerTypes: [String]
    public let layerRopeTheta: [Float]
    public let ropeParameters: [String: StringOrNumber]?
    public let packagedMLXFormat: Int?

    var ropeTheta: Float {
        ropeParameters?["rope_theta"]?.asFloat() ?? 500_000
    }

    var usesPackagedMLXFormat: Bool { packagedMLXFormat == 1 }
    var usesAttentionOutputGate: Bool { usesPackagedMLXFormat }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case postNormEps = "post_norm_eps"
        case attentionBias = "attention_bias"
        case slidingWindow = "sliding_window"
        case qkScaleFactor = "qk_scale_factor"
        case outputMultiplier = "output_multiplier"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case layerRopeTheta = "layer_rope_theta"
        case ropeParameters = "rope_parameters"
        case museGlimmerMLXFormat = "muse_glimmer_mlx_format"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "muse_glimmer_text"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 202_048
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 6_656
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 19_968
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 52
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        postNormEps = try c.decodeIfPresent(Float.self, forKey: .postNormEps) ?? 1e-8
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 2_048
        qkScaleFactor = try c.decodeIfPresent(Float.self, forKey: .qkScaleFactor) ?? 3.87
        outputMultiplier =
            try c.decodeIfPresent(Float.self, forKey: .outputMultiplier)
            ?? 0.196_116_135_138_184_04
        finalLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 20
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        ropeParameters = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        packagedMLXFormat = try c.decodeIfPresent(Int.self, forKey: .museGlimmerMLXFormat)


        let hiddenLayerCount = hiddenLayers
        let defaultRopeTheta =
            ropeParameters?["rope_theta"]?.asFloat() ?? 500_000
        let resolvedLayerTypes: [String]
        if let types = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            resolvedLayerTypes = types
        } else {
            resolvedLayerTypes = (0 ..< hiddenLayerCount).map { index in
                (hiddenLayerCount - 1 - index) % 4 == 0 ? "full_attention" : "sliding_attention"
            }
        }

        let resolvedLayerRopeTheta: [Float]
        if let thetaValues = try c.decodeIfPresent([Float].self, forKey: .layerRopeTheta) {
            resolvedLayerRopeTheta = thetaValues
        } else {
            resolvedLayerRopeTheta = resolvedLayerTypes.map {
                $0 == "full_attention" ? 0 : defaultRopeTheta
            }
        }

        layerTypes = resolvedLayerTypes
        layerRopeTheta = resolvedLayerRopeTheta
    }

}

final class MuseGlimmerTextLLMRMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

final class MuseGlimmerTextLLMCenteredRMSNorm: Module, UnaryLayer {
    @ModuleInfo var weight: MLXArray
    let eps: Float
    let usesOffsetWeights: Bool

    init(dimensions: Int, eps: Float, usesOffsetWeights: Bool = true) {
        self.usesOffsetWeights = usesOffsetWeights
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if usesOffsetWeights {
            MLXFast.rmsNorm(x, weight: 1 + weight, eps: eps)
        } else {
            MLXFast.rmsNorm(x, weight: weight, eps: eps)
        }
    }
}

final class MuseGlimmerTextLLMPlainRMSNorm: Module, UnaryLayer {
    @ModuleInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

final class MuseGlimmerTextLLMMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: MuseGlimmerTextLLMConfiguration) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class MuseGlimmerTextLLMAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "gate_proj") var gateProj: Linear?
    @ModuleInfo(key: "qk_norm") var qkNorm: MuseGlimmerTextLLMRMSNormNoScale
    @ModuleInfo var rope: RoPELayer

    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let qkScaleFactor: Float
    let usesDirectQKScale: Bool
    let useRope: Bool
    let isSliding: Bool
    let usesFusedOutputGate: Bool

    init(_ config: MuseGlimmerTextLLMConfiguration, layerIndex: Int) {
        attentionHeads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
        qkScaleFactor = config.qkScaleFactor
        usesDirectQKScale = config.usesPackagedMLXFormat
        scale = usesDirectQKScale
            ? config.qkScaleFactor / Float(config.headDim)
            : pow(Float(config.headDim), -0.5)

        let theta = config.layerRopeTheta[layerIndex]
        useRope = theta != 0
        isSliding = config.layerTypes[layerIndex] == "sliding_attention"

        let queryDim = config.attentionHeads * config.headDim
        let kvDim = config.kvHeads * config.headDim
        usesFusedOutputGate = config.usesAttentionOutputGate
        _queryProj.wrappedValue = Linear(
            config.hiddenSize,
            usesFusedOutputGate ? 2 * queryDim : queryDim,
            bias: config.attentionBias)
        _keyProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: config.attentionBias)
        _valueProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: config.attentionBias)
        if !usesFusedOutputGate {
            _gateProj.wrappedValue = Linear(config.hiddenSize, queryDim, bias: false)
        }
        _outputProj.wrappedValue = Linear(queryDim, config.hiddenSize, bias: config.attentionBias)
        _qkNorm.wrappedValue = MuseGlimmerTextLLMRMSNormNoScale(eps: config.rmsNormEps)

        let effectiveTheta = useRope ? theta : config.ropeTheta
        _rope.wrappedValue = RoPE(
            dimensions: config.headDim,
            traditional: false,
            base: effectiveTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)

        var queries = queryProj(x)
        var keys = keyProj(x).reshaped(batch, length, kvHeads, headDim)
        let values = valueProj(x).reshaped(batch, length, kvHeads, headDim)
        var attentionGate: MLXArray?
        if usesFusedOutputGate {
            let split = queries
                .reshaped(batch, length, attentionHeads, 2 * headDim)
                .split(parts: 2, axis: -1)
            queries = split[0]
            attentionGate = split[1].reshaped(batch, length, -1)
        } else {
            queries = queries.reshaped(batch, length, attentionHeads, headDim)
        }

        queries = qkNorm(queries)
        if !usesDirectQKScale {
            queries = queries * qkScaleFactor
        }
        queries = queries.transposed(0, 2, 1, 3)
        keys = qkNorm(keys).transposed(0, 2, 1, 3)

        let valuesBh = values.transposed(0, 2, 1, 3)

        if useRope {
            let offset = cache?.offset ?? 0
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
        }
        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: valuesBh,
            cache: cache,
            scale: scale,
            mask: mask)
        output = output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        if let gateProj {
            output = output * sigmoid(gateProj(x))
        }
        if let attentionGate {
            output = output * sigmoid(attentionGate)
        }
        return outputProj(output)
    }
}

final class MuseGlimmerTextLLMBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: MuseGlimmerTextLLMAttention
    @ModuleInfo var mlp: MuseGlimmerTextLLMMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: MuseGlimmerTextLLMCenteredRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm:
        MuseGlimmerTextLLMCenteredRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm:
        MuseGlimmerTextLLMCenteredRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        MuseGlimmerTextLLMCenteredRMSNorm

    let isSliding: Bool

    init(_ config: MuseGlimmerTextLLMConfiguration, layerIndex: Int) {
        _attention.wrappedValue = MuseGlimmerTextLLMAttention(config, layerIndex: layerIndex)
        _mlp.wrappedValue = MuseGlimmerTextLLMMLP(config)
        _inputLayerNorm.wrappedValue = MuseGlimmerTextLLMCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps,
            usesOffsetWeights: !config.usesPackagedMLXFormat)
        _postAttentionLayerNorm.wrappedValue = MuseGlimmerTextLLMCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.postNormEps,
            usesOffsetWeights: !config.usesPackagedMLXFormat)
        _preFeedforwardLayerNorm.wrappedValue = MuseGlimmerTextLLMCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps,
            usesOffsetWeights: !config.usesPackagedMLXFormat)
        _postFeedforwardLayerNorm.wrappedValue = MuseGlimmerTextLLMCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.postNormEps,
            usesOffsetWeights: !config.usesPackagedMLXFormat)
        isSliding = config.layerTypes[layerIndex] == "sliding_attention"
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var h = x + postAttentionLayerNorm(
            attention(inputLayerNorm(x), mask: mask, cache: cache))
        h = h + postFeedforwardLayerNorm(mlp(preFeedforwardLayerNorm(h)))
        return h
    }
}

final class MuseGlimmerTextLLMInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_norm") var embedNorm: MuseGlimmerTextLLMRMSNormNoScale
    @ModuleInfo var layers: [MuseGlimmerTextLLMBlock]
    @ModuleInfo var norm: RMSNorm

    let config: MuseGlimmerTextLLMConfiguration
    let fullAttentionIndex: Int
    let slidingAttentionIndex: Int?

    init(_ config: MuseGlimmerTextLLMConfiguration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _embedNorm.wrappedValue = MuseGlimmerTextLLMRMSNormNoScale(eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            MuseGlimmerTextLLMBlock(config, layerIndex: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        fullAttentionIndex = config.layerTypes.firstIndex(of: "full_attention") ?? 0
        slidingAttentionIndex = config.layerTypes.firstIndex(of: "sliding_attention")
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let h = embedNorm(embedTokens(inputs))
        let activeCaches = cache ?? []
        func cacheAt(index: Int) -> KVCache? {
            index < activeCaches.count ? activeCaches[index] : nil
        }
        let fullCache = cacheAt(index: fullAttentionIndex)
        let fullMask = createAttentionMask(
            h: h,
            cache: fullCache,
            windowSize: nil)
        var slidingMask = MLXFast.ScaledDotProductAttentionMaskMode.none
        if let slidingAttentionIndex {
            slidingMask = createAttentionMask(
                h: h,
                cache: cacheAt(index: slidingAttentionIndex),
                windowSize: config.slidingWindow)
        }

        var output = h
        for (index, layer) in layers.enumerated() {
            output = layer(
                output,
                mask: layer.isSliding ? slidingMask : fullMask,
                cache: cacheAt(index: index))
        }
        let normalized = norm(output)
        return normalized
    }
}

public final class MuseGlimmerTextLLMModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let configuration: MuseGlimmerTextLLMConfiguration

    @ModuleInfo var model: MuseGlimmerTextLLMInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: MuseGlimmerTextLLMConfiguration) {
        configuration = config
        vocabularySize = config.vocabularySize
        kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        _model.wrappedValue = MuseGlimmerTextLLMInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var logits = (model(inputs, cache: cache).asType(.float32)
            * configuration.outputMultiplier)
        let softcapping = configuration.finalLogitSoftcapping
        if softcapping > 0 {
            logits = tanh(logits / softcapping) * softcapping
        }
        return logits
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        if configuration.usesPackagedMLXFormat {
            // Packaged Muse-Glimmer artifacts use full-history caches for all
            // layers; window semantics are in the banded attention mask.
            return (0 ..< configuration.hiddenLayers).map { _ in KVCacheSimple() }
        }
        return try configuration.layerTypes.map { layerType in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: configuration.slidingWindow,
                usesSlidingWindow: layerType == "sliding_attention")
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("vision_tower.") || key.hasPrefix("vision_adapter.")
                || key.hasPrefix("vision_projection.") {
                continue
            }
            if key.hasPrefix("language_model.model.") {
                result["model." + key.dropFirst("language_model.model.".count)] = value
            } else if key.hasPrefix("language_model.lm_head.") {
                result["lm_head." + key.dropFirst("language_model.lm_head.".count)] = value
            } else if configuration.usesPackagedMLXFormat {
                var mapped = key
                if mapped.hasPrefix("model.layers.") {
                    if mapped.contains(".post_attn_norm.") {
                        mapped = mapped.replacingOccurrences(
                            of: ".post_attn_norm.", with: ".post_attention_layernorm.")
                    } else if mapped.contains(".post_attention_layernorm.") {
                        mapped = mapped.replacingOccurrences(
                            of: ".post_attention_layernorm.", with: ".pre_feedforward_layernorm.")
                    } else if mapped.contains(".post_ffn_norm.") {
                        mapped = mapped.replacingOccurrences(
                            of: ".post_ffn_norm.", with: ".post_feedforward_layernorm.")
                    }
                }
                result[mapped] = value
            } else {
                result[key] = value
            }
        }
        return result
    }
}

extension MuseGlimmerTextLLMModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
