import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen3MoEDirectReductionTests: XCTestCase {
    private func makeBlock() throws -> Qwen3MoESparseMoeBlock {
        let json = """
            {
                "model_type": "qwen3_moe",
                "vocab_size": 128,
                "hidden_size": 64,
                "intermediate_size": 128,
                "moe_intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_experts": 16,
                "num_experts_per_tok": 8,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "rms_norm_eps": 1e-6,
                "norm_topk_prob": true
            }
            """
        let configuration = try JSONDecoder().decode(
            Qwen3MoEConfiguration.self, from: Data(json.utf8))
        let block = Qwen3MoESparseMoeBlock(configuration)

        // Exercise production eligibility: BF16 router scores and quantized
        // expert projections, as in the quantized inference model.
        try block.update(
            parameters: ModuleParameters.unflattened([
                "gate.weight": MLXRandom.normal([16, 64]).asType(.bfloat16)
            ]),
            verify: [])
        quantize(model: block.switchMLP, groupSize: 32, bits: 4)
        eval(block)
        return block
    }

    private func routed(
        _ input: MLXArray, block: Qwen3MoESparseMoeBlock
    ) -> (indices: MLXArray, scores: MLXArray) {
        let gates = block.gate(input)
        let softGates = MLX.softmax(gates, axis: -1, precise: true)

        if supportsFusedRouterTopK(gates, k: block.topK) {
            return fusedRouterTopK(
                selection: gates,
                values: softGates,
                k: block.topK,
                normalize: block.normTopkProb,
                order: .descending)
        }

        let indices = MLX.argPartition(
            -gates, kth: block.topK - 1, axis: -1)[.ellipsis, ..<block.topK]
        var scores = MLX.takeAlong(softGates, indices, axis: -1)
        if block.normTopkProb {
            scores = scores / MLX.sum(scores, axis: -1, keepDims: true)
        }
        return (indices, scores)
    }

    private func legacyOutput(
        _ input: MLXArray, block: Qwen3MoESparseMoeBlock
    ) -> MLXArray {
        let (indices, scores) = routed(input, block: block)
        let tokenCount = input.size / input.dim(-1)
        let flatInput = input.reshaped(tokenCount, input.dim(-1))
        let flatIndices = indices.reshaped(tokenCount, block.topK)
        let flatScores = scores.reshaped(tokenCount, block.topK)
        return weightedExpertSum(
            block.switchMLP(flatInput, flatIndices), flatScores
        ).reshaped(input.shape)
    }

    func testPrefillDirectReductionMatchesLegacyQwen3MoEOutputExactly() throws {
        MLXRandom.seed(37)
        let block = try makeBlock()
        let input = MLXRandom.normal([1, 8, 64]).asType(.bfloat16)
        let (indices, scores) = routed(input, block: block)
        let flatInput = input.reshaped(8, 64)
        let flatIndices = indices.reshaped(8, block.topK)
        let flatScores = scores.reshaped(8, block.topK)

        XCTAssertTrue(
            block.switchMLP.supportsDirectWeightedReduction(
                flatInput, flatIndices, weights: flatScores),
            "The prefill fixture must exercise direct-kernel eligibility")

        let expected = legacyOutput(input, block: block)
        let actual = block(input)
        eval(expected, actual)
        XCTAssertEqual(actual.shape, input.shape)
        XCTAssertTrue(allClose(actual, expected, rtol: 0, atol: 0).item(Bool.self))
    }

    func testSingleTokenDecodeRetainsLegacyReductionExactly() throws {
        MLXRandom.seed(41)
        let block = try makeBlock()
        let input = MLXRandom.normal([1, 1, 64]).asType(.bfloat16)
        let (indices, scores) = routed(input, block: block)

        XCTAssertFalse(
            block.switchMLP.supportsDirectWeightedReduction(
                input.reshaped(1, 64),
                indices.reshaped(1, block.topK),
                weights: scores.reshaped(1, block.topK)),
            "A decode-sized route set must stay on the established path")

        let expected = legacyOutput(input, block: block)
        let actual = block(input)
        eval(expected, actual)
        XCTAssertEqual(actual.shape, input.shape)
        XCTAssertTrue(allClose(actual, expected, rtol: 0, atol: 0).item(Bool.self))
    }
}
