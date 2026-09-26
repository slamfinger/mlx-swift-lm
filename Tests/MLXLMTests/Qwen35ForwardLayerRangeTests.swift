// Copyright © 2026 Mr.Simi.
//
// Math-equivalence tests for `Qwen35Model.forwardLayerRange`: a full-range
// call must reproduce the model's normal forward, and chained sub-range
// calls must reproduce a single full-range call. These prove that the
// segment-execution API changes the physical execution topology without
// changing the model's mathematical semantics.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen35ForwardLayerRangeTests: XCTestCase {

    private func makeTinyMoEModel() throws -> Qwen35Model {
        let json = """
            {
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "hidden_size": 64,
                    "num_hidden_layers": 4,
                    "intermediate_size": 128,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 16,
                    "linear_num_value_heads": 4,
                    "linear_num_key_heads": 2,
                    "linear_key_head_dim": 16,
                    "linear_value_head_dim": 16,
                    "linear_conv_kernel_dim": 4,
                    "vocab_size": 128,
                    "full_attention_interval": 2,
                    "num_experts": 4,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 64,
                    "shared_expert_intermediate_size": 64
                }
            }
            """
        let config = try JSONDecoder().decode(
            Qwen35Configuration.self, from: Data(json.utf8))
        return withRandomState(MLXRandom.RandomState(seed: 42)) {
            Qwen35Model(config)
        }
    }

    private func tokens(_ count: Int, vocab: Int = 128) -> MLXArray {
        // Deterministic token IDs — avoids MLXRandom.randint API differences.
        let values = (0 ..< count).map { Int32(($0 * 7 + 3) % vocab) }
        return MLXArray(values, [1, count])
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = abs(a - b).max().item(Float.self)
        return diff
    }

    /// A single `forwardLayerRange(0..<all)` + `projectOutput` must produce
    /// logits identical to the model's standard full forward.
    func testFullRangeMatchesStandardForward() throws {
        let model = try makeTinyMoEModel()
        let input = tokens(8)

        // Standard path: model.callAsFunction → embeds, all layers, norm, lmHead.
        let standardLogits = model(input, cache: nil)

        // Segment path: embed → forwardLayerRange(0..<4) → projectOutput.
        let hidden = model.embedInputs(input)
        let layerOutput = model.forwardLayerRange(hidden, layerRange: 0..<4)
        let segmentLogits = model.projectOutput(layerOutput)

        eval(standardLogits, segmentLogits)
        let diff = maxAbsDiff(standardLogits, segmentLogits)
        // Bit-for-bit for bf16 (same ops in same order; no reordering).
        XCTAssertLessThanOrEqual(diff, 0.0, "full-range max_abs_diff = \(diff)")
    }

    /// Chained sub-range calls (0..<2 then 2..<4) must produce the same
    /// logits as a single full-range call. This proves that segment-wise
    /// execution does not change the mathematical result.
    func testChainedSegmentsMatchFullRange() throws {
        let model = try makeTinyMoEModel()
        let input = tokens(8)

        // Control: full-range segment call.
        let hidden0 = model.embedInputs(input)
        let fullOutput = model.forwardLayerRange(hidden0, layerRange: 0..<4)
        let fullLogits = model.projectOutput(fullOutput)

        // Chained: two segments, no final norm at the boundary.
        let seg1 = model.forwardLayerRange(hidden0, layerRange: 0..<2)
        let seg2 = model.forwardLayerRange(seg1, layerRange: 2..<4)
        let chainedLogits = model.projectOutput(seg2)

        eval(fullLogits, chainedLogits)
        let diff = maxAbsDiff(fullLogits, chainedLogits)
        XCTAssertLessThanOrEqual(diff, 0.0, "chained max_abs_diff = \(diff)")
    }

    /// Three-way chained (0..<1, 1..<3, 3..<4) must also match. This tests
    /// an uneven segment boundary that crosses the linear/full attention
    /// layer-type transition (layer 0 = linear, layer 1 = full, layer 2 =
    /// linear, layer 3 = full with full_attention_interval = 2).
    func testUnevenChainedSegmentsMatchFullRange() throws {
        let model = try makeTinyMoEModel()
        let input = tokens(8)

        let hidden0 = model.embedInputs(input)
        let fullOutput = model.forwardLayerRange(hidden0, layerRange: 0..<4)
        let fullLogits = model.projectOutput(fullOutput)

        let seg1 = model.forwardLayerRange(hidden0, layerRange: 0..<1)
        let seg2 = model.forwardLayerRange(seg1, layerRange: 1..<3)
        let seg3 = model.forwardLayerRange(seg2, layerRange: 3..<4)
        let chainedLogits = model.projectOutput(seg3)

        eval(fullLogits, chainedLogits)
        let diff = maxAbsDiff(fullLogits, chainedLogits)
        XCTAssertLessThanOrEqual(diff, 0.0, "uneven chained max_abs_diff = \(diff)")
    }
}
