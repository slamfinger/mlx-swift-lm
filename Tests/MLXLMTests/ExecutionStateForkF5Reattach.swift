// Copyright © 2026 SimiGo-Lab. F5-1 — Reattach 最小原型(表示形态迁移第一步)
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径)
// 记录:SimiGo-Lab trackB-execution-state/(F5-1,承外部审核定序)
//
// F5-1 只回答一个问题(审核原话):
//   "把多段共享状态重新组织成连续状态,是否真的能够把已经观察到的
//    1.256× 长深度税消掉,并且 merge 的一次性成本是否值得?"
// 范围:64K @4 段,不做自动策略,不优化。
// 验收三项:
//   1. 正确性:merge 前后逐元素一致 + merge 后逐 token 门(对 P 对照);
//   2. 性能:merge 后稳态 decode ≤ 1.05× fresh baseline;
//   3. 成本:完整 merge latency 与实际 copy 字节。
// 方法:同 F4-E(warm-up 后测、P/被测侧交替、深度锁步、首 token 分离)。
// 流程:P+链构到 64K(叶=4 段)→ warm-up → rep1(P vs 叶COW,合并前税)
//   → reattach(叶 → 单连续 KVCacheSimple ×10 attention 层;GDN 不动)
//   → rep2/rep3(P vs merged,合并后税 + token 门)。
//
// reattach 语义:state 赋值(惰性 concat 视图)→ eval(物化即合并,
//   单次 O(depth) 拷贝)→ 后续 KVCacheSimple 就地写。
//   注意:合并后 capacity==offset,首个 decode 步触发一次 step-256
//   增长拼接(KVCacheSimple 常态行为),计入首 token,不入稳态。

import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

@Suite(.serialized)
struct ExecutionStateForkF5Reattach {

    private func report(_ line: @autoclosure () -> String) {
        print("[F5] " + line())
    }

    private func ms(_ duration: Duration) -> Double {
        let ns = Double(duration.components.seconds) * 1e9
            + Double(duration.components.attoseconds) / 1e9
        return ns / 1e6
    }

    private func modelDirectory() throws -> URL {
        let hub = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        let name = "models--peculiar-ragdoll--Cyber-Tiel-Coder-35B-A3B-MLX-oQ4e"
        let snapshots = hub.appendingPathComponent(name).appendingPathComponent("snapshots")
        let dirs = try FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil
        ).filter { $0.hasDirectoryPath }.sorted { $0.path < $1.path }
        guard let dir = dirs.first else {
            throw NSError(domain: "F5", code: 1, userInfo: [NSLocalizedDescriptionKey: "not found"])
        }
        return dir
    }

    private func syntheticIds(_ n: Int, seed: Int) -> [Int] {
        (0..<n).map { i in 2000 + ((i * 7919 + seed * 104729) % 140_000) }
    }

    @discardableResult
    private func prefill(
        _ lm: Qwen35Language.LanguageModel,
        cache: [KVCache],
        ids: [Int],
        state: inout LMOutput.State?
    ) -> MLXArray {
        var logits = MLXArray.zeros([0], dtype: .float32)
        var start = 0
        while start < ids.count {
            let n = min(512, ids.count - start)
            let out = lm(
                MLXArray(Array(ids[start..<(start + n)]).map(Int32.init)),
                cache: cache.map { $0 as KVCache? },
                state: state)
            logits = out.logits
            state = out.state
            eval(logits)
            start += n
        }
        return logits
    }

    private func greedyStep(
        _ lm: Qwen35Language.LanguageModel,
        cache: [KVCache],
        state: inout LMOutput.State?,
        input: Int
    ) -> (Double, Int) {
        let clock = ContinuousClock()
        var next = input
        let t = clock.measure {
            let out = lm(
                MLXArray([Int32(next)]),
                cache: cache.map { $0 as KVCache? },
                state: state)
            state = out.state
            eval(out.logits)
            next = out.logits[0, out.logits.dim(1) - 1].argMax().item(Int.self)
        }
        let ns = Double(t.components.seconds) * 1e9 + Double(t.components.attoseconds) / 1e9
        return (ns / 1e6, next)
    }

    /// reattach:多段 COW → 单连续 KVCacheSimple。
    /// state 赋值 = 惰性视图;eval = 物化(即合并,单次 O(depth) 拷贝)。
    private func reattach(_ cow: COWForkKVCache) -> KVCacheSimple {
        let merged = KVCacheSimple()
        merged.state = cow.state  // [kView, vView] 惰性;offset = 逻辑深度
        return merged
    }

    @Test func f5_1ReattachMinimalPrototype() async throws {
        report("=== F5-1: Reattach 最小原型(64K @4 段)===")
        let dir: URL
        do {
            dir = try modelDirectory()
        } catch {
            Issue.record("模型不在本机,跳过: \(error)")
            return
        }
        GPU.set(memoryLimit: 24 * 1024 * 1024 * 1024)
        let container = try await VLMModelFactory.shared.loadContainer(
            from: dir, using: NoOpTokenizerLoader())
        report(String(format: "模型加载后 active=%dMiB", GPU.activeMemory / 1048576))

        let segments = 4
        let seg = 16_384
        let total = seg * segments
        let warmupSteps = 6
        let measuredSteps = 24
        let clock = ContinuousClock()

        try await container.perform { context in
            guard let moe = context.model as? Qwen35MoE else {
                Issue.record("模型类型不符")
                return
            }
            let lm = moe.languageModel
            try lm.prepare()

            let ids = syntheticIds(total, seed: 11)

            // P(首段)+ 链(叶 = 4 段)
            let cacheP = lm.makeCache(capacity: nil)
            var stateP: LMOutput.State? = nil
            _ = prefill(lm, cache: cacheP, ids: Array(ids[0..<seg]), state: &stateP)
            var leafCaches = cacheP
            var leafState = stateP
            for i in 1..<segments {
                leafCaches = forkModelCache(leafCaches)
                _ = prefill(
                    lm, cache: leafCaches,
                    ids: Array(ids[(i * seg)..<((i + 1) * seg)]), state: &leafState)
            }
            // P 解冻续写补至全深(fresh 1 段对照)
            _ = prefill(lm, cache: cacheP, ids: Array(ids[seg..<total]), state: &stateP)
            report(String(format: "构链完成 64K@4段 | active=%dMiB", GPU.activeMemory / 1048576))

            // warm-up
            var pInput = 0
            var lInput = 0
            var mInput = 0
            for _ in 0..<warmupSteps {
                let (_, t) = greedyStep(lm, cache: cacheP, state: &stateP, input: pInput)
                pInput = t
                let (_, t2) = greedyStep(lm, cache: leafCaches, state: &leafState, input: lInput)
                lInput = t2
            }

            func measure(
                _ cache: [KVCache], state: inout LMOutput.State?, input: inout Int
            ) -> (Double, Double, [Int]) {
                var steps: [Double] = []
                var tokens: [Int] = []
                for _ in 0..<measuredSteps {
                    let (m, t) = greedyStep(lm, cache: cache, state: &state, input: input)
                    input = t
                    steps.append(m)
                    tokens.append(t)
                }
                let steady = steps.dropFirst().reduce(0, +) / Double(steps.count - 1)
                return (steps[0], steady, tokens)
            }

            // ---- rep1:合并前(P 先,叶 COW 后)----
            let (_, steadyP1, pTokens1) = measure(cacheP, state: &stateP, input: &pInput)
            let (_, steadyL1, lTokens1) = measure(leafCaches, state: &leafState, input: &lInput)
            #expect(pTokens1 == lTokens1)
            report(
                String(
                    format: "合并前: P %.3f ms/token | 叶COW %.3f → 税 %.3f× | token 门 %@",
                    steadyP1, steadyL1, steadyL1 / steadyP1, pTokens1 == lTokens1 ? "✓" : "✗"))

            // ---- reattach ----
            // 正确性参照:首个 attention 层合并前视图(独立快照)
            let firstCowIdx = leafCaches.firstIndex { $0 is COWForkKVCache }!
            let beforeView = (leafCaches[firstCowIdx] as! COWForkKVCache).state
            eval(beforeView)

            var mergedCaches: [KVCache] = []
            let tMerge = clock.measure {
                mergedCaches = leafCaches.map { layer -> KVCache in
                    if let cow = layer as? COWForkKVCache {
                        return reattach(cow)
                    }
                    return layer  // GDN(MambaCache)不动
                }
                // 物化即合并:对全部 attention 层 eval
                for case let m as KVCacheSimple in mergedCaches {
                    eval(m.state)
                }
            }
            var mergedState = leafState
            mInput = lInput  // 合并侧续跑的下一输入 = 叶侧当前输入(rep1 门已证与 P 同步)
            let mergeDepth = mergedCaches[firstCowIdx].offset
            // 合并后逐元素一致(首层)
            let afterView = (mergedCaches[firstCowIdx] as! KVCacheSimple).state
            eval(afterView)
            let exact = sum(afterView[0] .== beforeView[0]).item(Int.self) == afterView[0].size
                && sum(afterView[1] .== beforeView[1]).item(Int.self) == afterView[1].size
            #expect(exact)
            // 释放叶 COW 引用(段表引用转移给 merged;GDN 对象转移)
            leafCaches = []
            report(
                String(
                    format: "reattach: %.1f ms(10 attention 层,逻辑 KV %.2f GiB,单次拷贝)| 合并后逐元素一致 %@ | active=%dMiB",
                    ms(tMerge), Double(total * 20 * 1024) / 1073741824.0, exact ? "✓" : "✗",
                    GPU.activeMemory / 1048576))
            report(String(format: "合并深度=%d(= 叶深度)", mergeDepth))

            // ---- rep2/rep3:合并后(P vs merged,交替)----
            var ratios: [Double] = []
            var gates = true
            for rep in 2...3 {
                if rep % 2 == 0 {
                    let (_, steadyP, pT) = measure(cacheP, state: &stateP, input: &pInput)
                    let (firstM, steadyM, mT) = measure(mergedCaches, state: &mergedState, input: &mInput)
                    gates = gates && (pT == mT)
                    ratios.append(steadyM / steadyP)
                    if rep == 2 { report(String(format: "  合并后首 token(含 step 增长拼接): %.1f ms", firstM)) }
                } else {
                    let (firstM, steadyM, mT) = measure(mergedCaches, state: &mergedState, input: &mInput)
                    let (_, steadyP, pT) = measure(cacheP, state: &stateP, input: &pInput)
                    gates = gates && (pT == mT)
                    ratios.append(steadyM / steadyP)
                    _ = firstM
                }
            }
            #expect(gates)
            let post = ratios.reduce(0, +) / Double(ratios.count)
            report(
                String(
                    format: "合并后稳态税: %@ → 均值 %.3f× | token 门 %@",
                    ratios.map { String(format: "%.3f", $0) }.joined(separator: "/"),
                    post, gates ? "✓ 48/48" : "✗"))

            // ---- 验收判定 ----
            let pass = post <= 1.05
            report(
                String(
                    format: "*** F5-1 验收:merge 后稳态 ≤1.05× fresh → %@(%.3f×)***",
                    pass ? "通过" : "未过", post))
            report(
                String(
                    format: "成本账:merge %.1f ms 一次性 vs 合并前税 %.1f ms/token × 预期 token 数(设计期 break-even 由 F5-2 矩阵定)",
                    ms(tMerge), steadyL1 - steadyP1))
        }
    }
}
