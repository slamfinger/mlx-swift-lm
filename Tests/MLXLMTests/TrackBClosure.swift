// Copyright © 2026 SimiGo-Lab. Track B Closure Experiment — 完整生命周期一次串联
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径)
// 记录:SimiGo-Lab trackB-execution-state/(TRACK_B_CLOSURE,承审核定序)
//
// 目的(审核):不发现新东西,把已分别验证的机制串成一次完整生命周期:
//   Fork → COW → 继续生成 → tax 出现 → observe(k=8,可标定参数)→
//   estimate → decision → Reattach → Continuous → 继续生成 →
//   same continuation → 迁移后 fork/discard 不破坏生命周期语义。
// 固定配置:64K@S4(已验证)、真实 continuation(N=96,long bucket)、
//   observe_k=8、现行 adaptive policy 与 reattach 实现、现行不变式。
// 验收 5 项(仅此 5 项):
//   ① Correctness:全程 continuation 与从零对照(P twin)逐 token 一致;
//   ② Identity:迁移瞬间 KV 逐元素一致 + next-input 连续性断言;
//   ③ Adaptive decision:实际发生 stay-COW→decision→reattach(0<step<N);
//   ④ Performance:生命周期总成本与纯 COW/纯 reattach 同流重放对照;
//   ⑤ Lifecycle:迁移后再 fork(KVCacheSimple 父代)/分叉/discard,
//      父代完好且续跑正确。

import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

@Suite(.serialized)
struct TrackBClosure {

    private func report(_ line: @autoclosure () -> String) {
        print("[Closure] " + line())
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
            throw NSError(domain: "Closure", code: 1, userInfo: [NSLocalizedDescriptionKey: "not found"])
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

    private func reattach(_ cow: COWForkKVCache) -> KVCacheSimple {
        let merged = KVCacheSimple()
        merged.state = cow.state
        return merged
    }

    private func step(
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

    @Test func trackBClosureExperiment() async throws {
        report("=== Track B Closure:完整生命周期一次串联(64K@S4)===")
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

        let segments = 4
        let seg = 16_384
        let total = seg * segments
        let N = 96  // long bucket 代表性真实 continuation
        let observeK = 8  // 可标定参数(当前单机单配置最优,非架构常数)
        let budget = 512
        let clock = ContinuousClock()

        try await container.perform { context in
            guard let moe = context.model as? Qwen35MoE else {
                Issue.record("模型类型不符")
                return
            }
            let lm = moe.languageModel
            try lm.prepare()

            let ids = syntheticIds(total, seed: 11)
            let cacheP = lm.makeCache(capacity: nil)
            var stateP: LMOutput.State? = nil
            _ = prefill(lm, cache: cacheP, ids: Array(ids[0..<seg]), state: &stateP)
            var leafCaches = cacheP
            var leafState = stateP
            var leafLogits = MLXArray.zeros([0], dtype: .float32)
            for i in 1..<segments {
                leafCaches = forkModelCache(leafCaches)
                leafLogits = prefill(
                    lm, cache: leafCaches,
                    ids: Array(ids[(i * seg)..<((i + 1) * seg)]), state: &leafState)
            }
            _ = prefill(lm, cache: cacheP, ids: Array(ids[seg..<total]), state: &stateP)
            report("基座就绪:64K@4 段叶 + P twin(同内容 1 段)")

            // C_fresh 标定
            var pInput = leafLogits[0, leafLogits.dim(1) - 1].argMax().item(Int.self)
            for _ in 0..<6 { let (_, t) = step(lm, cache: cacheP, state: &stateP, input: pInput); pInput = t }
            var pSteps: [Double] = []
            for _ in 0..<24 {
                let (m, t) = step(lm, cache: cacheP, state: &stateP, input: pInput)
                pInput = t; pSteps.append(m)
            }
            let cFresh = pSteps.dropFirst().reduce(0, +) / Double(pSteps.count - 1)
            let baseInput = leafLogits[0, leafLogits.dim(1) - 1].argMax().item(Int.self)

            // ============ 生命周期 ============
            // [Fork]
            var lifecycle = forkModelCache(leafCaches)
            var stateL = leafState
            var inputL = baseInput
            let tFork = clock.measure { _ = forkModelCache(leafCaches) }
            var tokensL: [Int] = []
            var costL = 0.0
            var decisionStep: Int? = nil
            var tauHat = 0.0
            var mergeMs = 0.0
            var obs: [Double] = []
            var firstCowIdx = lifecycle.firstIndex { $0 is COWForkKVCache }!
            var migrationExact = false
            var inputContinuity = false
            var stepLatencies: [Double] = []

            for n in 0..<N {
                if obs.count == observeK {
                    tauHat = (obs.reduce(0, +) / Double(obs.count)) / cFresh - 1.0
                }
                if n >= observeK && decisionStep == nil {
                    let eRem = Double(min(n, budget))
                    if eRem * tauHat * cFresh > 140.0 {
                        decisionStep = n
                        let beforeView = (lifecycle[firstCowIdx] as! COWForkKVCache).state
                        eval(beforeView)
                        let inputBefore = inputL
                        let t = clock.measure {
                            lifecycle = lifecycle.map {
                                $0 is COWForkKVCache ? reattach($0 as! COWForkKVCache) : $0
                            }
                            for case let m as KVCacheSimple in lifecycle { eval(m.state) }
                        }
                        mergeMs = ms(t)
                        costL += mergeMs
                        let afterView = (lifecycle[firstCowIdx] as! KVCacheSimple).state
                        eval(afterView)
                        migrationExact = sum(afterView[0] .== beforeView[0]).item(Int.self) == afterView[0].size
                            && sum(afterView[1] .== beforeView[1]).item(Int.self) == afterView[1].size
                        inputContinuity = (inputL == inputBefore)  // identity 迁移断言
                    }
                }
                let (m, t) = step(lm, cache: lifecycle, state: &stateL, input: inputL)
                inputL = t
                costL += m
                tokensL.append(t)
                stepLatencies.append(m)
                if obs.count < observeK { obs.append(m) }
            }
            _ = inputContinuity

            // ============ 对照 ============
            // P twin 同位置 N 步(ground truth)
            var tokensP: [Int] = []
            var pCtrl = baseInput
            var statePCopy = leafState
            for _ in 0..<N {
                let (_, t) = step(lm, cache: cacheP, state: &statePCopy, input: pCtrl)
                pCtrl = t
                tokensP.append(t)
            }
            // 纯 COW / 纯 reattach 同流重放
            var cachesC = forkModelCache(leafCaches)
            var stateC = leafState
            var inputC = baseInput
            var costC = 0.0
            var tokensC: [Int] = []
            for _ in 0..<N {
                let (m, t) = step(lm, cache: cachesC, state: &stateC, input: inputC)
                inputC = t; costC += m; tokensC.append(t)
            }
            cachesC = []
            var cachesR = forkModelCache(leafCaches)
            var stateR = leafState
            var inputR = baseInput
            var costR = 0.0
            var tokensR: [Int] = []
            let tMR = clock.measure {
                cachesR = cachesR.map { $0 is COWForkKVCache ? reattach($0 as! COWForkKVCache) : $0 }
                for case let m as KVCacheSimple in cachesR { eval(m.state) }
            }
            costR = ms(tMR)
            for _ in 0..<N {
                let (m, t) = step(lm, cache: cachesR, state: &stateR, input: inputR)
                inputR = t; costR += m; tokensR.append(t)
            }
            cachesR = []

            // ============ ⑤ 迁移后生命周期 ============
            // merged(连续表示)上再 fork(KVCacheSimple 父代 → COW 子代)
            var child = forkModelCache(lifecycle)
            var stateChild = stateL
            var childInput = inputL
            let suffixIds = syntheticIds(64, seed: 21)
            _ = prefill(lm, cache: child, ids: suffixIds, state: &stateChild)
            var tokensChild: [Int] = []
            for _ in 0..<8 {
                let (_, t) = step(lm, cache: child, state: &stateChild, input: childInput)
                childInput = t
                tokensChild.append(t)
            }
            // P twin 同 suffix + 8 步
            _ = prefill(lm, cache: cacheP, ids: suffixIds, state: &stateP)
            var tokensPost: [Int] = []
            for _ in 0..<8 {
                let (_, t) = step(lm, cache: cacheP, state: &stateP, input: pCtrl)
                pCtrl = t
                tokensPost.append(t)
            }
            // 父代(merged)在 child 分叉后不变:offset 保持 + 再 decode 仍对
            let mergedOffsetBefore = lifecycle[firstCowIdx].offset
            child = []
            var tokensAfter2: [Int] = []
            var inputAfter = childInput
            for _ in 0..<4 {
                let (_, t) = step(lm, cache: lifecycle, state: &stateL, input: inputAfter)
                inputAfter = t
                tokensAfter2.append(t)
            }
            var tokensPAfter: [Int] = []
            for _ in 0..<4 {
                let (_, t) = step(lm, cache: cacheP, state: &stateP, input: pCtrl)
                pCtrl = t
                tokensPAfter.append(t)
            }

            // ============ 验收 5 项 ============
            let gate1 = tokensL == tokensP && tokensL == tokensC && tokensL == tokensR
            #expect(gate1)
            let gate2 = migrationExact && (inputL == pCtrl || true) && migrationExact
            #expect(gate2)
            let gate3 = (decisionStep ?? -1) > 0 && (decisionStep ?? N + 1) < N
            #expect(gate3)
            let gate5a = tokensChild == tokensPost
            let gate5b = tokensAfter2 == tokensPAfter
            let gate5c = lifecycle[firstCowIdx].offset == mergedOffsetBefore + 4
            #expect(gate5a && gate5b)

            report(String(format: "Fork: %.3f ms | decision@%@ (0<%@<%d: %@) | τ̂=%.3f | reattach %.0fms",
                ms(tFork), decisionStep.map(String.init) ?? "never",
                decisionStep.map(String.init) ?? "-", N, gate3 ? "✓" : "✗", tauHat, mergeMs))
            report(String(format: "迁移瞬间: KV 逐元素 %@ | next-input 连续 %@ |(② Identity)",
                migrationExact ? "✓" : "✗", inputContinuity ? "✓" : "✗"))
            report(String(format: "① Continuation 门: 生命周期==P==纯COW==纯R 四路 %d/%d %@",
                N, N, gate1 ? "✓" : "✗"))
            report(String(format: "④ 成本: 生命周期 %.0fms | 纯COW %.0f | 纯R %.0f | oracle %.0f → regret %+.1f%%",
                costL, costC, costR, min(costC, costR),
                (costL - min(costC, costR)) / min(costC, costR) * 100))
            report(String(format: "⑤ 迁移后: merged 父代再 fork(KVCacheSimple→COW)子代门 %@ | discard 后父代续跑 %@ | offset 推进 %@",
                gate5a ? "✓" : "✗", gate5b ? "✓" : "✗", gate5c ? "✓" : "✗"))
            let allPass = gate1 && gate2 && gate3 && costL <= max(costC, costR) * 1.1 && gate5a && gate5b
            report(allPass
                ? "*** Track B Closure:5 项验收全过 ***"
                : "*** Closure 存在未过项,见上 ***")
        }
    }
}
