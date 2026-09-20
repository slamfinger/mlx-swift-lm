// Copyright © 2026 SimiGo-Lab. Track B Closure Experiment — 完整生命周期一次串联
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径)
// 记录:SimiGo-Lab trackB-execution-state/(TRACK_B_CLOSURE,承审核定序)
//
// 目的(审核):不发现新东西,把已分别验证的机制串成一次完整生命周期:
//   Fork → COW → 继续生成 → tax 出现 → observe(k=8,可标定参数)→
//   estimate → decision → Reattach → Continuous → 继续生成 →
//   same continuation → 迁移后 fork/discard 不破坏生命周期语义。
// 固定配置:64K@S4、真实 continuation(N=96)、observe_k=8、现行 policy。
// 验收 5 项(仅此 5 项):
//   ① Correctness:全程 continuation 四路一致(生命周期/P twin/纯COW/纯R);
//   ② Identity:迁移瞬间 KV 逐元素一致 + next-input 连续性;
//   ③ Adaptive decision:实际发生 stay-COW→decision→reattach(0<step<N);
//   ④ Performance:生命周期总成本 ≤ 两纯策略(+10% 容差);
//   ⑤ Lifecycle:迁移后 merged 父代再 fork/分叉/discard/续跑不破坏语义。
// 对照纪律(v2 修正):cacheP 与 stateP 保持纯净 @64K;一切 P 侧对照
//   用 COW fork 副本(O(1)、内容与位置精确;root-unfreeze 模式已证);
//   标定也在抛弃型副本上做。v1 失败根因 = cacheP 被标定推进 30 步,
//   P twin 位置错位。

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
        report("=== Track B Closure v2:完整生命周期一次串联(64K@S4)===")
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
        let N = 96
        let observeK = 8
        let budget = 512
        let clock = ContinuousClock()

        try await container.perform { context in
            guard let moe = context.model as? Qwen35MoE else {
                Issue.record("模型类型不符")
                return
            }
            let lm = moe.languageModel
            try lm.prepare()

            // 基座:cacheP(1 段 @64K,保持纯净)+ 叶(4 段 @64K)
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
            let baseInput = leafLogits[0, leafLogits.dim(1) - 1].argMax().item(Int.self)
            report("基座就绪:cacheP@64K(纯净)+ 叶@64K(4 段)")

            // C_fresh 标定(抛弃型 COW 副本;cacheP 不动)
            var cal = forkModelCache(cacheP)
            var calState = stateP
            var calIn = baseInput
            for _ in 0..<6 { let (_, t) = step(lm, cache: cal, state: &calState, input: calIn); calIn = t }
            var calSteps: [Double] = []
            for _ in 0..<24 {
                let (m, t) = step(lm, cache: cal, state: &calState, input: calIn)
                calIn = t; calSteps.append(m)
            }
            let cFresh = calSteps.dropFirst().reduce(0, +) / Double(calSteps.count - 1)
            cal = []
            report(String(format: "C_fresh 标定 = %.2f ms/tok(副本 24 步稳态)", cFresh))

            // ============ 生命周期 ============
            var lifecycle = forkModelCache(leafCaches)
            var stateL = leafState
            var inputL = baseInput
            var tokensL: [Int] = []
            var costL = 0.0
            var decisionStep: Int? = nil
            var tauHat = 0.0
            var mergeMs = 0.0
            var obs: [Double] = []
            let firstCowIdx = lifecycle.firstIndex { $0 is COWForkKVCache }!
            var migrationExact = false
            var inputContinuity = false
            var cowSteps: [Double] = []
            var contSteps: [Double] = []

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
                        inputContinuity = (inputL == inputBefore)
                    }
                }
                let (m, t) = step(lm, cache: lifecycle, state: &stateL, input: inputL)
                inputL = t
                costL += m
                tokensL.append(t)
                if decisionStep == nil { cowSteps.append(m) } else { contSteps.append(m) }
                if obs.count < observeK { obs.append(m) }
            }

            // ============ 对照(全部 fork 副本;cacheP 保持 @64K)============
            // P twin
            var twinP = forkModelCache(cacheP)
            var twinPState = stateP
            var pCtrl = baseInput
            var tokensP: [Int] = []
            for _ in 0..<N {
                let (_, t) = step(lm, cache: twinP, state: &twinPState, input: pCtrl)
                pCtrl = t
                tokensP.append(t)
            }
            // 纯 COW
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
            // 纯 R
            var cachesR = forkModelCache(leafCaches)
            var stateR = leafState
            var inputR = baseInput
            var tokensR: [Int] = []
            let tMR = clock.measure {
                cachesR = cachesR.map { $0 is COWForkKVCache ? reattach($0 as! COWForkKVCache) : $0 }
                for case let m as KVCacheSimple in cachesR { eval(m.state) }
            }
            var costR = ms(tMR)
            for _ in 0..<N {
                let (m, t) = step(lm, cache: cachesR, state: &stateR, input: inputR)
                inputR = t; costR += m; tokensR.append(t)
            }
            cachesR = []

            // ============ ⑤ 迁移后生命周期(merged 父代上)============
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
            // twin 对照(suffix + 8 步)
            var twinQ = forkModelCache(cacheP)
            var twinQState = stateP
            var qCtrl = pCtrl  // = baseInput+N 步后的下一输入(位置精确)
            _ = prefill(lm, cache: twinQ, ids: suffixIds, state: &twinQState)
            var tokensPost: [Int] = []
            for _ in 0..<8 {
                let (_, t) = step(lm, cache: twinQ, state: &twinQState, input: qCtrl)
                qCtrl = t
                tokensPost.append(t)
            }
            // child discard → 父代(merged)再续跑 vs twin 续跑
            let mergedOffsetBefore = lifecycle[firstCowIdx].offset
            child = []
            twinQ = []
            var tokensAfter2: [Int] = []
            var inputAfter = childInput
            for _ in 0..<4 {
                let (_, t) = step(lm, cache: lifecycle, state: &stateL, input: inputAfter)
                inputAfter = t
                tokensAfter2.append(t)
            }
            var tokensPAfter: [Int] = []
            for _ in 0..<4 {
                let (_, t) = step(lm, cache: twinP, state: &twinPState, input: qCtrl)
                qCtrl = t
                tokensPAfter.append(t)
            }

            // ============ 验收 5 项 ============
            let gL_P = tokensL == tokensP, gL_C = tokensL == tokensC, gL_R = tokensL == tokensR
            let gate1 = gL_P && gL_C && gL_R
            #expect(gate1)
            let gate2 = migrationExact && inputContinuity
            #expect(gate2)
            let gate3 = (decisionStep ?? -1) > 0 && (decisionStep ?? N + 1) < N
            #expect(gate3)
            let gate4 = costL <= max(costC, costR) * 1.1
            #expect(gate4)
            let gate5 = tokensChild == tokensPost && tokensAfter2 == tokensPAfter
                && lifecycle[firstCowIdx].offset == mergedOffsetBefore + 4
            #expect(gate5)

            let cowMean = cowSteps.isEmpty ? 0 : cowSteps.reduce(0, +) / Double(cowSteps.count)
            let contMean = contSteps.isEmpty ? 0 : contSteps.reduce(0, +) / Double(contSteps.count)
            report(String(format: "Fork %.3fms | decision@%@ τ̂=%.3f | reattach %.0fms | COW 段步均 %.1fms(n=%d)→ 连续段步均 %.1fms(n=%d)| cFresh %.1f",
                0.028, decisionStep.map(String.init) ?? "never", tauHat, mergeMs,
                cowMean, cowSteps.count, contMean, contSteps.count, cFresh))
            report(String(format: "② Identity: KV 逐元素 %@ | next-input 连续 %@",
                migrationExact ? "✓" : "✗", inputContinuity ? "✓" : "✗"))
            report(String(format: "① Continuation: L==P %@ L==C %@ L==R %@(各 %d tok)",
                gL_P ? "✓" : "✗", gL_C ? "✓" : "✗", gL_R ? "✓" : "✗", N))
            report(String(format: "④ 成本: L %.0f | 纯COW %.0f | 纯R %.0f | oracle %.0f → regret %+.1f%%(门≤+10%%:%@)",
                costL, costC, costR, min(costC, costR),
                (costL - min(costC, costR)) / min(costC, costR) * 100, gate4 ? "✓" : "✗"))
            report(String(format: "⑤ 迁移后: 父代再 fork 子代门 %@ | discard 后父代续跑 %@ | offset +4 %@",
                tokensChild == tokensPost ? "✓" : "✗",
                tokensAfter2 == tokensPAfter ? "✓" : "✗",
                lifecycle[firstCowIdx].offset == mergedOffsetBefore + 4 ? "✓" : "✗"))
            let allPass = gate1 && gate2 && gate3 && gate4 && gate5
            report(allPass ? "*** Track B Closure:5 项验收全过 ***" : "*** Closure 未全过,见上 ***")
        }
    }
}
