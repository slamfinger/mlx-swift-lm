// Copyright © 2026 SimiGo-Lab. F5-3 Step 2/3 — 真机四策略 + 在线估计器验证
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径)
// 记录:SimiGo-Lab trackB-execution-state/(F5-3 S2/S3,承审核收紧的验收)
//
// Step 2(真机 policy):同一 continuation(同 token 流)四路重放——
//   always-COW / always-reattach / adaptive / oracle。
//   oracle = min(COW 实测, Reattach 实测)(免假设下界,非运行策略)。
//   Decision Regret = C_adaptive − min(C_cow, C_reattach) 为主指标。
//   每试次全量遥测(审核清单):N / decision step / τ̂ / τ_actual /
//   E[rem] / actual remaining / merge 估计与实测 / 决策 / 四路总延迟。
//   跨策略 token 门:三路重放 token 序列必须一致(execution identity
//   不变式的最强形式:表示选择不得改变 continuation)。
// Step 3(估计器):tax estimation error = |τ̂−τ|/τ 对 k∈{2,4,8}
//   ——用 always-COW 路前若干步的实测延迟后验计算,不额外跑;
//   决策稳定性 = 各 k 的事后决策与 oracle 择优的一致率。
// 配置:64K@S4(F5-2 最强信号格之一);C_fresh 用 P 侧实测标定一次
//   (生产合法:无 fresh twin 依赖,一次标定线)。

import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

@Suite(.serialized)
struct ExecutionStateForkF5Live {

    private func report(_ line: @autoclosure () -> String) {
        print("[F5L] " + line())
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
            throw NSError(domain: "F5L", code: 1, userInfo: [NSLocalizedDescriptionKey: "not found"])
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

    /// 一步 greedy(返回耗时 ms 与新 token)
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

    @Test func f5Step2And3LivePolicies() async throws {
        report("=== F5-3 S2/S3: 真机四策略 + 估计器(64K@S4)===")
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
        let budget = 512
        let observeK = 4
        let buckets: [(String, Int, Int)] = [
            ("short", 2, 8), ("mid", 16, 32), ("long", 64, 128), ("xlong", 256, 512),
        ]
        let trialsPerBucket = 3
        let clock = ContinuousClock()

        try await container.perform { context in
            var rng = SeededRandom(seed: 42)
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
            report("构链完成 64K@4段")

            // C_fresh 标定:P 侧 warm-up 6 + 测 24 步
            var pInput = leafLogits[0, leafLogits.dim(1) - 1].argMax().item(Int.self)
            for _ in 0..<6 {
                let (_, t) = step(lm, cache: cacheP, state: &stateP, input: pInput)
                pInput = t
            }
            var pSteps: [Double] = []
            for _ in 0..<24 {
                let (m, t) = step(lm, cache: cacheP, state: &stateP, input: pInput)
                pInput = t
                pSteps.append(m)
            }
            let cFresh = pSteps.dropFirst().reduce(0, +) / Double(pSteps.count - 1)
            report(String(format: "C_fresh 标定 = %.2f ms/token(P 侧 24 步稳态)", cFresh))

            // 基础 continuation 起点(所有试次同一起点,同 token 流)
            let baseInput = leafLogits[0, leafLogits.dim(1) - 1].argMax().item(Int.self)

            var grand = ["cow": 0.0, "always": 0.0, "ad": 0.0, "oracle": 0.0]
            var bucketStats: [String: (cow: Double, always: Double, ad: Double, oracle: Double, n: Int, regretSum: Double, adReatt: Int, orReatt: Int)] = [:]
            var estErrors: [Int: [Double]] = [2: [], 4: [], 8: []]
            var decisionConsistency: [Int: Int] = [2: 0, 4: 0, 8: 0]
            var decisionTotal = 0

            for (bname, blo, bhi) in buckets {
                for trial in 1...trialsPerBucket {
                    let N = rng.int(blo, bhi)

                    // ---- 路 A:always-COW ----
                    var cachesA = forkModelCache(leafCaches)
                    var stateA = leafState
                    var inputA = baseInput
                    var stepsA: [Double] = []
                    var tokensA: [Int] = []
                    for _ in 0..<N {
                        let (m, t) = step(lm, cache: cachesA, state: &stateA, input: inputA)
                        inputA = t
                        stepsA.append(m)
                        tokensA.append(t)
                    }
                    let costA = stepsA.reduce(0, +)
                    cachesA = []

                    // ---- 路 R:always-reattach ----
                    var cachesR = forkModelCache(leafCaches)
                    var stateR = leafState
                    var inputR = baseInput
                    var tokensR: [Int] = []
                    let tMergeR = clock.measure {
                        cachesR = cachesR.map { $0 is COWForkKVCache ? reattach($0 as! COWForkKVCache) : $0 }
                        for case let m as KVCacheSimple in cachesR { eval(m.state) }
                    }
                    var costR = ms(tMergeR)
                    for _ in 0..<N {
                        let (m, t) = step(lm, cache: cachesR, state: &stateR, input: inputR)
                        inputR = t
                        costR += m
                        tokensR.append(t)
                    }
                    cachesR = []

                    // ---- 路 AD:adaptive(observe→滚动决策,可中途 reattach)----
                    var cachesAD = forkModelCache(leafCaches)
                    var stateAD = leafState
                    var inputAD = baseInput
                    var tokensAD: [Int] = []
                    var costAD = 0.0
                    var decisionStep: Int? = nil
                    var tauHat = 0.0
                    var mergeActual = 0.0
                    var obsSteps: [Double] = []
                    for n in 0..<N {
                        if n == min(observeK, N - 1) {
                            tauHat = (obsSteps.reduce(0, +) / Double(obsSteps.count)) / cFresh - 1.0
                        }
                        // 滚动决策(观察窗后)
                        if n >= observeK && decisionStep == nil {
                            let eRem = Double(min(n, budget))
                            if eRem * tauHat * cFresh > 2.0 * 70.0 {  // merge≈65ms(F5-2)
                                decisionStep = n
                                let tm = clock.measure {
                                    cachesAD = cachesAD.map {
                                        $0 is COWForkKVCache ? reattach($0 as! COWForkKVCache) : $0
                                    }
                                    for case let m as KVCacheSimple in cachesAD { eval(m.state) }
                                }
                                mergeActual = ms(tm)
                                costAD += mergeActual
                            }
                        }
                        let (m, t) = step(lm, cache: cachesAD, state: &stateAD, input: inputAD)
                        inputAD = t
                        costAD += m
                        tokensAD.append(t)
                        if obsSteps.count < observeK { obsSteps.append(m) }
                    }
                    cachesAD = []

                    // ---- oracle = min(A, R) 实测 ----
                    let costO = min(costA, costR)
                    let regret = costAD - costO
                    let tauActual = (stepsA.reduce(0, +) / Double(N)) / cFresh - 1.0

                    // ---- token 门(三路同 continuation = execution identity)----
                    let gate = tokensA == tokensR && tokensA == tokensAD
                    #expect(gate)

                    // ---- Step3:估计器误差与决策稳定性(后验,用 A 路步延迟)----
                    for k in [2, 4, 8] {
                        if N > k {
                            let tauK = (stepsA.prefix(k).reduce(0, +) / Double(k)) / cFresh - 1.0
                            if tauActual > 0.02 {
                                estErrors[k]!.append(abs(tauK - tauActual) / max(tauActual, 1e-9))
                            }
                            // 事后决策(各 k):在 n=k 处用 tauK 判断
                            let wouldFire = Double(min(k, budget)) * max(tauK, 0) * cFresh > 140.0
                            let oracleFire = costR < costA
                            if wouldFire == oracleFire { decisionConsistency[k]! += 1 }
                        }
                    }
                    decisionTotal += 1

                    // ---- 遥测一行(审核清单)----
                    report(
                        String(
                            format: "  [%@ #%d] N=%3d | τ̂=%.3f τ=%.3f | decision@%@ E[rem]@dec=%@ rem=%d | merge 实测 %.0fms | COW %6.0f R %6.0f AD %6.0f oracle %6.0f | regret %+7.0f (%+.1f%%) | 门 %@",
                            bname, trial, N, tauHat, tauActual,
                            decisionStep.map(String.init) ?? "never",
                            decisionStep.map { String(min($0, budget)) } ?? "-",
                            decisionStep.map { N - $0 } ?? N,
                            mergeActual, costA, costR, costAD, costO,
                            regret, regret / costO * 100, gate ? "✓" : "✗"))

                    grand["cow"]! += costA; grand["always"]! += costR
                    grand["ad"]! += costAD; grand["oracle"]! += costO
                    var b = bucketStats[bname] ?? (0, 0, 0, 0, 0, 0, 0, 0)
                    b.cow += costA; b.always += costR; b.ad += costAD; b.oracle += costO
                    b.n += 1; b.regretSum += regret / costO
                    b.adReatt += (decisionStep != nil ? 1 : 0)
                    b.orReatt += (costR < costA ? 1 : 0)
                    bucketStats[bname] = b
                }
            }

            // ---- 汇总 ----
            report("=== 汇总(64K@S4,12 试次)===")
            for (bname, _, _) in buckets {
                if let b = bucketStats[bname] {
                    report(
                        String(
                        format: "  %@: COW %6.0f | R %6.0f | AD %6.0f | oracle %6.0f | 平均 regret %+.1f%% | AD 迁移 %d/%d(oracle 择迁 %d)",
                        bname, b.cow / Double(b.n), b.always / Double(b.n), b.ad / Double(b.n),
                        b.oracle / Double(b.n), b.regretSum / Double(b.n) * 100, b.adReatt, b.n, b.orReatt))
                }
            }
            let n = Double(buckets.count * trialsPerBucket)
            report(
                String(
                    format: "  总: COW %.0f | R %.0f | AD %.0f | oracle %.0f → AD vs COW %+.1f%%,vs R %+.1f%%,regret %+.1f%%",
                    grand["cow"]! / n, grand["always"]! / n, grand["ad"]! / n, grand["oracle"]! / n,
                    (grand["ad"]! / grand["cow"]! - 1) * 100,
                    (grand["ad"]! / grand["always"]! - 1) * 100,
                    (grand["ad"]! / grand["oracle"]! - 1) * 100))
            for k in [2, 4, 8] {
                let errs = estErrors[k]!
                if !errs.isEmpty {
                    report(
                        String(
                            format: "  Step3 估计器 k=%d: |τ̂−τ|/τ 平均 %.1f%%(n=%d)| 事后决策与 oracle 择优一致 %d/%d",
                            k, errs.reduce(0, +) / Double(errs.count) * 100, errs.count,
                            decisionConsistency[k]!, decisionTotal))
                }
            }
            let _ = clock
        }
    }
}

/// 简单可复现随机数(避免 Swift Testing 环境差异)
struct SeededRandom {
    var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E3779B97F4A7C15 }
    mutating func int(_ lo: Int, _ hi: Int) -> Int {
        state ^= state >> 12
        state ^= state << 25
        state ^= state >> 27
        let r = UInt64(truncatingIfNeeded: state &* 0x2545F4914F6CDD1D)
        return lo + Int(r % UInt64(hi - lo + 1))
    }
}
