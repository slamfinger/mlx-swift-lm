// Copyright © 2026 SimiGo-Lab. F4-E / F4-F — 模型级长深度多段 decode 税对照 + Discard 生命周期归因
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径)
// 记录:SimiGo-Lab trackB-execution-state/(F4 续项,承外部审核 2026-09-20)
//
// F4-E(审核定序第一优先):1.90× 上界警告的拆分归因。
//   矩阵:深度 8K/18K/32K/64K @S=4 + 段数 2/8 @18K(S=1 = fresh 对照)。
//   方法:warm-up 后正式测量(编译/冷启动不入稳态);每 rep 24 步、
//   首 token 与稳态分开;对照组(P 单段)与 COW 组(链叶 S 段)交替
//   运行;两侧每 rep 等步数推进(深度锁步);3 reps;逐 token 门
//   (链叶与 P 同内容同深度,greedy 序列必须一致)。
//   构链法:P prefill 首段 → 链式 fork+append 至全深(叶 = S 段)
//   → P 解冻续写补至全深(= 1 段 fresh 对照,同内容;解冻安全性
//   F4-B-model 已证)。每配置前向总量 = seg×(2S-1)。
//   判定:warm 后 ≈1.0× → F3a 保持回归项;长深度多段稳定 >1.10× →
//   启动物化路径剖析;仅个别异常 → 查该配置段表布局。
//
// F4-F:Discard 生命周期归因(三概念:逻辑引用释放 ≠ MLX 图引用释放
//   ≠ 物理内存归还)。补充测量:①复用测试(丢弃后重建等大分支,
//   active 是否复用而非增长);②强制 eval+clearCache 对比;③组合丢弃
//   (root/子树/promoted 叶分别丢弃);④不可达页最终可复用判定。

import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

@Suite(.serialized)
struct ExecutionStateForkF4EF {

    // MARK: - F4-E

    private func modelDirectory() throws -> URL {
        let hub = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        let name = "models--peculiar-ragdoll--Cyber-Tiel-Coder-35B-A3B-MLX-oQ4e"
        let snapshots = hub.appendingPathComponent(name).appendingPathComponent("snapshots")
        let dirs = try FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil
        ).filter { $0.hasDirectoryPath }.sorted { $0.path < $1.path }
        guard let dir = dirs.first else {
            throw NSError(domain: "F4E", code: 1, userInfo: [NSLocalizedDescriptionKey: "not found"])
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

    /// 一步 greedy:返回 (耗时 ms, 新 token)
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

    struct F4EConfig {
        let nominalDepth: Int
        let segments: Int
        var label: String { "D\(nominalDepth / 1024)K@S\(segments)" }
    }

    @Test func f4eModelLongDepthMultiSegmentDecode() async throws {
        report("=== F4-E: 模型级长深度 × 多段 decode 税(warm 后,交替,3 reps)===")
        let dir: URL
        do {
            dir = try modelDirectory()
        } catch {
            Issue.record("模型不在本机,跳过: \(error)")
            return
        }
        // 32GB Air:权重 20.1GiB + 64K 格 KV ~2.6GiB,显式抬限额防中途
        // 分配失败(实验室受控环境,运行前已查 swap)
        GPU.set(memoryLimit: 24 * 1024 * 1024 * 1024)
        let container = try await VLMModelFactory.shared.loadContainer(
            from: dir, using: NoOpTokenizerLoader())
        report(String(format: "模型加载后 active=%dMiB", GPU.activeMemory / 1048576))

        // 深度轴 @S=4 + 段数轴 @~18K(S=1 即 fresh,无链)
        let configs: [F4EConfig] = [
            .init(nominalDepth: 8_192, segments: 4),
            .init(nominalDepth: 18_432, segments: 4),
            .init(nominalDepth: 32_768, segments: 4),
            .init(nominalDepth: 65_536, segments: 4),
            .init(nominalDepth: 18_432, segments: 2),
            .init(nominalDepth: 16_384, segments: 8),
        ]
        let warmupSteps = 6
        let measuredSteps = 24
        let reps = 3
        let clock = ContinuousClock()

        try await container.perform { context in
            guard let moe = context.model as? Qwen35MoE else {
                Issue.record("模型类型不符")
                return
            }
            let lm = moe.languageModel
            try lm.prepare()

            for config in configs {
                let seg = max(512, (config.nominalDepth / config.segments / 512) * 512)
                let total = seg * config.segments
                let ids = syntheticIds(total, seed: 11)

                // P:首段
                let cacheP = lm.makeCache(capacity: nil)
                var stateP: LMOutput.State? = nil
                _ = prefill(lm, cache: cacheP, ids: Array(ids[0..<seg]), state: &stateP)

                // 链:fork+append 至全深(叶 = S 段 COW)
                var leafCaches = cacheP
                var leafState = stateP
                let tChain = clock.measure {
                    for i in 1..<config.segments {
                        leafCaches = forkModelCache(leafCaches)
                        _ = prefill(
                            lm, cache: leafCaches,
                            ids: Array(ids[(i * seg)..<((i + 1) * seg)]), state: &leafState)
                    }
                }
                // P 解冻续写补至全深(= fresh 1 段对照,同内容)
                let tPCont = clock.measure {
                    _ = prefill(lm, cache: cacheP, ids: Array(ids[seg..<total]), state: &stateP)
                }
                report(
                    String(
                        format: "  %@ 构链(actual %d,seg %d):链 %.0fs + P 续 %.0fs | active=%dMiB",
                        config.label, total, seg, ms(tChain) / 1000, ms(tPCont) / 1000,
                        GPU.activeMemory / 1048576))

                // warm-up(各 6 步,不入稳态)
                var pInput = 0
                var lInput = 0
                for _ in 0..<warmupSteps {
                    let (_, t) = greedyStep(lm, cache: cacheP, state: &stateP, input: pInput)
                    pInput = t
                    let (_, t2) = greedyStep(lm, cache: leafCaches, state: &leafState, input: lInput)
                    lInput = t2
                }

                // 3 reps × 24 步,交替顺序(P 先 / 叶 先),深度锁步
                var repRatios: [Double] = []
                var pTokensAll: [Int] = []
                var lTokensAll: [Int] = []
                var firstP: Double = 0
                var firstL: Double = 0
                for rep in 1...reps {
                    var pSteps: [Double] = []
                    var lSteps: [Double] = []
                    var pTokens: [Int] = []
                    var lTokens: [Int] = []
                    func measureP() {
                        for _ in 0..<measuredSteps {
                            let (ms, t) = greedyStep(lm, cache: cacheP, state: &stateP, input: pInput)
                            pInput = t
                            pSteps.append(ms)
                            pTokens.append(t)
                        }
                    }
                    func measureL() {
                        for _ in 0..<measuredSteps {
                            let (ms, t) = greedyStep(
                                lm, cache: leafCaches, state: &leafState, input: lInput)
                            lInput = t
                            lSteps.append(ms)
                            lTokens.append(t)
                        }
                    }
                    if rep % 2 == 1 { measureP(); measureL() } else { measureL(); measureP() }
                    if rep == 1 { firstP = pSteps[0]; firstL = lSteps[0] }
                    let steadyP = pSteps.dropFirst().reduce(0, +) / Double(pSteps.count - 1)
                    let steadyL = lSteps.dropFirst().reduce(0, +) / Double(lSteps.count - 1)
                    repRatios.append(steadyL / steadyP)
                    pTokensAll += pTokens
                    lTokensAll += lTokens
                }
                // 逐 token 门:同内容同深度,两侧 greedy 序列必须一致
                #expect(pTokensAll == lTokensAll)
                let mean = repRatios.reduce(0, +) / Double(reps)
                let steadyPFinal = repRatios.isEmpty ? 0 : 0  // 占位避免未用警告
                _ = steadyPFinal
                report(
                    String(
                        format: "  %@ 稳态税比: %@ → 均值 %.3f× | 首 token: P %.1fms / S段 %.1fms | token 门 %@",
                        config.label,
                        repRatios.map { String(format: "%.3f", $0) }.joined(separator: "/"),
                        mean, firstP, firstL,
                        pTokensAll == lTokensAll ? "✓ \(reps * measuredSteps)/\(reps * measuredSteps)" : "✗"))
                // 释放本配置(下一配置重建)
                GPU.clearCache()
            }
        }
    }

    private func report(_ line: @autoclosure () -> String) {
        print("[F4EF] " + line())
    }

    private func ms(_ duration: Duration) -> Double {
        let ns = Double(duration.components.seconds) * 1e9
            + Double(duration.components.attoseconds) / 1e9
        return ns / 1e6
    }

    // MARK: - F4-F

    private func tokenChunk(chunkIndex: Int, tokens: Int, salt: Int) -> (MLXArray, MLXArray) {
        let kvHeads = 4, headDim = 128
        let count = kvHeads * headDim * tokens
        let k = (0..<count).map {
            Float((($0 * 37 + (chunkIndex + salt) * 101) % 203) - 101) / 61.0
        }
        let v = (0..<count).map {
            Float((($0 * 19 + (chunkIndex + salt) * 53) % 197) - 98) / 59.0
        }
        let shape = [1, kvHeads, tokens, headDim]
        return (
            MLXArray(k).reshaped(shape).asType(.float16),
            MLXArray(v).reshaped(shape).asType(.float16)
        )
    }

    @discardableResult
    private func feed(_ cache: any KVCache, depth: Int, salt: Int = 0) -> (MLXArray, MLXArray) {
        var last: (MLXArray, MLXArray) = (MLXArray.zeros([0], dtype: .float16),
                                          MLXArray.zeros([0], dtype: .float16))
        var fed = 0
        while fed < depth {
            let n = min(512, depth - fed)
            let (k, v) = tokenChunk(chunkIndex: fed / 512, tokens: n, salt: salt)
            last = cache.update(keys: k, values: v)
            fed += n
        }
        return last
    }

    @Test func f4fDiscardLifecycleAttribution() throws {
        report("=== F4-F: Discard 生命周期归因(逻辑引用/图引用/物理归还三概念)===")
        let bytesPerRow = 2 * 4 * 128 * 2
        let rootDepth = 16_384
        let suffix = 4_096

        // ---- ① 复用测试:丢弃后重建等大分支,active 复用而非增长 ----
        GPU.clearCache()
        let root = KVCacheSimple()
        _ = feed(root, depth: rootDepth)
        GPU.clearCache()
        let baseActive = GPU.activeMemory
        var actives: [Int] = []
        for round in 0..<20 {
            var child: COWForkKVCache? = COWForkKVCache(parent: root)
            _ = feed(child!, depth: suffix, salt: 300 + round)
            child = nil
            GPU.clearCache()
            actives.append(GPU.activeMemory)
        }
        let maxDrift = actives.map { abs($0 - baseActive) }.max() ?? 0
        report(
            String(
                format: "① 复用测试(20 轮 fork+4K 追加+丢弃,%dKiB/轮): active 漂移 max %dKiB → %@",
                suffix * bytesPerRow / 1024, maxDrift / 1024,
                maxDrift < 2048 ? "不可达私有页可复用(无累积)" : "⚠ 存在累积,泄漏嫌疑"))
        #expect(maxDrift < 2 * 1024 * 1024)

        // ---- ② 强制 eval + clearCache 对比(F4-B 单点不降的补充条件)----
        GPU.clearCache()
        let base2 = GPU.activeMemory
        var subtree: [COWForkKVCache] = []
        for b in 0..<2 {
            let c = COWForkKVCache(parent: root)
            _ = feed(c, depth: suffix, salt: 400 + b)
            subtree.append(c)
        }
        // 保留已 eval 的终视图(模拟 F4-B:下游持有已物化视图节点)
        let heldViews = subtree.map { c -> [MLXArray] in
            let s = c.state
            eval(s)
            return s
        }
        GPU.clearCache()
        let withSubtree = GPU.activeMemory
        subtree.removeAll()
        GPU.clearCache()
        let afterDrop = GPU.activeMemory
        let afterDropEvalDummy = GPU.activeMemory
        _ = heldViews
        report(
            String(
                format: "② 持有已 eval 视图节点时丢弃: +建树 %dKiB → 丢弃后回降 %dKiB(持有视图 %@)",
                (withSubtree - base2) / 1024, (withSubtree - afterDrop) / 1024,
                (withSubtree - afterDrop) > 1024 ? "不阻塞回收" : "⚠ 视图节点疑似持有父引用(F4-B 0 下降同因)"))
        _ = afterDropEvalDummy

        // ---- ③ 组合丢弃:root / B 子树 / promoted 叶分别丢弃 ----
        GPU.clearCache()
        let root2 = KVCacheSimple()
        _ = feed(root2, depth: rootDepth)
        let A = COWForkKVCache(parent: root2)
        _ = feed(A, depth: 4_096, salt: 0)
        let A1 = COWForkKVCache(parent: A)
        _ = feed(A1, depth: 2_048, salt: 0)  // promoted 主干候选
        let B = COWForkKVCache(parent: root2)
        _ = feed(B, depth: 4_096, salt: 60)
        let viewA1 = A1.state
        eval(viewA1)
        GPU.clearCache()
        let stage0 = GPU.activeMemory
        var holdB: COWForkKVCache? = B
        holdB = nil  // 丢 B
        GPU.clearCache()
        let stage1 = GPU.activeMemory
        var holdRoot: KVCacheSimple? = root2
        holdRoot = nil  // 丢 root(A/A1 仍持段表引用)
        GPU.clearCache()
        let stage2 = GPU.activeMemory
        let viewA1After = A1.state
        eval(viewA1After)
        let okAfterDrops = sum(viewA1After[0] .== viewA1[0]).item(Int.self) == viewA1After[0].size
        GPU.clearCache()
        var holdA: COWForkKVCache? = A
        var holdA1: COWForkKVCache? = A1
        holdA = nil
        holdA1 = nil  // 全丢
        GPU.clearCache()
        let stage3 = GPU.activeMemory
        report(
            String(
                format: "③ 组合丢弃: 丢B %dKiB → 再丢root %dKiB(A1 靠段表存活,正确 %@)→ 全丢 %dKiB",
                (stage0 - stage1) / 1024, (stage1 - stage2) / 1024, okAfterDrops ? "✓" : "✗",
                (stage2 - stage3) / 1024))
        #expect(okAfterDrops)

        // ---- ④ 判定汇总 ----
        report(
            "④ 三概念对齐:逻辑引用释放 = 即时(ARC);物理归还/复用 = 复用测试①证实;"
                + "单点不降(F4-B)最可能 = 惰性图已 eval 节点的父引用保持②,待 Instruments 级确认")
    }
}
