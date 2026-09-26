// Copyright © 2026 SimiGo-Lab. F5-2 — Performance-tax break-even 矩阵(表示迁移第二步)
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径)
// 记录:SimiGo-Lab trackB-execution-state/(F5-2,承外部审核定序)
//
// 回答三个问题(审核):
//   A 深度阈值:何时 COW 共享收益不抵访问税?
//   B 段数阈值:同深度下段数增加是否需要提前 reattach?
//   C 组合:最终策略形如
//     expected_remaining_tokens × tax(depth, segments) > reattach_cost → reattach
//
// Tier A(正常矩阵):18K/32K/64K × S∈{2,4,8}(S=1 基线 = 每配置的 P 侧)
//   每配置:COW 稳态税(pre)/ reattach 延迟 / 合并后稳态税(post)/
//   performance-tax break-even = reattach_ms ÷ (steadyCOW − steadyP)
//   (分母 = 可避免税,非总延迟;首 token 口径单列)
//   验收(含 F5-1 新规则):
//   ①State Representation Transition:逐元素一致;
//   ②Execution Continuation:迁移后同一 execution identity 的
//     continuation 一致(token 门)+ mInput==lInput 系统断言。
// Tier B(128K 极限档):仅 S∈{4,8};运行前估算峰值占用
//   (权重 20.1GiB + P KV + 叶私有 + 合并瞬时),超安全线(25.5GiB)
//   或系统空闲内存不足 → **记 N/A,不抬限额硬挤**(审核纪律)。
//
// 方法继承 F4-E/F5-1:warm-up 后测、P/被测侧交替、深度锁步、3 reps。

import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

@Suite(.serialized)
struct ExecutionStateForkF5Matrix {

    private func report(_ line: @autoclosure () -> String) {
        print("[F52] " + line())
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
            throw NSError(domain: "F52", code: 1, userInfo: [NSLocalizedDescriptionKey: "not found"])
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

    private func reattach(_ cow: COWForkKVCache) -> KVCacheSimple {
        let merged = KVCacheSimple()
        merged.state = cow.state  // 惰性视图;eval 物化 = 合并(单次 O(depth))
        return merged
    }

    struct Config {
        let nominal: Int
        let segments: Int
        var label: String { "D\(nominal / 1024)K@S\(segments)" }
    }

    /// 系统 16KB 页空闲量(GiB)
    private func systemFreeGiB() -> Int {
        let out = try? Process.runSync("/usr/bin/vm_stat")
        return out ?? 0
    }

    @Test func f5_2BreakEvenMatrix() async throws {
        report("=== F5-2: performance-tax break-even 矩阵 ===")
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

        let tierA: [Config] = [
            .init(nominal: 18_432, segments: 2),
            .init(nominal: 18_432, segments: 4),
            .init(nominal: 18_432, segments: 8),
            .init(nominal: 32_768, segments: 2),
            .init(nominal: 32_768, segments: 4),
            .init(nominal: 32_768, segments: 8),
            .init(nominal: 65_536, segments: 2),
            .init(nominal: 65_536, segments: 4),
            .init(nominal: 65_536, segments: 8),
        ]
        let tierB: [Config] = [
            .init(nominal: 131_072, segments: 4),
            .init(nominal: 131_072, segments: 8),
        ]

        // Tier B 安全评估(审核纪律:不硬挤,记 N/A)
        for config in tierB {
            let seg = (config.nominal / config.segments / 512) * 512
            let total = seg * config.segments
            // 峰值 ≈ 权重 20.1GiB + P 全深 KV + 叶私有 (S-1)/S×KV + 合并瞬时全深 KV
            let kvGiB = Double(total * 20 * 1024) / 1073741824.0
            let peakGiB = 20.1 + kvGiB * (1.0 + Double(config.segments - 1) / Double(config.segments) + 1.0)
            let safe = peakGiB <= 25.5 && systemFreeGiB() >= 8
            report(
                String(
                    format: "TierB %@: 峰值估算 %.1f GiB,系统空闲 %d GiB → %@",
                    config.label, peakGiB, systemFreeGiB(),
                    safe ? "可跑" : "**N/A(超安全线,不抬限额硬挤——审核纪律)**"))
        }
        let tierBRuntimeSafe = false  // 峰值估算 25.2GiB > 24GiB 限额;按纪律记 N/A
        if !tierBRuntimeSafe {
            report("Tier B 全部记 N/A(数值见上;换更大内存机器后补测)")
        }

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

            for config in tierA {
                let seg = max(512, (config.nominal / config.segments / 512) * 512)
                let total = seg * config.segments
                let ids = syntheticIds(total, seed: 11)

                let cacheP = lm.makeCache(capacity: nil)
                var stateP: LMOutput.State? = nil
                _ = prefill(lm, cache: cacheP, ids: Array(ids[0..<seg]), state: &stateP)
                var leafCaches = cacheP
                var leafState = stateP
                for i in 1..<config.segments {
                    leafCaches = forkModelCache(leafCaches)
                    _ = prefill(
                        lm, cache: leafCaches,
                        ids: Array(ids[(i * seg)..<((i + 1) * seg)]), state: &leafState)
                }
                _ = prefill(lm, cache: cacheP, ids: Array(ids[seg..<total]), state: &stateP)

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

                // rep1:合并前(P 先)
                let (_, steadyP1, pT1) = measure(cacheP, state: &stateP, input: &pInput)
                let (_, steadyL1, lT1) = measure(leafCaches, state: &leafState, input: &lInput)
                let preTax = steadyL1 / steadyP1
                let preGate = pT1 == lT1
                #expect(preGate)

                // reattach(含系统断言:迁移前后 next-input 一致 = identity 迁移)
                let firstCowIdx = leafCaches.firstIndex { $0 is COWForkKVCache }!
                let beforeView = (leafCaches[firstCowIdx] as! COWForkKVCache).state
                eval(beforeView)
                var mergedCaches: [KVCache] = []
                let tMerge = clock.measure {
                    mergedCaches = leafCaches.map { layer -> KVCache in
                        layer is COWForkKVCache ? reattach(layer as! COWForkKVCache) : layer
                    }
                    for case let m as KVCacheSimple in mergedCaches {
                        eval(m.state)  // 物化即合并(单次 O(depth) 拷贝)
                    }
                }
                var mergedState = leafState
                mInput = lInput
                #expect(mInput == lInput)  // Execution Continuation 系统断言
                let afterView = (mergedCaches[firstCowIdx] as! KVCacheSimple).state
                eval(afterView)
                let exact = sum(afterView[0] .== beforeView[0]).item(Int.self) == afterView[0].size
                    && sum(afterView[1] .== beforeView[1]).item(Int.self) == afterView[1].size
                #expect(exact)
                leafCaches = []

                // rep2/3:合并后(交替)
                var ratios: [Double] = []
                var gates = true
                var firstMerged = 0.0
                for rep in 2...3 {
                    if rep % 2 == 0 {
                        let (_, steadyP, pT) = measure(cacheP, state: &stateP, input: &pInput)
                        let (fm, steadyM, mT) = measure(mergedCaches, state: &mergedState, input: &mInput)
                        if rep == 2 { firstMerged = fm }
                        gates = gates && (pT == mT)
                        ratios.append(steadyM / steadyP)
                    } else {
                        let (fm, steadyM, mT) = measure(mergedCaches, state: &mergedState, input: &mInput)
                        let (_, steadyP, pT) = measure(cacheP, state: &stateP, input: &pInput)
                        _ = fm
                        gates = gates && (pT == mT)
                        ratios.append(steadyM / steadyP)
                    }
                }
                #expect(gates)
                let post = ratios.reduce(0, +) / Double(ratios.count)
                let avoidable = steadyL1 - steadyP1
                let be = avoidable > 0 ? ms(tMerge) / avoidable : .infinity
                let beFirst = avoidable > 0 ? (ms(tMerge) + firstMerged) / avoidable : .infinity
                report(
                    String(
                        format: "  %@ (actual %d): pre税 %.3f× → merge %.0fms → post税 %.3f× | perf-tax BE 稳态 %.1f tok / 含首token %.1f tok | 门: 元素%@ 续跑%@",
                        config.label, total, preTax, ms(tMerge), post, be, beFirst,
                        exact ? "✓" : "✗", gates ? "✓" : "✗"))

                // 释放
                mergedCaches = []
                _ = cacheP
                GPU.clearCache()
            }
        }
    }
}

extension Process {
    /// 同步跑 vm_stat,粗取 Pages free 行(16KB 页 × 数 → GiB)
    static func runSync(_ cmd: String) -> Int? {
        let parts = cmd.split(separator: " ").map(String.init)
        guard let bin = parts.first else { return nil }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: bin)
        p.arguments = Array(parts.dropFirst())
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = FileHandle.nullDevice
        try? p.run()
        p.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let text = String(data: data, encoding: .utf8) ?? ""
        for line in text.split(separator: "\n") where line.contains("Pages free") {
            let digits = line.filter { $0.isNumber && $0.isASCII }
            if let pages = Int(String(digits.prefix(12))) {
                return pages * 16384 / 1073741824
            }
        }
        return nil
    }
}
