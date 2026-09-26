// Copyright © 2026 SimiGo-Lab. Track E(命名待定)Residency P0.2 —
// Resident Pressure Curve(内存压力下的 decode 延迟曲线)
//
// 分支:exp/execution-state-fork(实验室分支;不回头修改已封口实验)
// 记录:SimiGo-Lab trackE-residency/(P0 容量探针 PASS/经济闭合 Pending;
// P0.1 字典序框架 v2)
//
// 研究问题:v0.1 字典序框架留白量——C_resident 是"纯字节占用"
// (容量约束)还是"字节 + 压力税"(驻留接近上限时 decode 劣化)?
// 若曲线平坦 → C_resident = 纯字节占用,经济模型在容量约束下闭合;
// 若上升 → 测得压力税项(并披露热节流混淆方向:后测偏热,上升结论弱)。
//
// 方法:P@4096 基座 + probe 链(@4616 固定);驻留子代 k = 0→56 递增
// (每节点 520 行 ≈ 76.4 MiB);检查点(k=0,8,16,...)各测 16 步
// decode burst;记录 (k, activeMem, ms/tok)。天花板逼近(+4/步)的
// MLX 分配失败本身即模拟器硬约束的实证。

import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

@Suite(.serialized)
struct ResidencyP02PressureCurve {

    private func report(_ line: @autoclosure () -> String) {
        print("[ResidencyP02] " + line())
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
            throw NSError(
                domain: "ResidencyP02", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "model snapshot not found"])
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

    private func step(
        _ lm: Qwen35Language.LanguageModel,
        cache: [KVCache],
        state: inout LMOutput.State?,
        input: Int
    ) -> Int {
        let out = lm(
            MLXArray([Int32(input)]),
            cache: cache.map { $0 as KVCache? },
            state: state)
        state = out.state
        eval(out.logits)
        return out.logits[0, out.logits.dim(1) - 1].argMax().item(Int.self)
    }

    @Test func p02ResidentPressureCurve() async throws {
        report("=== P0.2:驻留压力曲线(decode 延迟 vs 驻留节点数)===")
        let dir: URL
        do { dir = try modelDirectory() } catch {
            Issue.record("模型不在本机,跳过: \(error)"); return
        }
        GPU.set(memoryLimit: 24 * 1024 * 1024 * 1024)
        let container = try await VLMModelFactory.shared.loadContainer(
            from: dir, using: NoOpTokenizerLoader())

        try await container.perform { context in
            guard let moe = context.model as? Qwen35MoE else {
                Issue.record("模型类型不符"); return
            }
            let lm = moe.languageModel
            try lm.prepare()
            let clock = ContinuousClock()

            // ---- 基座 P@4096 + probe 链(@4616;burst 均在 probe 上)
            let base = self.syntheticIds(4_096, seed: 181)
            let suffix = self.syntheticIds(512, seed: 182)
            var cacheP = lm.makeCache(capacity: nil)
            var stateP: LMOutput.State? = nil
            _ = self.prefill(lm, cache: cacheP, ids: base, state: &stateP)

            var probe = forkModelCache(cacheP)
            var stateProbe = stateP
            let logits = self.prefill(lm, cache: probe, ids: suffix, state: &stateProbe)
            var input = logits[0, logits.dim(1) - 1].argMax().item(Int.self)
            for _ in 0..<8 {
                input = self.step(lm, cache: probe, state: &stateProbe, input: input)
            }

            // ---- 压力曲线:驻留子代 k 递增,检查点测 16 步 burst
            let burstSteps = 16
            var children: [[KVCache]] = []
            var built = 0
            var curve: [(Int, Double, Int)] = []  // (k, ms/tok, active MiB)

            func measureCheckpoint(_ k: Int) {
                GPU.clearCache()
                let mem = GPU.activeMemory / 1048576
                let t = clock.measure {
                    for _ in 0..<burstSteps {
                        input = self.step(lm, cache: probe, state: &stateProbe, input: input)
                    }
                }
                let perTok = self.ms(t) / Double(burstSteps)
                curve.append((k, perTok, mem))
                report(String(format: "k=%2d | active %5d MiB | burst %6.2f ms/tok",
                    k, mem, perTok))
            }

            measureCheckpoint(0)

            let checkpoints: [Int] = [8, 16, 24, 32, 40, 48, 56]
            var ci = 0
            while built < 56 {
                for _ in 0..<8 {
                    var c = forkModelCache(cacheP)
                    var st = stateP
                    let lg = self.prefill(lm, cache: c, ids: suffix, state: &st)
                    var nx = lg[0, lg.dim(1) - 1].argMax().item(Int.self)
                    for _ in 0..<8 {
                        nx = self.step(lm, cache: c, state: &st, input: nx)
                    }
                    children.append(c)
                    built += 1
                }
                GPU.clearCache()
                if ci < checkpoints.count && built >= checkpoints[ci] {
                    measureCheckpoint(built)
                    ci += 1
                }
            }

            // ---- 曲线判读(热节流方向披露:上升结论弱于平坦结论)
            let first = curve.first!.1
            let last = curve.last!.1
            let drift = (last - first) / first * 100
            let flat = abs(drift) < 10.0
            report(String(format: "曲线判读:首末漂移 %+.1f%%(%@)", drift,
                flat ? "平坦——C_resident = 纯字节占用(本配置)" :
                       "超 10%——压力税/热漂移混淆,需交替对照分离,本轮不下结论"))
            for (k, perTok, mem) in curve {
                report(String(format: "  k=%2d → %6.2f ms/tok(active %d MiB)", k, perTok, mem))
            }

            // ---- 天花板逼近(+4/步;MLX 分配失败=硬约束实证,log 止于失败前)
            report("天花板逼近试验:+4 节点/步,直至 MLX 分配失败")
            while built < 64 {
                report("  尝试建至 k=\(built + 4)(active \(GPU.activeMemory / 1048576) MiB)")
                for _ in 0..<4 {
                    var c = forkModelCache(cacheP)
                    var st = stateP
                    let lg = self.prefill(lm, cache: c, ids: suffix, state: &st)
                    var nx = lg[0, lg.dim(1) - 1].argMax().item(Int.self)
                    for _ in 0..<8 {
                        nx = self.step(lm, cache: c, state: &st, input: nx)
                    }
                    children.append(c)
                    built += 1
                }
                GPU.clearCache()
            }
            report("*** P0.2 完成:压力曲线如上;天花板行为如实入档。***")
        }
    }
}
