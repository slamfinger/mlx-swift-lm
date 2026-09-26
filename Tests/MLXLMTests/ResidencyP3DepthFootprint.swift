// Copyright © 2026 SimiGo-Lab. Track E(命名待定)Residency P3 前置探针 —
// Resident Footprint Depth Sweep(驻留分配的深度扫描)
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径;
// 承接 P0.1 PASS(v0.1 Policy Objective Definition);不回头修改已封口实验)
// 记录:SimiGo-Lab trackE-residency/(P0 审计第④条:attention"随深度
// 增长"= 结构推断,需独立 resident-footprint depth sweep;M3-B 外化
// 载荷平坦性不替代本扫描)
//
// 内容:对 P@depth(depth ∈ {4096, 16384, 65536})的完整驻留链做
// nbytes 精确计量(P0 方法论沿用):
//   attention 分配字节(keys+values,含 step-256 padding)/ 逻辑字节 /
//   GDN 槽字节(预期深度无关)/ 占比与交叉点。
// 目的:把"attention 随深度增长、GDN 固定底价"从结构推断升级为
// 本配置下的实测事实,喂给 P3 的 state-size 信号设计。

import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

@Suite(.serialized)
struct ResidencyP3DepthFootprint {

    private func report(_ line: @autoclosure () -> String) {
        print("[ResidencyP3sweep] " + line())
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
                domain: "ResidencyP3sweep", code: 1,
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

    @Test func p3prerequisiteDepthFootprintSweep() async throws {
        report("=== P3 前置:驻留 footprint 深度扫描(4K / 16K / 64K;nbytes 精计)===")
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

            report("depth | attn_alloc | attn_logical | padding | GDN | GDN占比 | attn占比")
            var gdnConst: Int? = nil
            for (di, depth) in [4_096, 16_384, 65_536].enumerated() {
                var cacheP = lm.makeCache(capacity: nil)
                var stateP: LMOutput.State? = nil
                _ = self.prefill(
                    lm, cache: cacheP,
                    ids: self.syntheticIds(depth, seed: 171 + di), state: &stateP)

                var attnAlloc = 0
                var attnLogical = 0
                for cache in cacheP {
                    guard let simple = cache as? KVCacheSimple,
                          let k = simple.keys, let v = simple.values else { continue }
                    attnAlloc += k.nbytes + v.nbytes
                    attnLogical += simple.offset * 4 * 128 * 2 * 2  // 逻辑行 × (K+V) × headDim×kvHeads × fp16
                }
                var gdn = 0
                for cache in cacheP {
                    guard let m = cache as? MambaCache else { continue }
                    for s in 0..<m.slotCount {
                        if let a = m[s] { eval(a); gdn += a.nbytes }
                    }
                }
                if let g = gdnConst {
                    #expect(g == gdn)  // GDN 深度无关(结构性确认)
                } else {
                    gdnConst = gdn
                }
                cacheP = []
                stateP = nil
                GPU.clearCache()

                let total = attnAlloc + gdn
                let padding = attnAlloc - attnLogical
                report(String(format: "%6d | %8d KiB | %8d KiB | %5.1f%% | %6d KiB | %4.1f%% | %4.1f%%",
                    depth, attnAlloc / 1024, attnLogical / 1024,
                    Double(padding) * 100 / Double(max(attnAlloc, 1)),
                    gdn / 1024,
                    Double(gdn) * 100 / Double(max(total, 1)),
                    Double(attnAlloc) * 100 / Double(max(total, 1))))
            }
            report(String(format: "GDN 槽总量(三深度一致,深度无关确认): %d KiB", gdnConst! / 1024))
            report("*** P3 前置:attention 随深度增长 + GDN 固定底价——结构推断升级为本配置实测(占比如上,交叉点 = 两分量相等处)***")
        }
    }
}
