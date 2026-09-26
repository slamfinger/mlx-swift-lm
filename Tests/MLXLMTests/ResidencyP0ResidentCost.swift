// Copyright © 2026 SimiGo-Lab. Track E(命名待定)Residency P0 —
// Cost Model Closure:C_resident 计价(驻留占用的字节经济学)
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径;
// 承接 Simulator v0 FINAL PASS;不回头修改 M 链 / M3 系 / v0)
// 记录:SimiGo-Lab trackE-residency/(v0 终判块:P0 先于 P3)
//
// 研究问题(v0 终判定序):policy 不等式
//   Keep Resident(C_resident) vs Externalize + Future Rehydrate(≈0.72 token)
// 目前只有右边有数。P0 = 把左边补上:
//   ① 每节点驻留字节(**nbytes 精确计价**:attn 私有页分配 + GDN 槽 +
//      State;GPU activeMemory 仅作粗交叉验证——统一内存计账含瞬态,
//      v1 教训:读数被 prefill 瞬态污染出现负增量,不作为定价基础);
//   ② 组件拆分与 padding 披露(attn step-256 分配行 vs 逻辑使用行);
//   ③ 机器真实预算:memoryLimit − 权重 = 可用驻留字节 → 折合最大
//      节点数(simulator budget 参数自此有真实映射);
//   ④ 每步计算成本 = 0(M4 已证:无超噪声稳态税)——驻留的剩余成本
//      是纯字节占用(机会成本)。
//
// 措辞纪律:本探针只计价,不做策略判断;C_resident 的"token 换算率"
// 依赖机器与场景,如实留白给 Residency Policy。

import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

@Suite(.serialized)
struct ResidencyP0ResidentCost {

    private func report(_ line: @autoclosure () -> String) {
        print("[ResidencyP0] " + line())
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
                domain: "ResidencyP0", code: 1,
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

    @Test func p0ResidentCostPricing() async throws {
        report("=== P0:C_resident 计价——每节点驻留字节(nbytes 精确)+ 组件拆分 + 真实预算上限 ===")
        let dir: URL
        do { dir = try modelDirectory() } catch {
            Issue.record("模型不在本机,跳过: \(error)"); return
        }
        let memoryLimit = 24 * 1024 * 1024 * 1024
        GPU.set(memoryLimit: memoryLimit)
        let container = try await VLMModelFactory.shared.loadContainer(
            from: dir, using: NoOpTokenizerLoader())

        try await container.perform { context in
            guard let moe = context.model as? Qwen35MoE else {
                Issue.record("模型类型不符"); return
            }
            let lm = moe.languageModel
            try lm.prepare()

            // ---- 1. 权重基线(粗交叉验证用;定价以 nbytes 为准)
            GPU.clearCache()
            let weightsMem = GPU.activeMemory
            report("权重驻留(activeMemory 粗读):\(weightsMem / 1048576) MiB")

            // ---- 2. 父代 P@4096 + 单子代构建(用于 nbytes 精确计价)
            let base = self.syntheticIds(4_096, seed: 161)
            var cacheP = lm.makeCache(capacity: nil)
            var stateP: LMOutput.State? = nil
            _ = self.prefill(lm, cache: cacheP, ids: base, state: &stateP)

            let suffix = self.syntheticIds(512, seed: 162)
            var c0 = forkModelCache(cacheP)
            var st0 = stateP
            let logits = self.prefill(lm, cache: c0, ids: suffix, state: &st0)
            var next = logits[0, logits.dim(1) - 1].argMax().item(Int.self)
            for _ in 0..<8 {
                next = self.step(lm, cache: c0, state: &st0, input: next)
            }

            let attnIdx: [Int] = c0.enumerated().compactMap {
                ($1 is COWForkKVCache || $1 is KVCacheSimple) ? $0 : nil
            }
            let gdnIdx: [Int] = c0.enumerated().compactMap { $1 is MambaCache ? $0 : nil }

            // ---- 3. 每节点驻留字节:nbytes 精确计价
            var attnAllocBytes = 0
            var attnAllocRows = 0
            var attnUsedRows = 0
            for i in attnIdx {
                let cow = c0[i] as! COWForkKVCache
                attnAllocBytes += (cow.privateKeys?.nbytes ?? 0) + (cow.privateValues?.nbytes ?? 0)
                attnAllocRows += cow.privateKeys?.dim(2) ?? 0
                attnUsedRows += cow.offset - cow.forkDepth
            }
            var gdnBytes = 0
            var gdnSlotCount = 0
            var gdnShapes: [String: (count: Int, bytes: Int)] = [:]
            for i in gdnIdx {
                let m = c0[i] as! MambaCache
                for s in 0..<m.slotCount {
                    guard let a = m[s] else { continue }
                    eval(a)
                    gdnBytes += a.nbytes
                    gdnSlotCount += 1
                    let key = "\(a.shape)@\(a.dtype)"
                    let prev = gdnShapes[key] ?? (0, 0)
                    gdnShapes[key] = (prev.count + 1, prev.bytes + a.nbytes)
                }
            }
            var stateBytes = 0
            var stateCount = 0
            if let st = st0, let dict = try? st.serializedArrays() {
                for (_, v) in dict {
                    eval(v)
                    stateBytes += v.nbytes
                    stateCount += 1
                }
            }
            let residentBytesPerNode = attnAllocBytes + gdnBytes + stateBytes

            report(String(format: "每节点驻留字节(nbytes 精计,520 逻辑行): attention 私有页(分配) %d KiB | GDN 槽 %d KiB | State %d KiB → 合计 %d KiB/节点",
                attnAllocBytes / 1024, gdnBytes / 1024, stateBytes / 1024, residentBytesPerNode / 1024))
            report(String(format: "attention padding 披露:分配 %d 行(step-256 对齐)/ 使用 %d 逻辑行——外部化载荷按逻辑行,驻留按分配行",
                attnAllocRows, attnUsedRows))
            for (shape, info) in gdnShapes.sorted(by: { $0.key < $1.key }) {
                report(String(format: "  GDN 槽形态 %@ × %d(计 %d KiB)", shape, info.count, info.bytes / 1024))
            }
            let gdnShare = gdnBytes * 100 / max(residentBytesPerNode, 1)
            let attnShare = attnAllocBytes * 100 / max(residentBytesPerNode, 1)
            report(String(format: "驻留成本主体 = GDN %d%% / attention %d%%——每节点 O(1) 的 GDN 态是不可压缩的固定驻留底价", gdnShare, attnShare))

            // ---- 4. 交叉验证:GPU activeMemory 粗读(尾部增量)
            var children: [[KVCache]] = [c0]
            var childStates: [LMOutput.State?] = [st0]
            var built = 1
            GPU.clearCache()
            var readings: [(Int, Int)] = [(1, GPU.activeMemory - weightsMem)]
            while built < 8 {
                var c = forkModelCache(cacheP)
                var st = stateP
                let lg = self.prefill(lm, cache: c, ids: suffix, state: &st)
                var nx = lg[0, lg.dim(1) - 1].argMax().item(Int.self)
                for _ in 0..<8 {
                    nx = self.step(lm, cache: c, state: &st, input: nx)
                }
                children.append(c)
                childStates.append(st)
                built += 1
                GPU.clearCache()
                readings.append((built, GPU.activeMemory - weightsMem))
            }
            let tail = readings.suffix(4)
            let slope = (tail.last!.1 - tail.first!.1) / (tail.last!.0 - tail.first!.0)
            report("activeMemory 交叉验证(粗读,含瞬态噪声):")
            for (k, delta) in readings {
                report(String(format: "  k=%d:累计 %d MiB", k, delta / 1048576))
            }
            report(String(format: "  尾部拟合斜率 ≈ %d KiB/节点 vs nbytes 精计 %d KiB/节点(量级一致即通过)",
                slope / 1024, residentBytesPerNode / 1024))

            // ---- 5. 真实预算上限(nbytes 计价)
            let headroom = memoryLimit - weightsMem
            let maxNodes = Int(Double(headroom) / Double(residentBytesPerNode))
            report(String(format: "预算上限:memoryLimit − 权重 = %d MiB → 折合 ≈ %d 个 520 行节点(等大小模拟的 budget 参数自此有真实映射)",
                headroom / 1048576, maxNodes))

            // ---- 6. 补齐后的 policy 不等式(数字闭合,不做策略判断)
            report("policy 不等式(数字闭合):")
            report(String(format: "  Keep(n, T) = %d KiB × T 字节占用 + 0 ms/token 计算成本(M4:无超噪声稳态税)", residentBytesPerNode / 1024))
            report("  Leave(n) = C_ext 0.69 + C_rehy 0.03 ≈ 0.72 token-equivalent(一次性,M4)+ 外部存储按逻辑行 73,292 KiB")
            report(String(format: "  → 驻留计算成本为零;C_resident 本体 = 字节机会成本(主体 = GDN %d%% 固定底价);策略分歧只出现在字节预算绑定时", gdnShare))
            #expect(residentBytesPerNode > 0)
            #expect(gdnBytes > attnAllocBytes)  // GDN 是主体这一发现的结构性确认
            #expect(headroom > 0)
        }
    }
}
