// Copyright © 2026 SimiGo-Lab. Track E(命名待定)Residency M3 — Continuation
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径;
// 承接 M2 PASS 终判,不回头修改 Track B / M1 / M2)
// 记录:SimiGo-Lab trackE-residency/(M2 终判块:M2 PASS → OPEN M3)
//
// 问题升级(M2 终判):M1/M2 证明"回来以后**看起来**还是原来的
// Execution State";M3 问"回来以后**是不是还能成为原来的 Execution**"
// ——从 snapshot comparison 跨到 behavioral continuation。
//
// 设计(第一轮,单变量纪律:不加多节点/64K/大规模树/性能/policy):
//
//   P @4096(驻留父代)
//     ├── treat = fork → 后缀512 → decode8   ── 外化(私有页)→
//     │                                        释放 → 重挂回驻留父代
//     │                                        (仍为 COW 段表,未物化)
//     └── ctrl  = fork → 后缀512 → decode8   ── 驻留全程(control)
//
//   可证伪前置门:treat 与 ctrl 在外化前逐元素相等 + next-input 相等
//   (同父代同操作 ⇒ 必须同态;不等 = 实验环境有问题,立即停)。
//
//   主门:Continuation(ctrl) == Continuation(treat),N=96 逐 token。
//   辅门:父代完好;双侧 offset 精确推进 +N;重建子代仍为段表形态。
//
// 措辞纪律:identity 判定 = 状态接地;计时仅描述性(性能归 M4 交替
// 对照)。失败 ⇒ "该机制在该配置下不成立",不反推 Track B / M1 / M2。

import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

@Suite(.serialized)
struct ResidencyM3Continuation {

    private func report(_ line: @autoclosure () -> String) {
        print("[ResidencyM3] " + line())
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
                domain: "ResidencyM3", code: 1,
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

    @Test func m3SegmentedChildContinuation() async throws {
        report("=== M3:外化/回来的段表子代,能否沿原来的未来继续生长(N=96 逐 token 对照)===")
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

            let N = 96
            let base = self.syntheticIds(4_096, seed: 101)
            let suffix = self.syntheticIds(512, seed: 102)
            let decodeN = 8
            let clock = ContinuousClock()

            // ---- 1. 驻留父代 + 同形双子代(treat / ctrl)
            var cacheP = lm.makeCache(capacity: nil)
            var stateP: LMOutput.State? = nil
            _ = self.prefill(lm, cache: cacheP, ids: base, state: &stateP)

            func buildChild() -> ([KVCache], LMOutput.State?, Int) {
                var c = forkModelCache(cacheP)
                var st = stateP
                let logits = self.prefill(lm, cache: c, ids: suffix, state: &st)
                var next = logits[0, logits.dim(1) - 1].argMax().item(Int.self)
                for _ in 0..<decodeN {
                    next = self.step(lm, cache: c, state: &st, input: next)
                }
                return (c, st, next)
            }
            var (treat, stateTreat, nextTreat) = buildChild()
            var (ctrl, stateCtrl, nextCtrl) = buildChild()

            let attnIdx: [Int] = treat.enumerated().compactMap {
                ($1 is COWForkKVCache || $1 is KVCacheSimple) ? $0 : nil
            }
            let gdnIdx: [Int] = treat.enumerated().compactMap {
                $1 is MambaCache ? $0 : nil
            }
            let layerCount = treat.count
            let position = treat[attnIdx.first!].offset
            let offsetsBefore = treat.map { $0.offset }
            let gdnSlotCounts = Dictionary(uniqueKeysWithValues: gdnIdx.map {
                ($0, (treat[$0] as! MambaCache).slotCount)
            })
            let used = position - 4096
            report("双子代就绪 @\(position)(forkDepth=4096,后缀512+decode\(decodeN)),私有 used=\(used) 行,nextInput=\(nextTreat)")

            // ---- 2. 可证伪前置门:外化前 treat ≡ ctrl(逐元素)+ next-input 相等
            var gatePre = (nextTreat == nextCtrl)
            for i in attnIdx {
                let vt = treat[i].state
                let vc = ctrl[i].state
                eval(vt, vc)
                gatePre = gatePre && vt[0].shape == vc[0].shape
                    && sum(vt[0] .== vc[0]).item(Int.self) == vc[0].size
                    && vt[1].shape == vc[1].shape
                    && sum(vt[1] .== vc[1]).item(Int.self) == vc[1].size
            }
            for i in gdnIdx {
                let mt = treat[i] as! MambaCache
                let mc = ctrl[i] as! MambaCache
                for s in 0..<mt.slotCount {
                    let at = mt[s]!
                    let ac = mc[s]!
                    eval(at, ac)
                    gatePre = gatePre && at.shape == ac.shape
                        && sum(at .== ac).item(Int.self) == ac.size
                }
            }
            report("前置门(外化前 treat ≡ ctrl,逐元素 + next-input): \(gatePre ? "✓" : "✗")")
            #expect(gatePre)

            // 父代完整性参照
            var parentBefore: [Int: [MLXArray]] = [:]
            for (i, cache) in cacheP.enumerated() where cache is KVCacheSimple {
                let v = cache.state
                eval(v)
                parentBefore[i] = v
            }

            // ---- 3. treat 外化(私有页 + GDN 槽 + state + 单元元数据)并释放
            var payload: [String: MLXArray] = [:]
            for i in attnIdx {
                let cow = treat[i] as! COWForkKVCache
                payload["p\(i).k"] = cow.privateKeys![.ellipsis, ..<used, 0...]
                payload["p\(i).v"] = cow.privateValues![.ellipsis, ..<used, 0...]
                eval(payload["p\(i).k"]!, payload["p\(i).v"]!)
            }
            for i in gdnIdx {
                let m = treat[i] as! MambaCache
                for s in 0..<m.slotCount {
                    let a = m[s]!
                    eval(a)
                    payload["g\(i).s\(s)"] = a
                }
            }
            var stateKeyCount = 0
            if let st = stateTreat, let dict = try? st.serializedArrays() {
                for (k, v) in dict {
                    eval(v)
                    payload["st.\(k)"] = v
                }
                stateKeyCount = dict.count
            }
            payload["meta.nextInput"] = MLXArray([Int32(nextTreat)])
            payload["meta.used"] = MLXArray([Int32(used)])

            let file = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + "-m3.safetensors")
            defer { try? FileManager.default.removeItem(at: file) }
            try save(arrays: payload, url: file)
            let bytes = (try? Data(contentsOf: file))?.count ?? 0

            treat = []
            stateTreat = nil
            GPU.clearCache()
            report("treat 外化(私有页 \(used) 行,载荷 \(bytes / 1024) KiB,state \(stateKeyCount) 数组)并释放;ctrl 驻留全程")

            // ---- 4. 重挂回驻留父代(仍为 COW 段表,不物化)
            let loaded = try loadArrays(url: file)
            let usedAfter = Int(loaded["meta.used"]!.item(Int32.self))
            var rebuilt = [KVCache?](repeating: nil, count: layerCount)
            for i in attnIdx {
                let cow = COWForkKVCache(parent: cacheP[i])
                cow.privateKeys = loaded["p\(i).k"]!
                cow.privateValues = loaded["p\(i).v"]!
                cow.offset = cacheP[i].offset + usedAfter
                rebuilt[i] = cow
            }
            for i in gdnIdx {
                let m = MambaCache()
                for s in 0..<(gdnSlotCounts[i] ?? 0) {
                    m[s] = loaded["g\(i).s\(s)"]!
                }
                m.offset = offsetsBefore[i]
                rebuilt[i] = m
            }
            var stateTA: LMOutput.State? = nil
            let stKeys = loaded.keys.filter { $0.hasPrefix("st.") }.sorted()
            if !stKeys.isEmpty {
                var dict: [String: MLXArray] = [:]
                for k in stKeys {
                    dict[String(k.dropFirst(3))] = loaded[k]!
                }
                stateTA = LMOutput.State(serializedArrays: dict)
            }
            let treatAfter = rebuilt.map { $0! }
            var gateSegmentForm = true
            for i in attnIdx {
                guard let cow = treatAfter[i] as? COWForkKVCache else {
                    gateSegmentForm = false; continue
                }
                gateSegmentForm = gateSegmentForm
                    && (cow.sharedRows == cacheP[i].offset)
                    && (cow.offset == position)
            }
            for i in gdnIdx {
                gateSegmentForm = gateSegmentForm
                    && (treatAfter[i].offset == ctrl[i].offset)
            }
            report("重建核查:treat 仍为 COW 段表(sharedRows=父代 offset,offset @\(treatAfter[attnIdx.first!].offset)),state 重建 \(stKeys.count) 数组: \(gateSegmentForm ? "✓" : "✗")")

            // ---- 5. 主门:双侧各自继续 N 步,逐 token 对照
            var inputT = Int(loaded["meta.nextInput"]!.item(Int32.self))
            var inputC = nextCtrl
            var tokensT: [Int] = []
            var tokensC: [Int] = []
            let tRun = clock.measure {
                for _ in 0..<N {
                    inputT = self.step(lm, cache: treatAfter, state: &stateTA, input: inputT)
                    inputC = self.step(lm, cache: ctrl, state: &stateCtrl, input: inputC)
                    tokensT.append(inputT)
                    tokensC.append(inputC)
                }
            }
            let gateContinuation = (tokensT == tokensC)
            let divergeAt = tokensT.firstDifferenceIndex(vs: tokensC)
            report(String(format: "主门:Continuation(treat) == Continuation(ctrl),N=%d 逐 token: %@(%.1f s,描述性)",
                N, gateContinuation ? "✓ \(N)/\(N) 全等" : "✗ 首个分歧 @\(divergeAt ?? -1)", ms(tRun) / 1000.0))
            #expect(gateContinuation)

            // ---- 6. 辅门:offset 精确推进 + 父代完好
            var gateOffsets = true
            for i in attnIdx {
                gateOffsets = gateOffsets
                    && (treatAfter[i].offset == position + N)
                    && (ctrl[i].offset == position + N)
            }
            var gateParent = true
            for (i, cache) in cacheP.enumerated() where cache is KVCacheSimple {
                let v = cache.state
                eval(v)
                let b = parentBefore[i]!
                gateParent = gateParent && v[0].shape == b[0].shape
                    && sum(v[0] .== b[0]).item(Int.self) == b[0].size
            }
            report("offset 精确推进(@\(position) → +\(N)): \(gateOffsets ? "✓" : "✗")")
            report("父代完好: \(gateParent ? "✓" : "✗")")
            #expect(gateSegmentForm)
            #expect(gateOffsets)
            #expect(gateParent)

            let allPass = gatePre && gateContinuation && gateSegmentForm && gateOffsets && gateParent
            report(allPass
                ? "*** M3:外化/回来的段表子代沿原来的未来继续生长 — PASS(逐 token \(N)/\(N);性能归 M4)***"
                : "*** M3:FAIL(该配置下;不反推 Track B / M1 / M2)***")
        }
    }
}

extension Array where Element == Int {
    func firstDifferenceIndex(vs other: [Int]) -> Int? {
        for (i, (a, b)) in zip(self, other).enumerated() where a != b {
            return i
        }
        return count == other.count ? nil : Swift.min(count, other.count)
    }
}
