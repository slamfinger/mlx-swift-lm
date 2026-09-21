// Copyright © 2026 SimiGo-Lab. Track E(命名待定)Residency M3-B —
// Deep Position Continuation(depth 轴:16K / 64K,段表路径)
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径;
// 承接 M3-A FINAL PASS 终判;独立实验,不修改已 PASS 的 M1/M2/M3/M3-A)
// 记录:SimiGo-Lab trackE-residency/(M3-A 终判块:M3-B 设计收紧)
//
// 靶子(审核定序):"continuation correctness 是否只是浅位置 @4616
// 的偶然现象"。设计收紧:不再三臂重复 M3-A——**只走段表路径**,
// 16K 与 64K 两个深度,各与驻留孪生 control 逐 token 对照。
// 轴分离:M3-A=representation / M3-B=depth / M3-C=topology / M4=cost。
//
//   每个深度:
//   P @depth(驻留父代)
//     ├── treat = fork → 后缀512 → decode8 → 私有页外化 → 释放 →
//     │         重挂回驻留父代(仍为 COW 段表)→ 继续 96 步
//     └── ctrl  = fork → 后缀512 → decode8 → 驻留全程 → 继续 96 步
//
//   可证伪前置门 + 主门 + 辅门,与 M3 同型。

import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

@Suite(.serialized)
struct ResidencyM3BDeepContinuation {

    private func report(_ line: @autoclosure () -> String) {
        print("[ResidencyM3B] " + line())
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
                domain: "ResidencyM3B", code: 1,
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

    @Test func m3bDeepPositionContinuation() async throws {
        report("=== M3-B:深位置段表 continuation(16K / 64K,depth 轴单变量)===")
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
            let decodeN = 8
            let suffixLen = 512
            let clock = ContinuousClock()
            var allPass = true

            for (di, depth) in [16_384, 65_536].enumerated() {
                report("--- depth \(depth) ---")
                let base = self.syntheticIds(depth, seed: 121 + di)
                let suffix = self.syntheticIds(suffixLen, seed: 122)

                // ---- 1. 父代 + 孪生双子代
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
                let gdnSlotCounts = Dictionary(uniqueKeysWithValues: gdnIdx.map {
                    ($0, (treat[$0] as! MambaCache).slotCount)
                })
                let used = position - depth
                report("双子代就绪 @\(position)(forkDepth=\(depth)),私有 used=\(used) 行,nextInput=\(nextTreat)")

                // ---- 2. 前置门
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
                report("前置门(treat ≡ ctrl): \(gatePre ? "✓" : "✗")")

                var parentBefore: [Int: [MLXArray]] = [:]
                for (i, cache) in cacheP.enumerated() where cache is KVCacheSimple {
                    let v = cache.state
                    eval(v)
                    parentBefore[i] = v
                }

                // ---- 3. treat 私有页外化并释放
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
                if let st = stateTreat, let dict = try? st.serializedArrays() {
                    for (k, v) in dict {
                        eval(v)
                        payload["st.\(k)"] = v
                    }
                }
                payload["meta.nextInput"] = MLXArray([Int32(nextTreat)])
                payload["meta.used"] = MLXArray([Int32(used)])

                let file = FileManager.default.temporaryDirectory
                    .appendingPathComponent(UUID().uuidString + "-m3b-\(depth).safetensors")
                defer { try? FileManager.default.removeItem(at: file) }
                try save(arrays: payload, url: file)
                let bytes = (try? Data(contentsOf: file))?.count ?? 0

                treat = []
                stateTreat = nil
                GPU.clearCache()
                report("treat 外化(私有页 \(used) 行,载荷 \(bytes / 1024) KiB)并释放;ctrl 驻留")

                // ---- 4. 段表重挂
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
                report("重建核查:treat 仍为 COW 段表(sharedRows=父代 offset): \(gateSegmentForm ? "✓" : "✗")")

                // ---- 5. 主门
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
                report(String(format: "主门 @\(depth):Continuation(treat) == Continuation(ctrl),N=%d: %@(%.1f s,描述性)",
                    N, gateContinuation ? "✓ \(N)/\(N) 全等" : "✗ 分歧", ms(tRun) / 1000.0))

                // ---- 6. 辅门
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
                report("辅门 @\(depth):offset +\(N) \(gateOffsets ? "✓" : "✗") | 父代完好 \(gateParent ? "✓" : "✗")")

                let depthPass = gatePre && gateSegmentForm && gateContinuation && gateOffsets && gateParent
                report(depthPass ? "*** depth \(depth): PASS ***" : "*** depth \(depth): FAIL ***")
                #expect(gatePre)
                #expect(gateSegmentForm)
                #expect(gateContinuation)
                #expect(gateOffsets)
                #expect(gateParent)
                allPass = allPass && depthPass

                // 释放本轮,进入下一深度
                ctrl = []; cacheP = []
                stateCtrl = nil; stateP = nil
                GPU.clearCache()
            }

            report(allPass
                ? "*** M3-B:16K 与 64K 深位置段表 continuation 均 PASS——'浅位置偶然'解释在测试深度上不成立(状态接地,配置限定)***"
                : "*** M3-B:FAIL(某深度;不反推 Track B / M1 / M2 / M3 / M3-A)***")
        }
    }
}
