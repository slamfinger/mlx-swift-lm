// Copyright © 2026 SimiGo-Lab. Track E(命名待定)Residency M4 —
// Rehydration Tax(cost 轴;交替对照)
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径;
// 承接 M3-C 源码级 FINAL PASS;M3 系冻结于 30b61e6,不回头修改)
// 记录:SimiGo-Lab trackE-residency/(M3-C 终判块:M4 开题)
//
// 核心问题(M3-C 终判定序):相同 execution state、相同 continuation
// workload 下,rehydration 相对 resident continuation 的额外成本?
//
// 纪律:alternating control/treatment;same execution state;same
// continuation workload;hot-throttling awareness(轮换先后序 +
// 中位数,抵消顺序热偏差);不混冷模型/不同 prefill/cache temperature。
//
// 计量分两部分:
//   ① 一次性边界成本:externalize(save)ms + rehydrate(load+重挂)ms
//     + 载荷字节;换算 decode-token 等价数(一次性成本 / ctrl 稳态
//     每步中位)。
//   ② 稳态 decode 税:重挂后的 treat vs 驻留 ctrl,交替 6 轮 × 16 步
//     (偶数轮 ctrl 先、奇数轮 treat 先),各轮均值取中位,tax% =
//     treat/ctrl − 1。
// 正确性守卫:全部测量步 treat 与 ctrl token 逐 token 相等(非门,
// 是测量有效性的前提)。M4 无成本 gate——数字即结果,优劣判断留
// Residency Policy。

import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

@Suite(.serialized)
struct ResidencyM4RehydrationTax {

    private func report(_ line: @autoclosure () -> String) {
        print("[ResidencyM4] " + line())
    }

    private func ms(_ duration: Duration) -> Double {
        let ns = Double(duration.components.seconds) * 1e9
            + Double(duration.components.attoseconds) / 1e9
        return ns / 1e6
    }

    private func median(_ xs: [Double]) -> Double {
        let s = xs.sorted()
        let n = s.count
        return n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
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
                domain: "ResidencyM4", code: 1,
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

    @Test func m4RehydrationTax() async throws {
        report("=== M4:Rehydration Tax(交替对照;一次性边界成本 + 稳态 decode 税)===")
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

            let decodeN = 8
            let suffixLen = 512
            let rounds = 6
            let stepsPerRound = 16
            let warmup = 8
            let clock = ContinuousClock()

            for (di, depth) in [4_096, 16_384].enumerated() {
                report("--- config @\(depth + suffixLen + decodeN)(base \(depth))---")

                // ---- 1. 驻留父代 + 孪生双子代
                let base = self.syntheticIds(depth, seed: 141 + di)
                let suffix = self.syntheticIds(suffixLen, seed: 142)
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
                let position = treat[attnIdx.first!].offset
                let gdnSlotCounts = Dictionary(uniqueKeysWithValues: gdnIdx.map {
                    ($0, (treat[$0] as! MambaCache).slotCount)
                })
                let offsetsBefore = treat.map { $0.offset }
                let layerCount = treat.count   // 释放前捕获(M3-C 教训:释放后引用)
                let used = position - depth

                // 前置门(测量有效性前提)
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
                #expect(gatePre)
                report("前置门(treat ≡ ctrl): \(gatePre ? "✓" : "✗")")

                // ---- 2. 一次性边界成本:externalize
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
                    .appendingPathComponent(UUID().uuidString + "-m4-\(depth).safetensors")
                defer { try? FileManager.default.removeItem(at: file) }
                let tExt = clock.measure {
                    try! save(arrays: payload, url: file)
                }
                let extMs = self.ms(tExt)
                let bytes = (try? Data(contentsOf: file))?.count ?? 0

                treat = []
                stateTreat = nil
                GPU.clearCache()

                // ---- 3. rehydrate(load + 段表重挂 + GDN offset 显式恢复;
                //      单遍计量:直接构建保留测量臂,不做无计量重复)
                var rebuilt = [KVCache?](repeating: nil, count: layerCount)
                var loaded: [String: MLXArray] = [:]
                var stateTA: LMOutput.State? = nil
                let tRehy = clock.measure {
                    loaded = try! loadArrays(url: file)
                    let usedAfter = Int(loaded["meta.used"]!.item(Int32.self))
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
                }
                let rehyMs = self.ms(tRehy)
                let stKeys = loaded.keys.filter { $0.hasPrefix("st.") }.sorted()
                if !stKeys.isEmpty {
                    var dict: [String: MLXArray] = [:]
                    for k in stKeys {
                        dict[String(k.dropFirst(3))] = loaded[k]!
                    }
                    stateTA = LMOutput.State(serializedArrays: dict)
                }
                let treatArm = rebuilt.map { $0! }
                report(String(format: "一次性边界成本: externalize %.1f ms + rehydrate %.1f ms(载荷 %d KiB;含段表重挂+GDN offset 恢复)", extMs, rehyMs, bytes / 1024))

                // ---- 4. 稳态 decode 税(交替对照;先 warmup 排除首触效应)
                var inC = nextCtrl
                var inT = Int(loaded["meta.nextInput"]!.item(Int32.self))
                var tokensC: [Int] = []
                var tokensT: [Int] = []
                for _ in 0..<warmup {
                    inC = self.step(lm, cache: ctrl, state: &stateCtrl, input: inC)
                    inT = self.step(lm, cache: treatArm, state: &stateTA, input: inT)
                    tokensC.append(inC)
                    tokensT.append(inT)
                }
                var ctrlMeans: [Double] = []
                var treatMeans: [Double] = []
                for r in 0..<rounds {
                    let leadCtrl = (r % 2 == 0)
                    if leadCtrl {
                        let t = clock.measure {
                            for _ in 0..<stepsPerRound {
                                inC = self.step(lm, cache: ctrl, state: &stateCtrl, input: inC)
                                tokensC.append(inC)
                            }
                        }
                        ctrlMeans.append(self.ms(t) / Double(stepsPerRound))
                        let t2 = clock.measure {
                            for _ in 0..<stepsPerRound {
                                inT = self.step(lm, cache: treatArm, state: &stateTA, input: inT)
                                tokensT.append(inT)
                            }
                        }
                        treatMeans.append(self.ms(t2) / Double(stepsPerRound))
                    } else {
                        let t = clock.measure {
                            for _ in 0..<stepsPerRound {
                                inT = self.step(lm, cache: treatArm, state: &stateTA, input: inT)
                                tokensT.append(inT)
                            }
                        }
                        treatMeans.append(self.ms(t) / Double(stepsPerRound))
                        let t2 = clock.measure {
                            for _ in 0..<stepsPerRound {
                                inC = self.step(lm, cache: ctrl, state: &stateCtrl, input: inC)
                                tokensC.append(inC)
                            }
                        }
                        ctrlMeans.append(self.ms(t2) / Double(stepsPerRound))
                    }
                }
                let medC = self.median(ctrlMeans)
                let medT = self.median(treatMeans)
                let tax = medT / medC - 1.0
                let tokenGuard = (tokensC == tokensT)
                let tokenEquiv = (extMs + rehyMs) / medC
                report(String(format: "交替稳态(%d 轮×%d 步,轮换先后): ctrl 中位 %.2f ms/tok(%.2f–%.2f) | treat 中位 %.2f ms/tok(%.2f–%.2f)",
                    rounds, stepsPerRound, medC, ctrlMeans.min()!, ctrlMeans.max()!, medT, treatMeans.min()!, treatMeans.max()!))
                report(String(format: "稳态 decode 税 = %+.1f%% | 一次性边界成本 ≈ %.0f decode-token 等价 | token 守卫 %@",
                    tax * 100, tokenEquiv, tokenGuard ? "✓ 全程逐 token 相等" : "✗"))
                #expect(gatePre)
                #expect(tokenGuard)

                report("*** config @\(position): 数字即结果,优劣判断留 Residency Policy ***")
                ctrl = []
                cacheP = []
                stateCtrl = nil
                stateP = nil
                GPU.clearCache()
            }
            report("*** M4 完成:两 config 数字如上(交替对照中位数;热节流敏感机,单机单模型,不外推)***")
        }
    }
}
