// Copyright © 2026 SimiGo-Lab. Track E(命名待定)Residency M2 — State Exactness
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径;
// 承接 M1 PASS 终判,不回头修改 Track B / M1)
// 记录:SimiGo-Lab trackE-residency/(M1 终判块:M1 PASS → OPEN M2)
//
// M2 靶子 = 泛化性:打掉"M1 只在这个 35B、这个 4616 position、这个
// 单节点配置下成立"的解释。三轴,三个独立测试:
//
//   m2a 深位置(16K / 64K):M1 同形探针在深位置重放。
//   m2b 段表形式直接外化:COW 子代只外化**私有页**(共享前缀留在
//       驻留父代上),重挂回驻留父代——**不先强制物化成连续表示**。
//       验证 residency 不要求表示转换(潜在架构线索)。
//   m2c 多节点隔离:P→A,B;A→A1;B→B1。各自外化/回来:每节点
//       五门全过 + 两两互异保持(A≠B、A1≠B1、A1≠A、A1≠B、B1≠A、
//       B1≠B)+ 父代完好。Execution Tree 的第一步。
//
// 措辞纪律(承 M1 终判):所有 identity 判定 = **状态接地**(五类
// 状态组件逐元素保持 ⇒ Execution Identity preserved),非 literal ID
// 字段。next-input 一律走迁移单元载荷(G2 契约形式,禁自比)。
// sha256 不作门(safetensors 键序逐进程不定);计时描述性,性能归
// M4。本实验失败 ⇒ "该机制在该配置下不成立",不反推 Track B / M1。

import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

@Suite(.serialized)
struct ResidencyM2StateExactness {

    // MARK: - 共享助手

    private func report(_ line: @autoclosure () -> String) {
        print("[ResidencyM2] " + line())
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
                domain: "ResidencyM2", code: 1,
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
    ) -> (Double, Int) {
        var next = input
        let out = lm(
            MLXArray([Int32(next)]),
            cache: cache.map { $0 as KVCache? },
            state: state)
        state = out.state
        eval(out.logits)
        next = out.logits[0, out.logits.dim(1) - 1].argMax().item(Int.self)
        return (0, next)
    }

    private func decode(
        _ lm: Qwen35Language.LanguageModel, cache: [KVCache],
        state: inout LMOutput.State?, input: Int, steps: Int
    ) -> Int {
        var next = input
        for _ in 0..<steps {
            let (_, t) = step(lm, cache: cache, state: &state, input: next)
            next = t
        }
        return next
    }

    /// 节点身份快照(状态接地;全部物化)
    private struct NodeSnapshot {
        let attnIdx: [Int]
        let gdnIdx: [Int]
        let kv: [Int: [MLXArray]]
        let gdn: [Int: [MLXArray]]
        let offsets: [Int]
        let position: Int
        let nextInput: Int
        let stateDict: [String: MLXArray]
        let stateCaptured: Bool
    }

    private func snapshot(
        _ caches: [KVCache], _ state: LMOutput.State?, nextInput: Int
    ) -> NodeSnapshot {
        var attnIdx: [Int] = []
        var gdnIdx: [Int] = []
        for (i, c) in caches.enumerated() {
            switch c {
            case is COWForkKVCache, is KVCacheSimple: attnIdx.append(i)
            case is MambaCache: gdnIdx.append(i)
            default: fatalError("unhandled cache type \(type(of: c))")
            }
        }
        var kv: [Int: [MLXArray]] = [:]
        for i in attnIdx {
            let view = caches[i].state
            eval(view)
            kv[i] = view
        }
        var gdn: [Int: [MLXArray]] = [:]
        for i in gdnIdx {
            let m = caches[i] as! MambaCache
            var slots: [MLXArray] = []
            for s in 0..<m.slotCount {
                let a = m[s]!
                eval(a)
                slots.append(a)
            }
            gdn[i] = slots
        }
        var stateDict: [String: MLXArray] = [:]
        var captured = false
        if let st = state {
            if let dict = try? st.serializedArrays() {
                for a in dict.values { eval(a) }
                stateDict = dict
                captured = true
            }
        }
        let position = caches[attnIdx.first!].offset
        return NodeSnapshot(
            attnIdx: attnIdx, gdnIdx: gdnIdx, kv: kv, gdn: gdn,
            offsets: caches.map { $0.offset }, position: position,
            nextInput: nextInput, stateDict: stateDict, stateCaptured: captured)
    }

    private func payload(_ snap: NodeSnapshot, caches: [KVCache]) -> [String: MLXArray] {
        var p: [String: MLXArray] = [:]
        for i in snap.attnIdx {
            p["a\(i).k"] = snap.kv[i]![0]
            p["a\(i).v"] = snap.kv[i]![1]
        }
        for i in snap.gdnIdx {
            let m = caches[i] as! MambaCache
            for s in 0..<m.slotCount {
                p["g\(i).s\(s)"] = snap.gdn[i]![s]
            }
        }
        for (k, v) in snap.stateDict { p["st.\(k)"] = v }
        p["meta.nextInput"] = MLXArray([Int32(snap.nextInput)])
        p["meta.position"] = MLXArray([Int32(snap.position)])
        return p
    }

    /// 连续表示重建(attention→KVCacheSimple;GDN 槽位;state 重组)
    private func rehydrateContinuous(
        _ snap: NodeSnapshot, layerCount: Int, loaded: [String: MLXArray]
    ) -> ([KVCache], LMOutput.State?) {
        var out = [KVCache?](repeating: nil, count: layerCount)
        for i in snap.attnIdx {
            let c = KVCacheSimple()
            c.state = [loaded["a\(i).k"]!, loaded["a\(i).v"]!]
            out[i] = c
        }
        for i in snap.gdnIdx {
            let m = MambaCache()
            for s in 0..<(snap.gdn[i]?.count ?? 0) {
                m[s] = loaded["g\(i).s\(s)"]!
            }
            m.offset = snap.offsets[i]
            out[i] = m
        }
        var state: LMOutput.State? = nil
        if snap.stateCaptured {
            var dict: [String: MLXArray] = [:]
            for (k, _) in snap.stateDict { dict[k] = loaded["st.\(k)"]! }
            state = LMOutput.State(serializedArrays: dict)
        }
        return (out.map { $0! }, state)
    }

    /// M1 五门(状态接地的 identity 判定);返回(通过, 逐门结果)
    private func identityGates(
        _ snap: NodeSnapshot, cachesAfter: [KVCache], stateAfter: LMOutput.State?,
        loaded: [String: MLXArray]
    ) -> (Bool, [String]) {
        var gatePosition = Int(loaded["meta.position"]!.item(Int32.self)) == snap.position
        for (i, c) in cachesAfter.enumerated() {
            gatePosition = gatePosition && (c.offset == snap.offsets[i])
        }
        let gateNextInput = Int(loaded["meta.nextInput"]!.item(Int32.self)) == snap.nextInput
        var gateKV = true
        for i in snap.attnIdx {
            let view = cachesAfter[i].state
            eval(view)
            let b = snap.kv[i]!
            gateKV = gateKV
                && view[0].shape == b[0].shape && sum(view[0] .== b[0]).item(Int.self) == b[0].size
                && view[1].shape == b[1].shape && sum(view[1] .== b[1]).item(Int.self) == b[1].size
        }
        var gateGDN = true
        for i in snap.gdnIdx {
            let m = cachesAfter[i] as! MambaCache
            for s in 0..<m.slotCount {
                let a = m[s]!
                eval(a)
                let b = snap.gdn[i]![s]
                gateGDN = gateGDN && a.shape == b.shape
                    && sum(a .== b).item(Int.self) == b.size
            }
        }
        var gateState = true
        if snap.stateCaptured {
            if let st = stateAfter, let reDict = try? st.serializedArrays() {
                gateState = Set(reDict.keys) == Set(snap.stateDict.keys)
                if gateState {
                    for (k, b) in snap.stateDict {
                        let a = reDict[k]!
                        eval(a)
                        gateState = gateState && a.shape == b.shape
                            && sum(a .== b).item(Int.self) == b.size
                    }
                }
            } else {
                gateState = false
            }
        }
        let lines = [
            "position: \(gatePosition ? "✓" : "✗")",
            "next-input: \(gateNextInput ? "✓" : "✗")",
            "KV: \(gateKV ? "✓" : "✗")",
            "GDN: \(gateGDN ? "✓" : "✗")",
            "State: \(snap.stateCaptured ? (gateState ? "✓" : "✗") : "n/a")",
        ]
        return (gatePosition && gateNextInput && gateKV && gateGDN && gateState, lines)
    }

    private func roundTripFile(_ payload: [String: MLXArray], tag: String) throws -> URL {
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + "-\(tag).safetensors")
        try save(arrays: payload, url: file)
        return file
    }

    // MARK: - m2a 深位置

    @Test func m2aDeepPositions() async throws {
        report("=== M2a:深位置(16K / 64K)——打掉'浅位置偶然成立'解释 ===")
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

            for depth in [16_384, 65_536] {
                report("--- 深位置 @\(depth) ---")
                let base = self.syntheticIds(depth, seed: 71)
                let suffix = self.syntheticIds(512, seed: 72)

                var cacheP = lm.makeCache(capacity: nil)
                var stateP: LMOutput.State? = nil
                _ = self.prefill(lm, cache: cacheP, ids: base, state: &stateP)
                var r = forkModelCache(cacheP)
                var stateR = stateP
                let logitsSuf = self.prefill(lm, cache: r, ids: suffix, state: &stateR)
                let first = logitsSuf[0, logitsSuf.dim(1) - 1].argMax().item(Int.self)
                let nextIn = self.decode(lm, cache: r, state: &stateR, input: first, steps: 8)

                let snap = self.snapshot(r, stateR, nextInput: nextIn)
                report("resident @\(snap.position)(attention 锚定),nextInput=\(nextIn)")

                let file = try self.roundTripFile(self.payload(snap, caches: r), tag: "m2a-\(depth)")
                let bytes = (try? Data(contentsOf: file))?.count ?? 0
                defer { try? FileManager.default.removeItem(at: file) }
                r = []; cacheP = []; stateR = nil; stateP = nil
                GPU.clearCache()
                report("Externalize \(bytes / 1048576) MiB → 驻留释放 → GPU active \(GPU.activeMemory / 1048576) MiB")

                let loaded = try loadArrays(url: file)
                let (after, stateAfter) = self.rehydrateContinuous(snap, layerCount: snap.offsets.count, loaded: loaded)
                let (pass, lines) = self.identityGates(snap, cachesAfter: after, stateAfter: stateAfter, loaded: loaded)
                report("M2a @\(depth): \(lines.joined(separator: " | ")) → \(pass ? "PASS" : "FAIL")")
                #expect(pass)
            }
            report("*** M2a 深位置:两配置判定如上(状态接地的 Execution Identity preserved)***")
        }
    }

    // MARK: - m2b 段表形式直接外化(不强制物化)

    @Test func m2bSegmentPreserving() async throws {
        report("=== M2b:COW 子代私有页外化 → 重挂回**驻留**父代(不物化) ===")
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

            // P @4096 驻留;C = fork + 后缀 512 + decode 8(私有页 520 行)
            let base = self.syntheticIds(4_096, seed: 81)
            let suffix = self.syntheticIds(512, seed: 82)
            var cacheP = lm.makeCache(capacity: nil)
            var stateP: LMOutput.State? = nil
            _ = self.prefill(lm, cache: cacheP, ids: base, state: &stateP)
            var c = forkModelCache(cacheP)
            var stateC = stateP
            let logitsSuf = self.prefill(lm, cache: c, ids: suffix, state: &stateC)
            let first = logitsSuf[0, logitsSuf.dim(1) - 1].argMax().item(Int.self)
            let nextIn = self.decode(lm, cache: c, state: &stateC, input: first, steps: 8)

            let attn0 = c.firstIndex { $0 is COWForkKVCache }!
            let position = c[attn0].offset
            let forkDepth = (c[attn0] as! COWForkKVCache).forkDepth
            let used = position - forkDepth
            report("子代 resident @\(position)(forkDepth=\(forkDepth),私有 used=\(used) 行)")

            // 快照:子代全视图 + 父代完整性参照
            let snap = self.snapshot(c, stateC, nextInput: nextIn)
            var parentBefore: [Int: [MLXArray]] = [:]
            for (i, cache) in cacheP.enumerated() where cache is KVCacheSimple {
                let v = cache.state
                eval(v)
                parentBefore[i] = v
            }

            // 外化 = **仅私有页** + GDN 槽 + state + 元数据(共享前缀不落盘)
            var payload: [String: MLXArray] = [:]
            for i in snap.attnIdx {
                let cow = c[i] as! COWForkKVCache
                payload["p\(i).k"] = cow.privateKeys![.ellipsis, ..<used, 0...]
                payload["p\(i).v"] = cow.privateValues![.ellipsis, ..<used, 0...]
                eval(payload["p\(i).k"]!, payload["p\(i).v"]!)
            }
            for i in snap.gdnIdx {
                let m = c[i] as! MambaCache
                for s in 0..<m.slotCount { payload["g\(i).s\(s)"] = snap.gdn[i]![s] }
            }
            for (k, v) in snap.stateDict { payload["st.\(k)"] = v }
            payload["meta.nextInput"] = MLXArray([Int32(nextIn)])
            payload["meta.position"] = MLXArray([Int32(position)])
            payload["meta.used"] = MLXArray([Int32(used)])

            let file = try self.roundTripFile(payload, tag: "m2b")
            defer { try? FileManager.default.removeItem(at: file) }
            let bytes = (try? Data(contentsOf: file))?.count ?? 0

            // 释放子代(父代保持驻留!)
            c = []
            stateC = nil
            GPU.clearCache()
            report("Externalize 私有页 \(bytes / 1024) KiB(共享前缀留驻父代);子代释放")

            // 重挂:新 COW 子代引用驻留父代段表 + 私有页回填(**不物化**)
            let loaded = try loadArrays(url: file)
            let usedAfter = Int(loaded["meta.used"]!.item(Int32.self))
            var rebuilt = [KVCache?](repeating: nil, count: snap.offsets.count)
            for i in snap.attnIdx {
                let cow = COWForkKVCache(parent: cacheP[i])
                cow.privateKeys = loaded["p\(i).k"]!
                cow.privateValues = loaded["p\(i).v"]!
                cow.offset = cacheP[i].offset + usedAfter
                rebuilt[i] = cow
            }
            for i in snap.gdnIdx {
                let m = MambaCache()
                for s in 0..<(snap.gdn[i]?.count ?? 0) { m[s] = loaded["g\(i).s\(s)"]! }
                m.offset = snap.offsets[i]
                rebuilt[i] = m
            }
            var stateAfter: LMOutput.State? = nil
            if snap.stateCaptured {
                var dict: [String: MLXArray] = [:]
                for (k, _) in snap.stateDict { dict[k] = loaded["st.\(k)"]! }
                stateAfter = LMOutput.State(serializedArrays: dict)
            }
            let cachesAfter = rebuilt.map { $0! }

            // 表示核查(架构线索):重建子代仍是段表形态,共享行 = 父代 offset
            var gateRepresentation = true
            for i in snap.attnIdx {
                guard let cow = cachesAfter[i] as? COWForkKVCache else {
                    gateRepresentation = false; continue
                }
                gateRepresentation = gateRepresentation
                    && (cow.sharedRows == cacheP[i].offset)
                    && (cow.privateKeys!.dim(2) >= usedAfter)
            }
            report("表示核查:重建子代全部仍为 COW 段表(未物化),sharedRows=父代 offset: \(gateRepresentation ? "✓" : "✗")")

            // 五门(全视图逐元素;视图由段表+私有页拼接而来)
            let (pass, lines) = self.identityGates(snap, cachesAfter: cachesAfter, stateAfter: stateAfter, loaded: loaded)

            // 父代完整性:全程未被子代外化/重挂扰动
            var gateParent = true
            for (i, cache) in cacheP.enumerated() where cache is KVCacheSimple {
                let v = cache.state
                eval(v)
                let b = parentBefore[i]!
                gateParent = gateParent && v[0].shape == b[0].shape
                    && sum(v[0] .== b[0]).item(Int.self) == b[0].size
            }

            report("五门: \(lines.joined(separator: " | ")) → \(pass ? "PASS" : "FAIL")")
            report("父代完整性(驻留全程未扰动): \(gateParent ? "✓" : "✗")")
            #expect(gateRepresentation)
            #expect(pass)
            #expect(gateParent)

            // 冒烟步(诊断)
            var stateSmoke = stateAfter
            let (_, smokeNext) = self.step(lm, cache: cachesAfter, state: &stateSmoke, input: snap.nextInput)
            let smokeOK = cachesAfter[attn0].offset == position + 1
            report("冒烟步(诊断):offset +1 \(smokeOK ? "✓" : "✗"),token=\(smokeNext)")
            #expect(smokeOK)

            report(pass && gateRepresentation && gateParent
                ? "*** M2b:段表形式直接外化成立——residency 不要求先把状态物化成连续表示 ***"
                : "*** M2b:FAIL(该配置下)***")
        }
    }

    // MARK: - m2c 多节点隔离

    @Test func m2cMultiNodeIsolation() async throws {
        report("=== M2c:Execution Tree 第一步——P→A,B;A→A1;B→B1 各自外化/回来 ===")
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

            // 建树:P@8192 → A,B(fork+512+4);A→A1, B→B1(fork+256+4)
            let base = self.syntheticIds(8_192, seed: 91)
            var cacheP = lm.makeCache(capacity: nil)
            var stateP: LMOutput.State? = nil
            _ = self.prefill(lm, cache: cacheP, ids: base, state: &stateP)

            func grow(_ parent: [KVCache], _ pstate: LMOutput.State?, suffix seed: Int, len: Int, decodeSteps: Int)
                -> ([KVCache], LMOutput.State?, Int) {
                var c = forkModelCache(parent)
                var st = pstate
                let ids = self.syntheticIds(len, seed: seed)
                let logits = self.prefill(lm, cache: c, ids: ids, state: &st)
                let first = logits[0, logits.dim(1) - 1].argMax().item(Int.self)
                let next = self.decode(lm, cache: c, state: &st, input: first, steps: decodeSteps)
                return (c, st, next)
            }
            var (a, stateA, nextA) = grow(cacheP, stateP, suffix: 92, len: 512, decodeSteps: 4)
            var (b, stateB, nextB) = grow(cacheP, stateP, suffix: 93, len: 512, decodeSteps: 4)
            var (a1, stateA1, nextA1) = grow(a, stateA, suffix: 94, len: 256, decodeSteps: 4)
            var (b1, stateB1, nextB1) = grow(b, stateB, suffix: 95, len: 256, decodeSteps: 4)
            report("树就绪:P@\(cacheP.firstIndex { $0 is KVCacheSimple }.map { cacheP[$0].offset } ?? 0)(attention)→ A,B → A1,B1")

            let snaps: [String: NodeSnapshot] = [
                "A": self.snapshot(a, stateA, nextInput: nextA),
                "B": self.snapshot(b, stateB, nextInput: nextB),
                "A1": self.snapshot(a1, stateA1, nextInput: nextA1),
                "B1": self.snapshot(b1, stateB1, nextInput: nextB1),
            ]
            let caches: [String: [KVCache]] = ["A": a, "B": b, "A1": a1, "B1": b1]
            for (name, s) in snaps.sorted(by: { $0.key < $1.key }) {
                report("节点 \(name):@\(s.position),nextInput=\(s.nextInput)")
            }

            // 隔离检查(语义化):同深度兄弟两两互异;子代 = 父代前缀 +
            // 严格延伸。跨谱系异深度对(A1 vs B 等)形状已不同,不逐元素比。
            let sameDepthPairs = [("A", "B"), ("A1", "B1")]
            let lineagePairs = [("A1", "A"), ("B1", "B")]  // (child, parent)

            func kViewOf(_ snap: NodeSnapshot, rehydratedDict: [String: ([KVCache], LMOutput.State?, [String])]?, name: String) -> MLXArray {
                let i = snap.attnIdx.first!
                if let r = rehydratedDict, let entry = r[name] { return entry.0[i].state[0] }
                return snap.kv[i]![0]
            }

            func distinctCheck(_ x: String, _ y: String, rehydratedDict: [String: ([KVCache], LMOutput.State?, [String])]?) -> Bool {
                let kx = kViewOf(snaps[x]!, rehydratedDict: rehydratedDict, name: x)
                let ky = kViewOf(snaps[y]!, rehydratedDict: rehydratedDict, name: y)
                precondition(kx.shape == ky.shape, "shape mismatch \(x) vs \(y)")
                return sum(kx .== ky).item(Int.self) != kx.size
            }

            func lineageCheck(_ child: String, _ parent: String, rehydratedDict: [String: ([KVCache], LMOutput.State?, [String])]?) -> (Bool, Bool) {
                let kc = kViewOf(snaps[child]!, rehydratedDict: rehydratedDict, name: child)
                let kp = kViewOf(snaps[parent]!, rehydratedDict: rehydratedDict, name: parent)
                let d = kp.dim(2)
                let prefixOK = sum(kc[.ellipsis, ..<d, 0...] .== kp).item(Int.self) == kp.size
                let tail = kc[.ellipsis, d..<kc.dim(2), 0...]
                let pTail = kp[.ellipsis, (d - tail.dim(2))..<d, 0...]
                let extends = sum(tail .== pTail).item(Int.self) != tail.size
                return (prefixOK, extends)
            }

            var gateDistinctBefore = true
            for (x, y) in sameDepthPairs {
                let d = distinctCheck(x, y, rehydratedDict: nil)
                if !d { report("互异失败(重建前): \(x) == \(y)") }
                gateDistinctBefore = gateDistinctBefore && d
            }
            var gateLineageBefore = true
            for (child, parent) in lineagePairs {
                let (p, e) = lineageCheck(child, parent, rehydratedDict: nil)
                if !p { report("谱系失败(重建前): \(child) 前缀 != \(parent)") }
                if !e { report("谱系失败(重建前): \(child) 未严格延伸 \(parent)") }
                gateLineageBefore = gateLineageBefore && p && e
            }
            report("重建前:同深度互异 \(gateDistinctBefore ? "✓" : "✗") | 谱系(前缀+延伸) \(gateLineageBefore ? "✓" : "✗")")

            // 父代完整性参照
            var parentBefore: [Int: [MLXArray]] = [:]
            for (i, cache) in cacheP.enumerated() where cache is KVCacheSimple {
                let v = cache.state
                eval(v)
                parentBefore[i] = v
            }

            // 各节点独立外化 → 全部释放(父代留驻)→ 各自独立回来
            var files: [String: URL] = [:]
            for (name, s) in snaps.sorted(by: { $0.key < $1.key }) {
                files[name] = try self.roundTripFile(self.payload(s, caches: caches[name]!), tag: "m2c-\(name)")
            }
            a = []; b = []; a1 = []; b1 = []
            stateA = nil; stateB = nil; stateA1 = nil; stateB1 = nil
            GPU.clearCache()
            report("四节点外化完成并释放(父代留驻);逐节点独立 rehydrate")

            var allPass = gateDistinctBefore && gateLineageBefore
            var gateDistinctAfter = true
            var rehydrated: [String: ([KVCache], LMOutput.State?, [String])] = [:]
            for (name, s) in snaps.sorted(by: { $0.key < $1.key }) {
                let loaded = try loadArrays(url: files[name]!)
                let (cachesAfter, stateAfter) = self.rehydrateContinuous(s, layerCount: s.offsets.count, loaded: loaded)
                let (pass, lines) = self.identityGates(s, cachesAfter: cachesAfter, stateAfter: stateAfter, loaded: loaded)
                rehydrated[name] = (cachesAfter, stateAfter, lines)
                report("节点 \(name) 五门: \(lines.joined(separator: " | ")) → \(pass ? "PASS" : "FAIL")")
                allPass = allPass && pass
            }
            for (x, y) in sameDepthPairs {
                let d = distinctCheck(x, y, rehydratedDict: rehydrated)
                if !d { report("互异失败(重建后): \(x) == \(y)") }
                gateDistinctAfter = gateDistinctAfter && d
            }
            var gateLineageAfter = true
            for (child, parent) in lineagePairs {
                let (p, e) = lineageCheck(child, parent, rehydratedDict: rehydrated)
                if !p { report("谱系失败(重建后): \(child) 前缀 != \(parent)") }
                if !e { report("谱系失败(重建后): \(child) 未严格延伸 \(parent)") }
                gateLineageAfter = gateLineageAfter && p && e
            }
            report("重建后:同深度互异 \(gateDistinctAfter ? "✓" : "✗") | 谱系(前缀+延伸) \(gateLineageAfter ? "✓" : "✗")")

            var gateParent = true
            for (i, cache) in cacheP.enumerated() where cache is KVCacheSimple {
                let v = cache.state
                eval(v)
                let bp = parentBefore[i]!
                gateParent = gateParent && v[0].shape == bp[0].shape
                    && sum(v[0] .== bp[0]).item(Int.self) == bp[0].size
            }
            report("父代 P 完好: \(gateParent ? "✓" : "✗")")
            #expect(gateDistinctBefore)
            #expect(gateLineageBefore)
            #expect(gateDistinctAfter)
            #expect(gateLineageAfter)
            #expect(gateParent)
            #expect(allPass)

            let m2cPass = allPass && gateDistinctAfter && gateLineageAfter && gateParent
            report(m2cPass
                ? "*** M2c:多节点各自跨驻留边界 + 互异保持 + 谱系保持 + 父代完好 — PASS ***"
                : "*** M2c:FAIL(该配置下)***")
        }
    }
}
