// Copyright © 2026 SimiGo-Lab. Track E(命名待定)Residency M3-C —
// Multi-node Tree Continuation(topology 轴,三层 invariant)
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径;
// 承接 M3-B FINAL PASS 终判;独立实验,不修改已 PASS 的 M1/M2/M3/M3-A/M3-B)
// 记录:SimiGo-Lab trackE-residency/(M3-B 终判块:M3-C 开题方向)
//
// 靶子(topology coincidence):"continuation 只是单节点碰巧能继续"。
// 三层 invariant(M3-B 终判开题方向,一起设计):
//   ① sibling isolation:A ≠ B、A1 ≠ B1(元素级 + continuation 轨迹级);
//   ② parent→child lineage:A1 的前缀 == A 且严格延伸;历史行经独立
//      residency 循环 + 双方 continuation 后仍不可变;
//   ③ 各节点独立 residency/continuation:四节点各自私有页外化/释放/
//      按树序重挂/与各自驻留孪生 control 逐 token 对照。
//
// 树(全部段表路径;**树序重挂** = 子节点重挂到"已重挂的父代"上):
//   P @8192 → A_t/A_c, B_t/B_c(suffix512+decode4,seed 132/133)
//           → A1_t/A1_c(fork A_*;suffix256+decode4,seed 134)
//           → B1_t/B1_c(fork B_*;suffix256+decode4,seed 135)
//
// 前向修正(承 M3-B 审计注记):GDN/Mamba offset 显式恢复
// (m.offset = 捕获值)并设独立恢复门;"offset +N" 结论限定 attention。

import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

/// 节点元数据(释放前捕获;file scope —— 闭包内不得声明类型)
private struct NodeMeta {
    let offsets: [Int]
    let gdnSlots: [Int: Int]
    let nextInput: Int
    let used: Int
    let forkDepth: Int
    let position: Int
    var stateDict: [String: MLXArray] = [:]
}

@Suite(.serialized)
struct ResidencyM3CTopologyContinuation {

    private func report(_ line: @autoclosure () -> String) {
        print("[ResidencyM3C] " + line())
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
                domain: "ResidencyM3C", code: 1,
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

    @Test func m3cMultiNodeTreeContinuation() async throws {
        report("=== M3-C:多节点树 continuation(四节点独立 residency;三层 invariant)===")
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
            let base = self.syntheticIds(8_192, seed: 131)
            let clock = ContinuousClock()

            // ---- 1. 建树:treat 与 ctrl 双分支同构
            var cacheP = lm.makeCache(capacity: nil)
            var stateP: LMOutput.State? = nil
            _ = self.prefill(lm, cache: cacheP, ids: base, state: &stateP)

            func grow(_ parent: [KVCache], _ pstate: LMOutput.State?, seed: Int, len: Int, steps: Int)
                -> ([KVCache], LMOutput.State?, Int) {
                var c = forkModelCache(parent)
                var st = pstate
                let ids = self.syntheticIds(len, seed: seed)
                let logits = self.prefill(lm, cache: c, ids: ids, state: &st)
                var next = logits[0, logits.dim(1) - 1].argMax().item(Int.self)
                for _ in 0..<steps {
                    next = self.step(lm, cache: c, state: &st, input: next)
                }
                return (c, st, next)
            }
            var (aT, stateAT, nextAT) = grow(cacheP, stateP, seed: 132, len: 512, steps: 4)
            var (aC, stateAC, nextAC) = grow(cacheP, stateP, seed: 132, len: 512, steps: 4)
            var (bT, stateBT, nextBT) = grow(cacheP, stateP, seed: 133, len: 512, steps: 4)
            var (bC, stateBC, nextBC) = grow(cacheP, stateP, seed: 133, len: 512, steps: 4)
            var (a1T, stateA1T, nextA1T) = grow(aT, stateAT, seed: 134, len: 256, steps: 4)
            var (a1C, stateA1C, nextA1C) = grow(aC, stateAC, seed: 134, len: 256, steps: 4)
            var (b1T, stateB1T, nextB1T) = grow(bT, stateBT, seed: 135, len: 256, steps: 4)
            var (b1C, stateB1C, nextB1C) = grow(bC, stateBC, seed: 135, len: 256, steps: 4)

            let attnIdx: [Int] = aT.enumerated().compactMap {
                ($1 is COWForkKVCache || $1 is KVCacheSimple) ? $0 : nil
            }
            let gdnIdx: [Int] = aT.enumerated().compactMap { $1 is MambaCache ? $0 : nil }
            let layerCount = aT.count
            let position = aT[attnIdx.first!].offset            // 8708
            let position1 = a1T[attnIdx.first!].offset          // 8968
            report("树就绪:A/B @\(position),A1/B1 @\(position1)(P@8192)")

            // ---- 2. 节点元数据捕获(释放前)
            func metaFor(_ c: [KVCache], _ st: LMOutput.State?, _ next: Int, forkDepth: Int) -> NodeMeta {
                let gdnSlots = Dictionary(uniqueKeysWithValues: gdnIdx.map {
                    ($0, (c[$0] as! MambaCache).slotCount)
                })
                var m = NodeMeta(
                    offsets: c.map { $0.offset }, gdnSlots: gdnSlots, nextInput: next,
                    used: c[attnIdx.first!].offset - forkDepth, forkDepth: forkDepth,
                    position: c[attnIdx.first!].offset)
                if let s = st, let dict = try? s.serializedArrays() {
                    for (k, v) in dict { eval(v) }
                    m.stateDict = dict
                }
                return m
            }
            let metas: [String: NodeMeta] = [
                "A": metaFor(aT, stateAT, nextAT, forkDepth: 8192),
                "B": metaFor(bT, stateBT, nextBT, forkDepth: 8192),
                "A1": metaFor(a1T, stateA1T, nextA1T, forkDepth: 8708),
                "B1": metaFor(b1T, stateB1T, nextB1T, forkDepth: 8708),
            ]

            // 原始视图快照(谱系/历史不变门用)
            func views(_ c: [KVCache]) -> [Int: [MLXArray]] {
                var v: [Int: [MLXArray]] = [:]
                for i in attnIdx {
                    let view = c[i].state
                    eval(view)
                    v[i] = view
                }
                return v
            }
            let origA = views(aT), origB = views(bT), origA1 = views(a1T), origB1 = views(b1T)

            // ---- 3. 前置门(三层 invariant 的元素级形式)
            func elementEqual(_ t: [KVCache], _ c: [KVCache]) -> Bool {
                var ok = true
                for i in attnIdx {
                    let vt = t[i].state
                    let vc = c[i].state
                    eval(vt, vc)
                    ok = ok && vt[0].shape == vc[0].shape
                        && sum(vt[0] .== vc[0]).item(Int.self) == vc[0].size
                        && vt[1].shape == vc[1].shape
                        && sum(vt[1] .== vc[1]).item(Int.self) == vc[1].size
                }
                for i in gdnIdx {
                    let mt = t[i] as! MambaCache
                    let mc = c[i] as! MambaCache
                    for s in 0..<mt.slotCount {
                        let at = mt[s]!
                        let ac = mc[s]!
                        eval(at, ac)
                        ok = ok && at.shape == ac.shape
                            && sum(at .== ac).item(Int.self) == ac.size
                    }
                }
                return ok
            }
            let gatePreA = elementEqual(aT, aC) && (nextAT == nextAC)
            let gatePreB = elementEqual(bT, bC) && (nextBT == nextBC)
            let gatePreA1 = elementEqual(a1T, a1C) && (nextA1T == nextA1C)
            let gatePreB1 = elementEqual(b1T, b1C) && (nextB1T == nextB1C)
            report("前置门(treat ≡ ctrl×4): A \(gatePreA ? "✓" : "✗") B \(gatePreB ? "✓" : "✗") A1 \(gatePreA1 ? "✓" : "✗") B1 \(gatePreB1 ? "✓" : "✗")")

            // ① sibling isolation(元素级):A_t ≠ B_t,A1_t ≠ B1_t
            func distinctKV(_ x: [Int: [MLXArray]], _ y: [Int: [MLXArray]]) -> Bool {
                let i = attnIdx.first!
                let kx = x[i]![0]
                let ky = y[i]![0]
                return sum(kx .== ky).item(Int.self) != kx.size
            }
            let gateSibPre = distinctKV(origA, origB) && distinctKV(origA1, origB1)
            // ② lineage(元素级):A1_t 前缀 == A_t 且严格延伸;B1/B 同
            func lineage(_ child: [Int: [MLXArray]], _ parent: [Int: [MLXArray]]) -> (Bool, Bool) {
                let i = attnIdx.first!
                let kc = child[i]![0]
                let kp = parent[i]![0]
                let d = kp.dim(2)
                let prefixOK = sum(kc[.ellipsis, ..<d, 0...] .== kp).item(Int.self) == kp.size
                let tail = kc[.ellipsis, d..<kc.dim(2), 0...]
                let pTail = kp[.ellipsis, (d - tail.dim(2))..<d, 0...]
                let extends = sum(tail .== pTail).item(Int.self) != tail.size
                return (prefixOK, extends)
            }
            let linA = lineage(origA1, origA)
            let linB = lineage(origB1, origB)
            let gateLineagePre = gateSibPre && linA.0 && linA.1 && linB.0 && linB.1
            report("① sibling isolation(前): \(gateSibPre ? "✓" : "✗") | ② lineage(前): A1⊆A \(linA.0 ? "✓" : "✗")+\(linA.1 ? "延伸✓" : "延伸✗") B1⊆B \(linB.0 ? "✓" : "✗")+\(linB.1 ? "延伸✓" : "延伸✗")")
            #expect(gatePreA); #expect(gatePreB); #expect(gatePreA1); #expect(gatePreB1)
            #expect(gateLineagePre)

            // 父代完整性参照
            var parentBefore: [Int: [MLXArray]] = [:]
            for (i, cache) in cacheP.enumerated() where cache is KVCacheSimple {
                let v = cache.state
                eval(v)
                parentBefore[i] = v
            }

            // ---- 3b. 四节点独立外化(私有页 + GDN + state + 单元元数据)
            func externalize(_ t: [KVCache], _ m: NodeMeta, _ tag: String) throws -> URL {
                var payload: [String: MLXArray] = [:]
                for i in attnIdx {
                    let cow = t[i] as! COWForkKVCache
                    payload["p\(i).k"] = cow.privateKeys![.ellipsis, ..<m.used, 0...]
                    payload["p\(i).v"] = cow.privateValues![.ellipsis, ..<m.used, 0...]
                    eval(payload["p\(i).k"]!, payload["p\(i).v"]!)
                }
                for i in gdnIdx {
                    let mm = t[i] as! MambaCache
                    for s in 0..<mm.slotCount {
                        let a = mm[s]!
                        eval(a)
                        payload["g\(i).s\(s)"] = a
                    }
                }
                for (k, v) in m.stateDict { payload["st.\(k)"] = v }
                payload["meta.nextInput"] = MLXArray([Int32(m.nextInput)])
                payload["meta.used"] = MLXArray([Int32(m.used)])
                payload["meta.forkDepth"] = MLXArray([Int32(m.forkDepth)])
                let f = FileManager.default.temporaryDirectory
                    .appendingPathComponent(UUID().uuidString + "-m3c-\(tag).safetensors")
                try save(arrays: payload, url: f)
                return f
            }
            let files: [String: URL] = [
                "A": try externalize(aT, metas["A"]!, "A"),
                "B": try externalize(bT, metas["B"]!, "B"),
                "A1": try externalize(a1T, metas["A1"]!, "A1"),
                "B1": try externalize(b1T, metas["B1"]!, "B1"),
            ]
            defer {
                for f in files.values { try? FileManager.default.removeItem(at: f) }
            }

            aT = []; bT = []; a1T = []; b1T = []
            stateAT = nil; stateBT = nil; stateA1T = nil; stateB1T = nil
            GPU.clearCache()
            report("四节点独立外化完成并释放(ctrl 四臂驻留);按树序重挂:A → A1(挂到重挂后的 A),B → B1")

            // ---- 4. 树序重挂 + GDN offset 显式恢复(前向修正)
            func reattach(_ name: String, parentArm: [KVCache]?) -> ([KVCache], LMOutput.State?) {
                let m = metas[name]!
                let loaded = try! loadArrays(url: files[name]!)
                let usedAfter = Int(loaded["meta.used"]!.item(Int32.self))
                var out = [KVCache?](repeating: nil, count: layerCount)
                for i in attnIdx {
                    let cow: COWForkKVCache
                    if let pa = parentArm {
                        cow = COWForkKVCache(parent: pa[i])
                    } else {
                        cow = COWForkKVCache(parent: cacheP[i])
                    }
                    cow.privateKeys = loaded["p\(i).k"]!
                    cow.privateValues = loaded["p\(i).v"]!
                    cow.offset = cow.forkDepth + usedAfter
                    out[i] = cow
                }
                for i in gdnIdx {
                    let mm = MambaCache()
                    for s in 0..<(m.gdnSlots[i] ?? 0) {
                        mm[s] = loaded["g\(i).s\(s)"]!
                    }
                    mm.offset = m.offsets[i]   // 前向修正:GDN offset 显式恢复
                    out[i] = mm
                }
                var state: LMOutput.State? = nil
                let stKeys = loaded.keys.filter { $0.hasPrefix("st.") }.sorted()
                if !stKeys.isEmpty {
                    var dict: [String: MLXArray] = [:]
                    for k in stKeys { dict[String(k.dropFirst(3))] = loaded[k]! }
                    state = LMOutput.State(serializedArrays: dict)
                }
                return (out.map { $0! }, state)
            }
            var (armA, stateArmA) = reattach("A", parentArm: nil)
            var (armB, stateArmB) = reattach("B", parentArm: nil)
            var (armA1, stateArmA1) = reattach("A1", parentArm: armA)   // 挂到重挂后的 A
            var (armB1, stateArmB1) = reattach("B1", parentArm: armB)   // 挂到重挂后的 B

            // 结构门:段表形态 + 树序链(forkDepth 链)+ GDN offset 恢复(独立门)
            var gateStruct = true
            for i in attnIdx {
                let cowA = armA[i] as? COWForkKVCache
                let cowB = armB[i] as? COWForkKVCache
                let cowA1 = armA1[i] as? COWForkKVCache
                let cowB1 = armB1[i] as? COWForkKVCache
                gateStruct = gateStruct
                    && cowA != nil && cowB != nil && cowA1 != nil && cowB1 != nil
                if let cA = cowA, let cB = cowB, let cA1 = cowA1, let cB1 = cowB1 {
                    gateStruct = gateStruct
                        && (cA.sharedRows == 8192) && (cA.offset == position)
                        && (cB.sharedRows == 8192) && (cB.offset == position)
                        && (cA1.forkDepth == position) && (cA1.sharedRows == position)
                        && (cA1.offset == position1)
                        && (cB1.forkDepth == position) && (cB1.sharedRows == position)
                        && (cB1.offset == position1)
                }
            }
            var gateGDNOffset = true
            for node in [("A", armA), ("B", armB), ("A1", armA1), ("B1", armB1)] as [(String, [KVCache])] {
                let m = metas[node.0]!
                for i in gdnIdx {
                    gateGDNOffset = gateGDNOffset
                        && (node.1[i].offset == m.offsets[i])
                }
            }
            report("结构门:四臂仍为 COW 段表,树序链(A1 forkDepth=A offset=\(position))✓/\(gateStruct ? "✓" : "✗") | ③ GDN offset 显式恢复独立门: \(gateGDNOffset ? "✓" : "✗")(前向修正)")
            #expect(gateStruct)
            #expect(gateGDNOffset)

            // ---- 5. 主门:八臂各自继续 N 步;四对逐 token 对照
            var inA = metas["A"]!.nextInput, inAc = nextAC
            var inB = metas["B"]!.nextInput, inBc = nextBC
            var inA1 = metas["A1"]!.nextInput, inA1c = nextA1C
            var inB1 = metas["B1"]!.nextInput, inB1c = nextB1C
            var tkA: [Int] = [], tkAc: [Int] = []
            var tkB: [Int] = [], tkBc: [Int] = []
            var tkA1: [Int] = [], tkA1c: [Int] = []
            var tkB1: [Int] = [], tkB1c: [Int] = []
            let tRun = clock.measure {
                for _ in 0..<N {
                    inA = self.step(lm, cache: armA, state: &stateArmA, input: inA)
                    inAc = self.step(lm, cache: aC, state: &stateAC, input: inAc)
                    inB = self.step(lm, cache: armB, state: &stateArmB, input: inB)
                    inBc = self.step(lm, cache: bC, state: &stateBC, input: inBc)
                    inA1 = self.step(lm, cache: armA1, state: &stateArmA1, input: inA1)
                    inA1c = self.step(lm, cache: a1C, state: &stateA1C, input: inA1c)
                    inB1 = self.step(lm, cache: armB1, state: &stateArmB1, input: inB1)
                    inB1c = self.step(lm, cache: b1C, state: &stateB1C, input: inB1c)
                    tkA.append(inA); tkAc.append(inAc)
                    tkB.append(inB); tkBc.append(inBc)
                    tkA1.append(inA1); tkA1c.append(inA1c)
                    tkB1.append(inB1); tkB1c.append(inB1c)
                }
            }
            let gateContA = (tkA == tkAc)
            let gateContB = (tkB == tkBc)
            let gateContA1 = (tkA1 == tkA1c)
            let gateContB1 = (tkB1 == tkB1c)
            report(String(format: "主门(各节点 vs 各自驻留 control,N=%d): A %@ | B %@ | A1 %@ | B1 %@(%.1f s,描述性)",
                N, gateContA ? "96/96 ✓" : "✗", gateContB ? "96/96 ✓" : "✗",
                gateContA1 ? "96/96 ✓" : "✗", gateContB1 ? "96/96 ✓" : "✗", ms(tRun) / 1000.0))
            #expect(gateContA); #expect(gateContB); #expect(gateContA1); #expect(gateContB1)

            // ---- 6. 三层 invariant 的 continuation 级形式
            // ① sibling isolation(轨迹级):兄弟 continuation 互不相同
            let gateSibPost = (tkA != tkB) && (tkA1 != tkB1) && (tkA != tkA1)
            // ② lineage(结构级):continuation 后历史行不可变;A1 仍含 A 前缀
            var gateHistA = true, gateHistA1 = true, gateLineagePost = true
            for i in attnIdx {
                let vA = armA[i].state
                eval(vA)
                let bA = origA[i]!
                gateHistA = gateHistA
                    && sum(vA[0][.ellipsis, ..<position, 0...] .== bA[0]).item(Int.self) == bA[0].size
                let vA1 = armA1[i].state
                eval(vA1)
                let bA1 = origA1[i]![0]
                gateHistA1 = gateHistA1
                    && sum(vA1[0][.ellipsis, ..<position1, 0...] .== bA1).item(Int.self) == bA1.size
                let sharedA = vA[0][.ellipsis, ..<position, 0...]
                let sharedA1 = vA1[0][.ellipsis, ..<position, 0...]
                gateLineagePost = gateLineagePost
                    && sum(sharedA1 .== sharedA).item(Int.self) == sharedA.size
            }
            // 辅门:attention offset +N(A/B 臂 = position+N;A1/B1 臂 = position1+N)
            var gateOffsets = true
            for arm in [armA, armB, aC, bC] {
                for i in attnIdx {
                    gateOffsets = gateOffsets && (arm[i].offset == position + N)
                }
            }
            for arm in [armA1, armB1, a1C, b1C] {
                for i in attnIdx {
                    gateOffsets = gateOffsets && (arm[i].offset == position1 + N)
                }
            }
            var gateParent = true
            for (i, cache) in cacheP.enumerated() where cache is KVCacheSimple {
                let v = cache.state
                eval(v)
                let b = parentBefore[i]!
                gateParent = gateParent && v[0].shape == b[0].shape
                    && sum(v[0] .== b[0]).item(Int.self) == b[0].size
            }
            report("① sibling isolation(轨迹级): \(gateSibPost ? "✓" : "✗") | ② lineage(continuation 后:历史不可变 A \(gateHistA ? "✓" : "✗") A1 \(gateHistA1 ? "✓" : "✗");A1 仍含 A 前缀 \(gateLineagePost ? "✓" : "✗"))")
            report("辅门:attention offset +\(N)(八臂) \(gateOffsets ? "✓" : "✗") | 父代完好: \(gateParent ? "✓" : "✗")")
            #expect(gateSibPost)
            #expect(gateHistA)
            #expect(gateHistA1)
            #expect(gateLineagePost)
            #expect(gateOffsets)
            #expect(gateParent)

            let allPass = gatePreA && gatePreB && gatePreA1 && gatePreB1 && gateLineagePre
                && gateStruct && gateGDNOffset
                && gateContA && gateContB && gateContA1 && gateContB1
                && gateSibPost && gateHistA && gateHistA1 && gateLineagePost
                && gateOffsets && gateParent
            report(allPass
                ? "*** M3-C:多节点树各自独立 residency/continuation + 三层 invariant — PASS(段表路径,本配置;topology coincidence 在此配置下不成立)***"
                : "*** M3-C:FAIL(该配置下;不反推 Track B / M1 / M2 / M3 系)***")
        }
    }
}
