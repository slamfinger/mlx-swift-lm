// Copyright © 2026 SimiGo-Lab. Track E(命名待定)Residency M3-A —
// Paired Representation Continuation(连续表示 vs 段表表示,成对对照)
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径;
// 承接 M3 FINAL PASS 终判;本实验独立成对,不修改已 PASS 的 M3)
// 记录:SimiGo-Lab trackE-residency/(M3 终判块:M3-A 开题)
//
// 攻击的架构分叉(M3 终判定序):"段表只是当前实现碰巧能跑" vs
// "residency continuation 本身不依赖表示形态"。设计:同一父代三臂——
//
//   P @4096(驻留父代)
//     ├── ctrl      = fork → 后缀512 → decode8   驻留全程(control)
//     ├── treatSeg  = 同上 → 私有页外化 → 重挂回驻留父代(**段表形态**)
//     └── treatCont = 同上 → 全视图外化 → 重建为连续表示(**物化形态**)
//
//   可证伪前置门:两 treat 外化前均 ≡ ctrl(逐元素 + next-input)。
//   主门×2:Continuation(arm) == Continuation(ctrl),N=96 逐 token。
//
// 措辞纪律:两臂均与 control 全等 ⇒ 在**本配置**下,continuation
// 对"外化后重挂为段表"与"外化后物化为连续"两条重表示路径**同时**
// 成立——这是表示无关性的 config 级证据,不是普遍结论;不构成
// "residency architecture 应采用 X"的架构结论。

import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

@Suite(.serialized)
struct ResidencyM3AContinuationPair {

    private func report(_ line: @autoclosure () -> String) {
        print("[ResidencyM3A] " + line())
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
                domain: "ResidencyM3A", code: 1,
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

    @Test func m3aPairedRepresentationContinuation() async throws {
        report("=== M3-A:成对表示 continuation 对照(段表重挂 vs 物化连续,N=96)===")
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
            let base = self.syntheticIds(4_096, seed: 111)
            let suffix = self.syntheticIds(512, seed: 112)
            let decodeN = 8
            let clock = ContinuousClock()

            // ---- 1. 父代 + 三臂(ctrl / treatSeg / treatCont,同构构建)
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
            var (ctrl, stateCtrl, nextCtrl) = buildChild()
            var (treatSeg, stateSeg, nextSeg) = buildChild()
            var (treatCont, stateCont, nextCont) = buildChild()

            let attnIdx: [Int] = ctrl.enumerated().compactMap {
                ($1 is COWForkKVCache || $1 is KVCacheSimple) ? $0 : nil
            }
            let gdnIdx: [Int] = ctrl.enumerated().compactMap {
                $1 is MambaCache ? $0 : nil
            }
            let layerCount = ctrl.count
            let position = ctrl[attnIdx.first!].offset
            let gdnSlotCounts = Dictionary(uniqueKeysWithValues: gdnIdx.map {
                ($0, (ctrl[$0] as! MambaCache).slotCount)
            })
            let used = position - 4096
            report("三臂就绪 @\(position),nextInput=\(nextCtrl)(seg=\(nextSeg),cont=\(nextCont)),私有 used=\(used) 行")

            // ---- 2. 可证伪前置门:两 treat 外化前均 ≡ ctrl
            func precheck(_ t: [KVCache], _ st: LMOutput.State?, _ next: Int) -> Bool {
                var ok = (next == nextCtrl)
                for i in attnIdx {
                    let vt = t[i].state
                    let vc = ctrl[i].state
                    eval(vt, vc)
                    ok = ok && vt[0].shape == vc[0].shape
                        && sum(vt[0] .== vc[0]).item(Int.self) == vc[0].size
                        && vt[1].shape == vc[1].shape
                        && sum(vt[1] .== vc[1]).item(Int.self) == vc[1].size
                }
                for i in gdnIdx {
                    let mt = t[i] as! MambaCache
                    let mc = ctrl[i] as! MambaCache
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
            let gatePreSeg = precheck(treatSeg, stateSeg, nextSeg)
            let gatePreCont = precheck(treatCont, stateCont, nextCont)
            report("前置门:seg 臂 ≡ ctrl \(gatePreSeg ? "✓" : "✗") | cont 臂 ≡ ctrl \(gatePreCont ? "✓" : "✗")")
            #expect(gatePreSeg)
            #expect(gatePreCont)

            // 父代完整性参照
            var parentBefore: [Int: [MLXArray]] = [:]
            for (i, cache) in cacheP.enumerated() where cache is KVCacheSimple {
                let v = cache.state
                eval(v)
                parentBefore[i] = v
            }

            // ---- 3. 两臂外化
            // seg 臂:私有页(m2b/M3 形态)
            func fileFor(_ payload: [String: MLXArray], _ tag: String) throws -> URL {
                let f = FileManager.default.temporaryDirectory
                    .appendingPathComponent(UUID().uuidString + "-\(tag).safetensors")
                try save(arrays: payload, url: f)
                return f
            }
            var segPayload: [String: MLXArray] = [:]
            for i in attnIdx {
                let cow = treatSeg[i] as! COWForkKVCache
                segPayload["p\(i).k"] = cow.privateKeys![.ellipsis, ..<used, 0...]
                segPayload["p\(i).v"] = cow.privateValues![.ellipsis, ..<used, 0...]
                eval(segPayload["p\(i).k"]!, segPayload["p\(i).v"]!)
            }
            for i in gdnIdx {
                let m = treatSeg[i] as! MambaCache
                for s in 0..<m.slotCount {
                    let a = m[s]!
                    eval(a)
                    segPayload["g\(i).s\(s)"] = a
                }
            }
            var segStateKeys = 0
            if let st = stateSeg, let dict = try? st.serializedArrays() {
                for (k, v) in dict {
                    eval(v)
                    segPayload["st.\(k)"] = v
                }
                segStateKeys = dict.count
            }
            segPayload["meta.nextInput"] = MLXArray([Int32(nextSeg)])
            segPayload["meta.used"] = MLXArray([Int32(used)])
            let segFile = try fileFor(segPayload, "m3a-seg")
            defer { try? FileManager.default.removeItem(at: segFile) }

            // cont 臂:全视图(M1 形态——物化路径所需)
            var contPayload: [String: MLXArray] = [:]
            for i in attnIdx {
                let v = treatCont[i].state
                eval(v)
                contPayload["a\(i).k"] = v[0]
                contPayload["a\(i).v"] = v[1]
            }
            for i in gdnIdx {
                let m = treatCont[i] as! MambaCache
                for s in 0..<m.slotCount {
                    let a = m[s]!
                    eval(a)
                    contPayload["g\(i).s\(s)"] = a
                }
            }
            var contStateKeys = 0
            if let st = stateCont, let dict = try? st.serializedArrays() {
                for (k, v) in dict {
                    eval(v)
                    contPayload["st.\(k)"] = v
                }
                contStateKeys = dict.count
            }
            contPayload["meta.nextInput"] = MLXArray([Int32(nextCont)])
            contPayload["meta.position"] = MLXArray([Int32(position)])
            let contFile = try fileFor(contPayload, "m3a-cont")
            defer { try? FileManager.default.removeItem(at: contFile) }
            let segBytes = (try? Data(contentsOf: segFile))?.count ?? 0
            let contBytes = (try? Data(contentsOf: contFile))?.count ?? 0

            treatSeg = []; treatCont = []
            stateSeg = nil; stateCont = nil
            GPU.clearCache()
            report("两臂外化:seg 私有页 \(segBytes / 1024) KiB | cont 全视图 \(contBytes / 1024) KiB → 释放(ctrl 驻留)")

            // ---- 4. 两臂重建(段表重挂 vs 物化连续)
            let segLoaded = try loadArrays(url: segFile)
            let usedAfter = Int(segLoaded["meta.used"]!.item(Int32.self))
            var seg = [KVCache?](repeating: nil, count: layerCount)
            for i in attnIdx {
                let cow = COWForkKVCache(parent: cacheP[i])
                cow.privateKeys = segLoaded["p\(i).k"]!
                cow.privateValues = segLoaded["p\(i).v"]!
                cow.offset = cacheP[i].offset + usedAfter
                seg[i] = cow
            }
            for i in gdnIdx {
                let m = MambaCache()
                for s in 0..<(gdnSlotCounts[i] ?? 0) { m[s] = segLoaded["g\(i).s\(s)"]! }
                seg[i] = m
            }
            var stateSegAfter: LMOutput.State? = nil
            let segStKeys = segLoaded.keys.filter { $0.hasPrefix("st.") }.sorted()
            if !segStKeys.isEmpty {
                var dict: [String: MLXArray] = [:]
                for k in segStKeys { dict[String(k.dropFirst(3))] = segLoaded[k]! }
                stateSegAfter = LMOutput.State(serializedArrays: dict)
            }
            let segArm = seg.map { $0! }

            let contLoaded = try loadArrays(url: contFile)
            var cont = [KVCache?](repeating: nil, count: layerCount)
            for i in attnIdx {
                let c = KVCacheSimple()
                c.state = [contLoaded["a\(i).k"]!, contLoaded["a\(i).v"]!]
                cont[i] = c
            }
            for i in gdnIdx {
                let m = MambaCache()
                for s in 0..<(gdnSlotCounts[i] ?? 0) { m[s] = contLoaded["g\(i).s\(s)"]! }
                m.offset = position
                cont[i] = m
            }
            var stateContAfter: LMOutput.State? = nil
            let contStKeys = contLoaded.keys.filter { $0.hasPrefix("st.") }.sorted()
            if !contStKeys.isEmpty {
                var dict: [String: MLXArray] = [:]
                for k in contStKeys { dict[String(k.dropFirst(3))] = contLoaded[k]! }
                stateContAfter = LMOutput.State(serializedArrays: dict)
            }
            let contArm = cont.map { $0! }

            // 表示核查:seg 臂仍为段表;cont 臂全部为连续(KVCacheSimple)
            var gateRepSeg = true
            for i in attnIdx {
                guard let cow = segArm[i] as? COWForkKVCache else { gateRepSeg = false; continue }
                gateRepSeg = gateRepSeg && (cow.sharedRows == cacheP[i].offset) && (cow.offset == position)
            }
            var gateRepCont = true
            for i in attnIdx {
                guard contArm[i] is KVCacheSimple else { gateRepCont = false; continue }
                gateRepCont = gateRepCont && (contArm[i].offset == position)
            }
            report("表示核查:seg 臂仍为 COW 段表(未物化) \(gateRepSeg ? "✓" : "✗") | cont 臂全部物化为连续 \(gateRepCont ? "✓" : "✗")")

            // ---- 5. 主门×2:三侧各自继续 N 步
            var inputS = Int(segLoaded["meta.nextInput"]!.item(Int32.self))
            var inputC2 = Int(contLoaded["meta.nextInput"]!.item(Int32.self))
            var inputK = nextCtrl
            var tokensS: [Int] = [], tokensC2: [Int] = [], tokensK: [Int] = []
            let tRun = clock.measure {
                for _ in 0..<N {
                    inputS = self.step(lm, cache: segArm, state: &stateSegAfter, input: inputS)
                    inputC2 = self.step(lm, cache: contArm, state: &stateContAfter, input: inputC2)
                    inputK = self.step(lm, cache: ctrl, state: &stateCtrl, input: inputK)
                    tokensS.append(inputS); tokensC2.append(inputC2); tokensK.append(inputK)
                }
            }
            let gateSeg = (tokensS == tokensK)
            let gateCont = (tokensC2 == tokensK)
            report(String(format: "主门(seg 臂,段表重挂): %d/%d %@ | 主门(cont 臂,物化连续): %d/%d %@(%.1f s,描述性)",
                tokensS.count, N, gateSeg ? "✓ 全等" : "✗",
                tokensC2.count, N, gateCont ? "✓ 全等" : "✗", ms(tRun) / 1000.0))
            #expect(gateSeg)
            #expect(gateCont)

            // ---- 6. 辅门:offset 推进 + 父代完好
            var gateOffsets = true
            for i in attnIdx {
                gateOffsets = gateOffsets
                    && (segArm[i].offset == position + N)
                    && (contArm[i].offset == position + N)
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
            report("offset 三侧精确推进 +\(N): \(gateOffsets ? "✓" : "✗") | 父代完好: \(gateParent ? "✓" : "✗")")
            #expect(gateRepSeg)
            #expect(gateRepCont)
            #expect(gateOffsets)
            #expect(gateParent)

            let allPass = gatePreSeg && gatePreCont && gateSeg && gateCont
                && gateRepSeg && gateRepCont && gateOffsets && gateParent
            report(allPass
                ? "*** M3-A:两条重表示路径(段表重挂/物化连续)均沿 ctrl 的未来 96/96 继续 — PASS(本配置;表示无关性的 config 级证据,非普遍结论)***"
                : "*** M3-A:FAIL(该配置下;不反推 Track B / M1 / M2 / M3)***")
        }
    }
}
