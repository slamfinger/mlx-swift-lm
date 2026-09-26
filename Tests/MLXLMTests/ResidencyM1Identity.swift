// Copyright © 2026 SimiGo-Lab. Track E(命名待定)Residency M1 — Identity Gate
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径;
// 承接 Track B 冻结点 acbcb54,不回头修改 Track B)
// 记录:SimiGo-Lab trackE-residency/(M1 开题,2026-09-21)
//
// 研究问题(唯一):Execution State 离开当前物理驻留(Externalize)
// 再回来(Rehydrate)之后,ExecutionId 是否保持?
//
//   ExecutionId(before) == ExecutionId(after)
//
// Execution State = KV + Identity + Position + Next Input + Continuation
// (Track B 审计链收敛模型)。M1 的 identity 判定 = 状态接地的:
//   ① position:逐层 offset 相等;
//   ② next-input:**以 G2 迁移单元契约的形式**——input 作为单元自身
//     载荷走驻留边界(存入 payload、从重建工件读回),禁止与本地
//     变量自比(v2 的 G2 伪断言教训:变量自比恒真);
//   ③ attention KV 全视图逐元素相等(全部 attention 层);
//   ④ GDN(MambaCache)各槽逐元素相等(全部 GDN 层);
//   ⑤ LMOutput.State 序列化字典:键集合相等 + 逐键逐元素相等。
// 字节级 sha256 仅为描述性记录(safetensors 键序不保证稳定,不作门)。
// 边界纪律:本实验失败 ⇒ "该 Externalize/Rehydrate 机制不成立",
// 不反推 Track B Closure。计时仅描述性,性能归属 M4(交替对照)。
// 表述纪律:resident 态释放后仅存 identity 锚点快照(比较用,
// 非 continuation-capable),在文档中如实说明。

import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

@Suite(.serialized)
struct ResidencyM1Identity {

    private func report(_ line: @autoclosure () -> String) {
        print("[ResidencyM1] " + line())
    }

    private func ms(_ duration: Duration) -> Double {
        let ns = Double(duration.components.seconds) * 1e9
            + Double(duration.components.attoseconds) / 1e9
        return ns / 1e6
    }

    private func sha256Hex(_ url: URL) -> String {
        let data = (try? Data(contentsOf: url)) ?? Data()
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
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
                domain: "ResidencyM1", code: 1,
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
        return (ms(t), next)
    }

    @Test func m1IdentityAcrossResidencyBoundary() async throws {
        report("=== Residency M1:Externalize → Rehydrate 后 ExecutionId 是否保持 ===")
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

        try await container.perform { context in
            guard let moe = context.model as? Qwen35MoE else {
                Issue.record("模型类型不符")
                return
            }
            let lm = moe.languageModel
            try lm.prepare()

            // ---- 1. 构造 resident execution state(Track B 遗产形态):
            // P @4096(基座)→ fork(COW 段表 + GDN 引用)→ 后缀 512 →
            // decode 8 步(decode regime 的 GDN 态 + 真实 next-input 链)
            let prefix = syntheticIds(4_096, seed: 31)
            let suffix = syntheticIds(512, seed: 32)
            let decodeN = 8
            let clock = ContinuousClock()

            var cacheP = lm.makeCache(capacity: nil)
            var stateP: LMOutput.State? = nil
            _ = prefill(lm, cache: cacheP, ids: prefix, state: &stateP)

            var r = forkModelCache(cacheP)
            var stateR = stateP
            let logitsSuf = prefill(lm, cache: r, ids: suffix, state: &stateR)
            var input = logitsSuf[0, logitsSuf.dim(1) - 1].argMax().item(Int.self)
            for _ in 0..<decodeN {
                let (_, t) = self.step(lm, cache: r, state: &stateR, input: input)
                input = t
            }
            report("resident 态就绪(解码链末端),nextInput=\(input);position = 首 attention 层 offset,见下")

            // ---- 2. ExecutionId(before) 捕获 + Externalize
            var layerKinds: [String] = []
            var attnIdx: [Int] = []
            var gdnIdx: [Int] = []
            for (i, c) in r.enumerated() {
                switch c {
                case is COWForkKVCache:
                    layerKinds.append("cow"); attnIdx.append(i)
                case is KVCacheSimple:
                    layerKinds.append("simple"); attnIdx.append(i)
                case is MambaCache:
                    layerKinds.append("gdn"); gdnIdx.append(i)
                default:
                    fatalError("unhandled cache type \(type(of: c))")
                }
            }
            report("层构成:attention \(attnIdx.count)(COW 段表/连续)+ GDN \(gdnIdx.count)")
            // 混合架构:position 锚定首个 attention 层(GDN offset 语义不同;
            // closure 的 firstCowIdx 先例)
            let residentPosition = r[attnIdx.first!].offset
            report("resident position @\(residentPosition)")

            // identity 锚点快照(全视图,物化;比较基准)
            var kvBefore: [Int: [MLXArray]] = [:]
            for i in attnIdx {
                let view = r[i].state
                eval(view)
                kvBefore[i] = view
            }
            var gdnBefore: [Int: [MLXArray]] = [:]
            for i in gdnIdx {
                let m = r[i] as! MambaCache
                var slots: [MLXArray] = []
                for s in 0..<m.slotCount {
                    let a = m[s]!
                    eval(a)
                    slots.append(a)
                }
                gdnBefore[i] = slots
            }
            let offsetsBefore = r.map { $0.offset }
            var stateDictBefore: [String: MLXArray] = [:]
            var stateCaptured = false
            if let st = stateR {
                do {
                    stateDictBefore = try st.serializedArrays()
                    for a in stateDictBefore.values { eval(a) }
                    stateCaptured = true
                    report("LMOutput.State 捕获:\(stateDictBefore.count) 个数组(键序无关,逐键比较)")
                } catch {
                    report("LMOutput.State 含不可序列化值,如实入档:\(error.localizedDescription)")
                }
            } else {
                report("LMOutput.State = nil(该 resident 态的 GDN 态由 MambaCache 槽承载)")
            }

            // 迁移单元(G2 契约形式):input 与 position 随单元载荷走边界
            var payload: [String: MLXArray] = [:]
            for i in attnIdx {
                payload["a\(i).k"] = kvBefore[i]![0]
                payload["a\(i).v"] = kvBefore[i]![1]
            }
            for i in gdnIdx {
                let m = r[i] as! MambaCache
                for s in 0..<m.slotCount {
                    payload["g\(i).s\(s)"] = gdnBefore[i]![s]
                }
            }
            for (k, v) in stateDictBefore {
                payload["st.\(k)"] = v
            }
            payload["meta.nextInput"] = MLXArray([Int32(input)])
            payload["meta.position"] = MLXArray([Int32(residentPosition)])

            let file = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + "-residency-m1.safetensors")
            defer { try? FileManager.default.removeItem(at: file) }

            let memResident = GPU.activeMemory
            let tX = clock.measure {
                try! save(arrays: payload, url: file)
            }
            let externalizeMs = ms(tX)
            let shaFile = self.sha256Hex(file)
            let fileBytes = (try? Data(contentsOf: file))?.count ?? 0

            // 释放 resident(真实离开物理驻留;锚点快照与文件保留)
            r = []
            cacheP = []
            stateR = nil
            stateP = nil
            GPU.clearCache()
            let memReleased = GPU.activeMemory
            report(String(format: "Externalize: %d tensors → %d bytes(%.1f ms,描述性);sha256=%@…",
                payload.count, fileBytes, externalizeMs, String(shaFile.prefix(16))))
            report(String(format: "驻留释放:GPU active %d MiB → %d MiB(模型权重常驻,差值 ≈ KV/GDN 态)",
                memResident / 1048576, memReleased / 1048576))

            // ---- 3. Rehydrate(连续表示;reattach 先例:表示可变,identity 不可变)
            let loaded = try loadArrays(url: file)
            var rebuilt = [KVCache?](repeating: nil, count: layerKinds.count)
            for i in attnIdx {
                let c = KVCacheSimple()
                c.state = [loaded["a\(i).k"]!, loaded["a\(i).v"]!]  // setter 自 offset = 逻辑行数
                rebuilt[i] = c
            }
            for i in gdnIdx {
                let m = MambaCache()
                for s in 0..<(gdnBefore[i]?.count ?? 0) {
                    m[s] = loaded["g\(i).s\(s)"]!
                }
                m.offset = offsetsBefore[i]
                rebuilt[i] = m
            }
            var stateAfter: LMOutput.State? = nil
            if stateCaptured {
                var stDict: [String: MLXArray] = [:]
                for (k, v) in stateDictBefore {
                    stDict[k] = loaded["st.\(k)"]!
                }
                stateAfter = LMOutput.State(serializedArrays: stDict)
            }
            let cachesAfter = rebuilt.map { $0! }
            let rehydrateMs = ms(clock.measure {
                for c in cachesAfter {
                    let st = c.state
                    eval(st)
                }
            })
            let nextInputAfter = Int(loaded["meta.nextInput"]!.item(Int32.self))
            let positionAfter = Int(loaded["meta.position"]!.item(Int32.self))
            report(String(format: "Rehydrate: %d 层重建(物化 %.1f ms,描述性);单元元数据回读 nextInput=%d position=%d",
                cachesAfter.count, rehydrateMs, nextInputAfter, positionAfter))

            // ---- 4. ExecutionId(after) 捕获 + 门
            var gatePosition = positionAfter == residentPosition
            for (i, c) in cachesAfter.enumerated() {
                gatePosition = gatePosition && (c.offset == offsetsBefore[i])
                if attnIdx.contains(i) {
                    gatePosition = gatePosition && (offsetsBefore[i] == residentPosition)
                }
            }
            // 非自比:input 走寄存器路径,nextInputAfter 走"载荷→文件→
            // loadArrays→元数据"路径;mInput 类丢失/污染在此被拦
            let gateNextInput = (nextInputAfter == input)

            var gateKV = true
            for i in attnIdx {
                let view = cachesAfter[i].state
                eval(view)
                let b = kvBefore[i]!
                gateKV = gateKV
                    && view[0].shape == b[0].shape && sum(view[0] .== b[0]).item(Int.self) == b[0].size
                    && view[1].shape == b[1].shape && sum(view[1] .== b[1]).item(Int.self) == b[1].size
            }
            var gateGDN = true
            for i in gdnIdx {
                let m = cachesAfter[i] as! MambaCache
                for s in 0..<m.slotCount {
                    let a = m[s]!
                    eval(a)
                    let b = gdnBefore[i]![s]
                    gateGDN = gateGDN && a.shape == b.shape
                        && sum(a .== b).item(Int.self) == b.size
                }
            }
            var gateState = true
            if stateCaptured {
                var keysOK = true
                var stDictAfter: [String: MLXArray] = [:]
                if let st = stateAfter {
                    let re = try st.serializedArrays()
                    keysOK = Set(re.keys) == Set(stateDictBefore.keys)
                    for (k, v) in re { eval(v); stDictAfter[k] = v }
                } else {
                    keysOK = false
                }
                gateState = keysOK
                if keysOK {
                    for (k, b) in stateDictBefore {
                        let a = stDictAfter[k]!
                        gateState = gateState && a.shape == b.shape
                            && sum(a .== b).item(Int.self) == b.size
                    }
                }
            }

            let m1Pass = gatePosition && gateNextInput && gateKV && gateGDN && gateState
            report("① position(逐层 offset + 单元元数据): \(gatePosition ? "✓" : "✗")")
            report("② next-input(单元载荷回读,G2 契约形式): \(gateNextInput ? "✓" : "✗")")
            report("③ attention KV 全视图逐元素(\(attnIdx.count) 层): \(gateKV ? "✓" : "✗")")
            report("④ GDN 槽逐元素(\(gdnIdx.count) 层): \(gateGDN ? "✓" : "✗")")
            report("⑤ LMOutput.State: \(stateCaptured ? (gateState ? "✓" : "✗") : "n/a(未捕获,见上)")")
            #expect(gatePosition)
            #expect(gateNextInput)
            #expect(gateKV)
            #expect(gateGDN)
            if stateCaptured { #expect(gateState) }
            #expect(m1Pass)

            // ---- 5. 消费性冒烟步(诊断,非 M3 门):重建态必须可被模型消费
            var stateSmoke = stateAfter
            let (smokeMs, smokeNext) = self.step(lm, cache: cachesAfter, state: &stateSmoke, input: nextInputAfter)
            let smokeOffsetOK = cachesAfter[attnIdx.first!].offset == residentPosition + 1
            report(String(format: "冒烟步(诊断,非门):单步 decode 可运行 %.1f ms,offset +1 %@,token=\(smokeNext)",
                smokeMs, smokeOffsetOK ? "✓" : "✗"))
            #expect(smokeOffsetOK)

            report(m1Pass
                ? "*** Residency M1:ExecutionId 跨驻留边界保持 — PASS(语义门;性能归 M4)***"
                : "*** Residency M1:FAIL — 该 Externalize/Rehydrate 机制不成立(不反推 Track B)***")
        }
    }
}
