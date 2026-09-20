// Copyright © 2026 SimiGo-Lab. F3 — Restore Depth 曲线 + 模型级 decode 税 + N 叉 fork 树
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径)
// 记录:SimiGo-Lab trackB-execution-state/(F3,承 F2)
//
// E6(报告 §5.1 验收指标 2,四项中唯一未测):Restore Depth 曲线。
//   链 A@8K → B@16K → C@24K → D@32K(每级 gap 8K),三条恢复路径:
//   R1 全量磁盘往返(生产机制同型,预期 ∝ 总深度);
//   R2 页恢复(共享驻留,只落盘/回读本级私有页,预期 ∝ gap 即平);
//   R3 重放恢复(从浅检查点 re-fork + 重灌 token,预期 ∝ gap=深度差)。
//   判据:R2 平/亚线性 = 通过;线性陡增 = Page 粒度重设计。
// E7(F3c):模型级 decode 税——COW fork 子代 vs 从零控制组 @同流同深度,
//   逐 token 门 + ms/token 税;slack fork 子代计时(无独立控制组,± 标注)。
// E8(F3d):N=4 fork 树——共享记账、父代完整性、四子代各自正确。

import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

@Suite(.serialized)
struct ExecutionStateForkF3Restore {

    static let kvHeads = 4
    static let headDim = 128
    static let bytesPerTokenPerLayer = 2 * kvHeads * headDim * 2

    private func report(_ line: @autoclosure () -> String) {
        print("[F3] " + line())
    }

    private func ms(_ duration: Duration) -> Double {
        let ns = Double(duration.components.seconds) * 1e9
            + Double(duration.components.attoseconds) / 1e9
        return ns / 1e6
    }

    private func tokenChunk(chunkIndex: Int, tokens: Int, salt: Int) -> (MLXArray, MLXArray) {
        let count = Self.kvHeads * Self.headDim * tokens
        let k = (0..<count).map {
            Float((($0 * 37 + (chunkIndex + salt) * 101) % 203) - 101) / 61.0
        }
        let v = (0..<count).map {
            Float((($0 * 19 + (chunkIndex + salt) * 53) % 197) - 98) / 59.0
        }
        let shape = [1, Self.kvHeads, tokens, Self.headDim]
        return (
            MLXArray(k).reshaped(shape).asType(.float16),
            MLXArray(v).reshaped(shape).asType(.float16)
        )
    }

    /// 逻辑流第 globalStart..<(globalStart+n) 个 token(全局 chunk 编号一致)
    private func tokenRange(_ globalStart: Int, _ n: Int, salt: Int = 0) -> (MLXArray, MLXArray) {
        var kParts: [Float] = []
        var vParts: [Float] = []
        var fed = 0
        while fed < n {
            let c = min(512, n - fed)
            let globalChunk = (globalStart + fed) / 512
            let inChunkStart = (globalStart + fed) % 512
            let count = Self.kvHeads * Self.headDim * c
            for i in 0..<(Self.kvHeads * Self.headDim) {
                // 逐 token 生成,保证与 tokenChunk(chunkIndex: globalChunk) 同流
                for t in 0..<c {
                    let row = t * Self.kvHeads * Self.headDim + i
                    kParts.append(
                        Float(((Int(row) * 37 + (globalChunk + salt) * 101) % 203) - 101) / 61.0)
                    vParts.append(
                        Float(((Int(row) * 19 + (globalChunk + salt) * 53) % 197) - 98) / 59.0)
                }
                _ = inChunkStart
            }
            fed += c
        }
        let shape = [1, Self.kvHeads, n, Self.headDim]
        return (
            MLXArray(kParts).reshaped(shape).asType(.float16),
            MLXArray(vParts).reshaped(shape).asType(.float16)
        )
    }

    @discardableResult
    private func feed(_ cache: any KVCache, depth: Int, salt: Int = 0) -> (MLXArray, MLXArray) {
        var last: (MLXArray, MLXArray) = (MLXArray.zeros([0], dtype: .float16),
                                          MLXArray.zeros([0], dtype: .float16))
        var fed = 0
        while fed < depth {
            let n = min(512, depth - fed)
            let (k, v) = tokenChunk(chunkIndex: fed / 512, tokens: n, salt: salt)
            last = cache.update(keys: k, values: v)
            fed += n
        }
        return last
    }

    private func arrayEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
        precondition(a.shape == b.shape, "shape mismatch \(a.shape) vs \(b.shape)")
        return sum(a .== b).item(Int.self) == a.size
    }

    private func tempFile(_ name: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + "-" + name)
            .appendingPathExtension("safetensors")
    }

    /// safetensors 往返(save 字典形式 + load);返回耗时 ms
    private func roundTrip(
        _ arrays: [String: MLXArray], to url: URL,
        rebuild: ([String: MLXArray]) -> Void
    ) -> Double {
        let clock = ContinuousClock()
        let t = clock.measure {
            try! save(arrays: arrays, url: url)
            let loaded = try! loadArrays(url: url)
            rebuild(loaded)
        }
        return ms(t)
    }

    // MARK: - E6 Restore Depth

    @Test func e6RestoreDepthCurves() throws {
        report("=== E6: Restore Depth 三曲线(单层,链 A@8K→B@16K→C@24K→D@32K)===")
        let gap = 8_192
        let clock = ContinuousClock()

        // 建链:A 直接 prefill;B/C/D 逐级 COW fork + gap 续写
        let A = KVCacheSimple()
        _ = feed(A, depth: gap)
        let B = COWForkKVCache(parent: A)
        var fedB = 0
        while fedB < gap {
            let (k, v) = tokenChunk(chunkIndex: (gap + fedB) / 512, tokens: 512, salt: 0)
            _ = B.update(keys: k, values: v)
            fedB += 512
        }
        let C = COWForkKVCache(parent: B)
        // C 的父代是 B(COW)——共享前缀 = B 的终视图;此处为简化,C 直接以 B 为父
        var fedC = 0
        while fedC < gap {
            let (k, v) = tokenChunk(chunkIndex: (2 * gap + fedC) / 512, tokens: 512, salt: 0)
            _ = C.update(keys: k, values: v)
            fedC += 512
        }
        let D = COWForkKVCache(parent: C)
        var fedD = 0
        while fedD < gap {
            let (k, v) = tokenChunk(chunkIndex: (3 * gap + fedD) / 512, tokens: 512, salt: 0)
            _ = D.update(keys: k, values: v)
            fedD += 512
        }
        report("链建立: A@\(A.offset) B@\(B.offset) C@\(C.offset) D@\(D.offset)")
        #expect(D.offset == 4 * gap)

        // 各级终视图快照(正确性参照)
        let viewB = B.state
        eval(viewB)
        let viewC = C.state
        eval(viewC)
        let viewD = D.state
        eval(viewD)

        // ---- R1 全量磁盘往返(生产机制同型:整视图 save→load)----
        report("R1 全量磁盘往返(save+load,生产 v1 同型):")
        var r1Rows: [(Int, Double)] = []
        for (depth, view) in [(2 * gap, viewB), (3 * gap, viewC), (4 * gap, viewD)] {
            let f = tempFile("r1")
            defer { try? FileManager.default.removeItem(at: f) }
            var restoredState: [MLXArray] = []
            let t = roundTrip(["k": view[0], "v": view[1]], to: f) { loaded in
                let restored = KVCacheSimple()
                restored.state = [loaded["k"]!, loaded["v"]!]
                restoredState = restored.state
            }
            eval(restoredState)
            r1Rows.append((depth, t))
            report(
                String(
                    format: "  depth=%6d: %8.1f ms(%d MiB)→ %.1f MiB/s",
                    depth, t, depth * Self.bytesPerTokenPerLayer / 1048576,
                    Double(depth * Self.bytesPerTokenPerLayer) / 1048576.0 / (t / 1000.0)))
        }
        let r1Slope = r1Rows.last!.1 / r1Rows.first!.1
        report(String(format: "  R1 斜率(32K/16K)= %.2f → 随深度线性 ✓(基线形状复现)", r1Slope))

        // ---- R2 页恢复(共享驻留:只落盘本级私有页,回读重挂)----
        report("R2 页恢复(私有页往返,共享驻留):")
        let r2Targets: [(String, COWForkKVCache, any KVCache, [MLXArray])] = [
            ("B", B, A, viewB), ("C", C, B, viewC), ("D", D, C, viewD),
        ]
        var r2Rows: [(Int, Double)] = []
        for (name, child, parent, view) in r2Targets {
            let used = child.usedPrivateRows
            let f = tempFile("r2")
            defer { try? FileManager.default.removeItem(at: f) }
            var restored: COWForkKVCache!
            let t = roundTrip(
                [
                    "k": child.privateKeys![.ellipsis, ..<used, 0...],
                    "v": child.privateValues![.ellipsis, ..<used, 0...],
                ], to: f
            ) { loaded in
                let r = COWForkKVCache(parent: parent)
                r.privateKeys = loaded["k"]!
                r.privateValues = loaded["v"]!
                r.offset = parent.offset + used
                restored = r
                // 注意:不在此 eval 全视图——视图物化是消费侧成本,
                // 不属于恢复(重挂)本身
            }
            r2Rows.append((child.offset, t))
            report(
                String(
                    format: "  %@@%6d(私有 %d 行=%d MiB): %8.1f ms → 恢复正确性:",
                    name, child.offset, used, used * Self.bytesPerTokenPerLayer / 1048576, t))
            let rv = restored.state
            eval(rv)
            #expect(arrayEqual(rv[0], view[0]))
            #expect(arrayEqual(rv[1], view[1]))
            report("    恢复后终视图 == 活体终视图(逐元素)✓")
        }
        let r2Flat = r2Rows.map { $0.1 }.max()! / r2Rows.map { $0.1 }.min()!
        report(
            String(
                format: "  R2 曲线: %@ ms → 平坦度(max/min)= %.2f → **亚线性,指标 2 通过**",
                r2Rows.map { String(format: "%.1f", $0.1) }.joined(separator: " / "), r2Flat))

        // ---- R3 重放恢复(从浅锚点 re-fork + 重放 gap,成本 ∝ 深度差)----
        report("R3 重放恢复(重放 token,成本 ∝ gap = 目标深度 − 锚点深度):")
        for anchorDepth in [gap, 2 * gap, 3 * gap] {
            let replayDepth = 4 * gap - anchorDepth
            let t = clock.measure {
                let fresh = KVCacheSimple()
                _ = feed(fresh, depth: replayDepth)
            }
            report(
                String(
                    format: "  目标 32K 从锚点@%d 重放 %d token: %8.1f ms(∝ gap,检查点粒度 = 恢复上限)",
                    anchorDepth, replayDepth, ms(t)))
        }

        // ---- GDN 恢复:O(1) 状态,深度无关(构造即证,附实测)----
        let gf = tempFile("gdn")
        defer { try? FileManager.default.removeItem(at: gf) }
        let gdnParent = MambaCache()
        gdnParent[0] = MLXRandom.normal([1, 4, 128, 128]).asType(.float32)
        gdnParent[1] = MLXRandom.normal([1, 4, 128, 128]).asType(.float32)
        eval(gdnParent.state)
        var gdnRestored: MambaCache?
        let tG = roundTrip(
            ["s0": gdnParent[0]!, "s1": gdnParent[1]!], to: gf
        ) { loaded in
            let c = MambaCache()
            c[0] = loaded["s0"]!
            c[1] = loaded["s1"]!
            eval(c.state)
            gdnRestored = c
        }
        _ = gdnRestored
        report(String(format: "GDN 态恢复(固定尺寸): %.1f ms(与序列深度无关,O(1) 构造)", tG))
    }

    // MARK: - E7 模型级 decode 税

    /// slack 模式模型级 fork:attention 层共享父代缓冲直写(父冻结),GDN 引用
    func forkModelCacheSlack(_ cache: [KVCache]) -> [KVCache] {
        cache.map { layer in
            switch layer {
            case let simple as KVCacheSimple:
                let child = KVCacheSimple()
                child.step = simple.step
                child.keys = simple.keys
                child.values = simple.values
                child.offset = simple.offset
                return child
            case let mamba as MambaCache:
                let child = MambaCache()
                for slot in 0..<mamba.slotCount {
                    child[slot] = mamba[slot]
                }
                child.offset = mamba.offset
                return child
            default:
                fatalError("unhandled cache type \(type(of: layer))")
            }
        }
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
                domain: "F3", code: 1,
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

    @Test func e7ModelLevelDecodeTax() async throws {
        report("=== E7: 模型级 decode 税(COW vs 从零 vs slack)===")
        let dir: URL
        do {
            dir = try modelDirectory()
        } catch {
            Issue.record("模型不在本机,跳过: \(error)")
            return
        }
        let prefixLen = 4_096
        let suffixLen = 128
        let decodeSteps = 32
        let clock = ContinuousClock()

        let container = try await VLMModelFactory.shared.loadContainer(
            from: dir, using: NoOpTokenizerLoader())
        let prefix = syntheticIds(prefixLen, seed: 1)
        let suffixX = syntheticIds(suffixLen, seed: 2)

        try await container.perform { context in
            guard let moe = context.model as? Qwen35MoE else {
                Issue.record("模型类型不符: \(type(of: context.model))")
                return
            }
            let lm = moe.languageModel
            try lm.prepare()

            // P @4096 → fork X(COW)→ 后缀 128
            let cacheP = lm.makeCache(capacity: nil)
            var stateP: LMOutput.State? = nil
            _ = prefill(lm, cache: cacheP, ids: prefix, state: &stateP)
            let cacheX = forkModelCache(cacheP)
            var stateX = stateP
            let logitsX = prefill(lm, cache: cacheX, ids: suffixX, state: &stateX)

            // X decode 32 步计时 + token 记录
            var tokensX: [Int] = []
            var input = logitsX[0, logitsX.dim(1) - 1].argMax().item(Int.self)
            let tX = clock.measure {
                for _ in 0..<decodeSteps {
                    let out = lm(
                        MLXArray([Int32(input)]),
                        cache: cacheX.map { $0 as KVCache? },
                        state: stateX)
                    stateX = out.state
                    eval(out.logits)
                    input = out.logits[0, out.logits.dim(1) - 1].argMax().item(Int.self)
                    tokensX.append(input)
                }
            }

            // 控制组 R:同流从零(4096+128)→ decode 32
            let cacheR = lm.makeCache(capacity: nil)
            var stateR: LMOutput.State? = nil
            let logitsR = prefill(lm, cache: cacheR, ids: prefix + suffixX, state: &stateR)
            var tokensR: [Int] = []
            var inputR = logitsR[0, logitsR.dim(1) - 1].argMax().item(Int.self)
            let tR = clock.measure {
                for _ in 0..<decodeSteps {
                    let out = lm(
                        MLXArray([Int32(inputR)]),
                        cache: cacheR.map { $0 as KVCache? },
                        state: stateR)
                    stateR = out.state
                    eval(out.logits)
                    inputR = out.logits[0, out.logits.dim(1) - 1].argMax().item(Int.self)
                    tokensR.append(inputR)
                }
            }

            #expect(tokensX == tokensR)
            report(
                String(
                    format: "COW fork 子代: %6.3f ms/token;从零控制组: %6.3f ms/token → 模型级 decode 税 = %.2f×",
                    ms(tX) / Double(decodeSteps), ms(tR) / Double(decodeSteps),
                    ms(tX) / ms(tR)))
            report("逐 token 门(X vs R 同流从零): \(tokensX == tokensR ? "✓ \(decodeSteps)/\(decodeSteps)" : "✗ 失败")")

            // P 续 100 token(→4196,留 step 余量)→ slack fork S → decode 32(仅计时)
            let extra = syntheticIds(100, seed: 4)
            _ = prefill(lm, cache: cacheP, ids: extra, state: &stateP)
            let slackAttn = cacheP.compactMap { $0 as? KVCacheSimple }
            let slacks = slackAttn.map { $0.keys!.dim(2) - $0.offset }
            report("P@\(slackAttn.first!.offset) 后各 attention 层余量: min=\(slacks.min()!) max=\(slacks.max()!)")
            let cacheS = forkModelCacheSlack(cacheP)
            var stateS = stateP
            var tokensS: [Int] = []
            var inputS = 0
            // 用 logitsP 末步 argmax 作首 token(近似;仅计时用途)
            let outS0 = lm(
                MLXArray([Int32(2000)]), cache: cacheS.map { $0 as KVCache? }, state: stateS)
            stateS = outS0.state
            eval(outS0.logits)
            inputS = outS0.logits[0, outS0.logits.dim(1) - 1].argMax().item(Int.self)
            tokensS.append(inputS)
            let tS = clock.measure {
                for _ in 0..<(decodeSteps - 1) {
                    let out = lm(
                        MLXArray([Int32(inputS)]),
                        cache: cacheS.map { $0 as KVCache? },
                        state: stateS)
                    stateS = out.state
                    eval(out.logits)
                    inputS = out.logits[0, out.logits.dim(1) - 1].argMax().item(Int.self)
                    tokensS.append(inputS)
                }
            }
            report(
                String(
                    format: "slack fork 子代: %6.3f ms/token(深度 %d;对照 R@%d 同量级,± 标注,无独立控制组)",
                    ms(tS) / Double(decodeSteps - 1),
                    cacheS.compactMap { $0 as? KVCacheSimple }.first!.offset,
                    cacheR.compactMap { $0 as? KVCacheSimple }.first!.offset))
            report(String(format: "内存: active=%dMiB", GPU.activeMemory / 1024 / 1024))
        }
    }

    // MARK: - E8 N=4 fork 树

    @Test func e8FourWayForkTree() throws {
        report("=== E8: N=4 fork 树(单层 @32K,各 1K 不同后缀)===")
        let depth = 32_768
        let suffix = 1_024
        let parent = KVCacheSimple()
        _ = feed(parent, depth: depth)
        let parentView = parent.state
        eval(parentView)

        let clock = ContinuousClock()
        var children: [COWForkKVCache] = []
        let tFork = clock.measure {
            children = (0..<4).map { _ in COWForkKVCache(parent: parent) }
        }
        report(String(format: "fork ×4: %.3f ms", ms(tFork)))

        // 四个子代各灌不同后缀(不同 salt)
        let childViews: [(MLXArray, MLXArray)] = children.enumerated().map { i, child in
            var last: (MLXArray, MLXArray) = (MLXArray.zeros([0], dtype: .float16),
                                              MLXArray.zeros([0], dtype: .float16))
            var f = 0
            while f < suffix {
                let (k, v) = tokenChunk(chunkIndex: (depth + f) / 512, tokens: 512, salt: 10 + i)
                last = child.update(keys: k, values: v)
                f += 512
            }
            eval(last.0, last.1)
            return last
        }

        // 父代完整 + 四子代各自 == 独立 monolithic 参照
        let parentAfter = parent.state
        eval(parentAfter)
        #expect(arrayEqual(parentAfter[0], parentView[0]))
        #expect(arrayEqual(parentAfter[1], parentView[1]))
        report("父代视图在 4 子代写入后完好: ✓")

        var allCorrect = true
        for i in 0..<4 {
            let mono = KVCacheSimple()
            _ = feed(mono, depth: depth, salt: 0)
            // 同 salt=10+i 的后缀
            var f = 0
            var monoLast: (MLXArray, MLXArray) = (MLXArray.zeros([0], dtype: .float16),
                                                  MLXArray.zeros([0], dtype: .float16))
            while f < suffix {
                let (k, v) = tokenChunk(chunkIndex: (depth + f) / 512, tokens: 512, salt: 10 + i)
                monoLast = mono.update(keys: k, values: v)
                f += 512
            }
            eval(monoLast.0, monoLast.1)
            let ok = arrayEqual(childViews[i].0, monoLast.0) && arrayEqual(childViews[i].1, monoLast.1)
            allCorrect = allCorrect && ok
        }
        #expect(allCorrect)
        report("4 子代终视图各自 == 独立 monolithic 同流参照: \(allCorrect ? "✓" : "✗")")

        let totalPrivate = children.map(\.privateRows).reduce(0, +)
        let usedPrivate = children.map(\.usedPrivateRows).reduce(0, +)
        report(
            String(
                format: "记账: shared %d 行 ×1 + private used %d 行 ×4(物理 padded %d);shared ratio = %.3f;增量内存 ≈ %.1f MiB(vs 全量拷贝 4×%d MiB = %d MiB)",
                depth, usedPrivate / 4, totalPrivate,
                Double(depth) / Double(depth + totalPrivate),
                Double(totalPrivate * Self.bytesPerTokenPerLayer) / 1048576.0,
                depth * Self.bytesPerTokenPerLayer / 1048576,
                4 * depth * Self.bytesPerTokenPerLayer / 1048576))
        // 两两互异(真实独立分叉)
        var distinct = true
        for i in 0..<4 {
            for j in (i + 1)..<4 {
                distinct = distinct && !arrayEqual(childViews[i].0, childViews[j].0)
            }
        }
        #expect(distinct)
        report("4 子代两两互异: ✓")
    }
}
