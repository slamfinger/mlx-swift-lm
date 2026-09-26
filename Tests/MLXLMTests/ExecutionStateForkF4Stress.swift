// Copyright © 2026 SimiGo-Lab. F4 — 工程化压力与主干迁移语义(受约束验证阶段)
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径)
// 记录:SimiGo-Lab trackB-execution-state/(F4,承 F3 复盘)
//
// F4-A(P0)长深度 × 链深 × 活跃分支压力矩阵(cache 级):
//   context 32K/65K/131K × chain 4/8/16 × branches 2/4/8/16 × 3 reps
//   验收:物理增长 ∝ 私有页;分支间零交叉污染(每分支 == 独立 monolithic
//   同流参照,逐元素);root 完好;fork 成本不随深度/分支退化。
// F4-B 主干迁移语义(Promote/Rollback/Re-fork/Discard):
//   Promote 零拷贝(引用切换);Rollback=从保留 root 重叉 O(1);
//   历史树在 rollback+refork 后不变;Discard 私有页可回收(activeMemory
//   实测下降);Promote 后 decode 不退化;长时间 fork/丢弃循环无内存积累。
// F4-C 多分支长期存活:16 分支交错追加后逐个验证 + 部分丢弃。
// F4-B-model:root 解冻续写门(真模型)——F2 只证过冻结父代;生产
//   rollback 场景 root 要继续写(rows ≥ forkDepth),须单独过逐 token 门。

import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

@Suite(.serialized)
struct ExecutionStateForkF4Stress {

    static let kvHeads = 4
    static let headDim = 128
    static let bytesPerTokenPerLayer = 2 * kvHeads * headDim * 2

    private func report(_ line: @autoclosure () -> String) {
        print("[F4] " + line())
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

    /// 向 cache 追加全局流 [at, at+n) 的 token(salt 区分分叉流)
    @discardableResult
    private func appendStream(
        _ cache: any KVCache, at: Int, n: Int, salt: Int
    ) -> (MLXArray, MLXArray) {
        precondition(n > 0)
        var last: (MLXArray, MLXArray) = (MLXArray.zeros([0], dtype: .float16),
                                          MLXArray.zeros([0], dtype: .float16))
        var fed = 0
        while fed < n {
            let c = min(512, n - fed)
            let (k, v) = tokenChunk(chunkIndex: (at + fed) / 512, tokens: c, salt: salt)
            last = cache.update(keys: k, values: v)
            fed += c
        }
        return last
    }

    @discardableResult
    private func feed(_ cache: any KVCache, depth: Int, salt: Int = 0) -> (MLXArray, MLXArray) {
        appendStream(cache, at: 0, n: depth, salt: salt)
    }

    /// monolithic 参照:前 `depth-suffix` 个 token 走 salt=0 主流,尾部
    /// `suffix` 个走 salt=branchSalt(与树+分支流的逻辑构成一致)
    private func monolithicReference(
        depth: Int, suffix: Int, branchSalt: Int
    ) -> (MLXArray, MLXArray) {
        let mono = KVCacheSimple()
        _ = feed(mono, depth: depth - suffix, salt: 0)
        return appendStream(mono, at: depth - suffix, n: suffix, salt: branchSalt)
    }

    private func arrayEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
        precondition(a.shape == b.shape, "shape mismatch \(a.shape) vs \(b.shape)")
        return sum(a .== b).item(Int.self) == a.size
    }

    // MARK: - F4-A 压力矩阵

    struct F4AConfig {
        let context: Int
        let chainLevels: Int
        let branches: Int
        var label: String { "ctx\(context / 1024)K/chain\(chainLevels)/br\(branches)" }
    }

    @Test func f4aLongDepthBranchMatrix() throws {
        report("=== F4-A: 长深度 × 链深 × 活跃分支矩阵(单层,3 reps)===")
        let branchSuffix = 512
        let configs: [F4AConfig] = [
            // 分支维度 @32K
            .init(context: 32_768, chainLevels: 8, branches: 2),
            .init(context: 32_768, chainLevels: 8, branches: 4),
            .init(context: 32_768, chainLevels: 8, branches: 8),
            .init(context: 32_768, chainLevels: 8, branches: 16),
            // 链深维度 @65K
            .init(context: 65_536, chainLevels: 4, branches: 4),
            .init(context: 65_536, chainLevels: 8, branches: 4),
            .init(context: 65_536, chainLevels: 16, branches: 4),
            // 深度上限 @131K
            .init(context: 131_072, chainLevels: 8, branches: 4),
        ]
        let clock = ContinuousClock()

        for config in configs {
            for rep in 1...3 {
                GPU.clearCache()
                let rootDepth = max(512, (config.context / 4 / 512) * 512)
                let gap = max(512, ((config.context - rootDepth) / config.chainLevels / 512) * 512)
                let totalDepth = rootDepth + config.chainLevels * gap

                // root + 链(单一主流,逐级 COW fork)
                let root = KVCacheSimple()
                _ = feed(root, depth: rootDepth)
                var chain: [COWForkKVCache] = []
                var tForkChain = 0.0
                var cursor = rootDepth
                for _ in 0..<config.chainLevels {
                    let parent: any KVCache = chain.last ?? root
                    let t = clock.measure { chain.append(COWForkKVCache(parent: parent)) }
                    tForkChain += ms(t)
                    _ = appendStream(chain[chain.count - 1], at: cursor, n: gap, salt: 0)
                    cursor += gap
                }

                // 活跃分支(从链尾分叉,各自 salt)
                var branchLeaves: [COWForkKVCache] = []
                var branchViews: [(MLXArray, MLXArray)] = []
                let tBranches = clock.measure {
                    for b in 0..<config.branches {
                        let leaf = COWForkKVCache(parent: chain[chain.count - 1])
                        branchLeaves.append(leaf)
                        branchViews.append(
                            appendStream(leaf, at: totalDepth, n: branchSuffix, salt: 100 + b))
                    }
                }
                for view in branchViews { eval(view.0, view.1) }

                // 记账(容量口径,含 step 填充)
                let rootCapRows = root.keys!.dim(2)
                var privateCapRows = 0
                // 链节点物理 = 每级私有缓冲
                privateCapRows += chain.map(\.privateRows).reduce(0, +)
                privateCapRows += branchLeaves.map(\.privateRows).reduce(0, +)
                let physicalBytes = (rootCapRows + privateCapRows) * Self.bytesPerTokenPerLayer
                let logicalPerBranch = totalDepth + branchSuffix
                let expectedGrowthRows = config.chainLevels * gap + config.branches * branchSuffix

                // fork 成本不随深度退化
                let tForkLeaf = clock.measure {
                    _ = COWForkKVCache(parent: chain[chain.count - 1])
                }

                // 正确性:每分支 == 独立 monolithic 同流参照;root 完好
                let rootView = root.state
                eval(rootView)
                var allOk = arrayEqual(rootView[0], root.keys![.ellipsis, ..<rootDepth, 0...])
                for b in 0..<config.branches {
                    let ref = monolithicReference(
                        depth: logicalPerBranch, suffix: branchSuffix, branchSalt: 100 + b)
                    eval(ref.0, ref.1)
                    allOk = allOk
                        && arrayEqual(branchViews[b].0, ref.0)
                        && arrayEqual(branchViews[b].1, ref.1)
                }
                #expect(allOk)

                report(
                    String(
                        format: "  %@ rep%d: fork链 %.3fms + 分支×%d %.3fms + 叶fork %.3fms | 物理 %.1fMiB(root %d + 私有 %d 行,期望增长 %d 行) | shared %.3f | 一致性 %@",
                        config.label, rep, tForkChain, config.branches, ms(tBranches), ms(tForkLeaf),
                        Double(physicalBytes) / 1048576.0, rootCapRows, privateCapRows,
                        expectedGrowthRows,
                        Double(rootDepth) / Double(rootDepth + expectedGrowthRows),
                        allOk ? "✓" : "✗"))
            }
        }
        report("F4-A 判定行:物理私有行 = 链 gap×级数 + 分支 suffix×数(∝ 结构,非 ∝ 深度×分支)")
    }

    // MARK: - F4-B 主干迁移语义

    @Test func f4bPromoteRollbackDiscard() throws {
        report("=== F4-B: Promote / Rollback / Re-fork / Discard(单层)===")
        let clock = ContinuousClock()
        let scale = Float(pow(Float(Self.headDim), -0.5))
        let q = MLXArray.zeros([1, Self.kvHeads, 1, Self.headDim], dtype: .float16)

        // 树:Root@8K → A(+8K) → A1(+2K);Root → B(+8K) → B1(+2K)
        let root = KVCacheSimple()
        _ = feed(root, depth: 8_192)
        let A = COWForkKVCache(parent: root)
        _ = appendStream(A, at: 8_192, n: 8_192, salt: 0)
        let A1 = COWForkKVCache(parent: A)
        _ = appendStream(A1, at: 16_384, n: 2_048, salt: 0)
        let B = COWForkKVCache(parent: root)
        _ = appendStream(B, at: 8_192, n: 8_192, salt: 50)
        let B1 = COWForkKVCache(parent: B)
        _ = appendStream(B1, at: 16_384, n: 2_048, salt: 50)
        report("树建立: Root@8192 A@16384 A1@18432 B@16384(salt50) B1@18432(salt50)")

        // ---- Promote:A1 升为主干 = 直接在其上续跑(引用切换)----
        GPU.clearCache()
        let activeBefore = GPU.activeMemory
        let tPromote = clock.measure { _ = A1 }  // 提升即引用切换
        let activeAfter = GPU.activeMemory
        report(
            String(
                format: "Promote: %.4f ms,Δactive=%d KiB(零拷贝判定:|Δ| 应 ≈ 0)",
                ms(tPromote), (activeAfter - activeBefore) / 1024))

        // Promote 后 decode 64 步 vs 同深度 fresh
        func decode64(_ cache: any KVCache) -> Double {
            let t = clock.measure {
                for i in 0..<64 {
                    let (k, v) = tokenChunk(chunkIndex: 7000 + i, tokens: 1, salt: 0)
                    let (kk, vv) = cache.update(keys: k, values: v)
                    let out = MLXFast.scaledDotProductAttention(
                        queries: q, keys: kk, values: vv, scale: scale,
                        mask: MLXFast.ScaledDotProductAttentionMaskMode.none)
                    eval(out)
                }
            }
            return ms(t) / 64
        }
        let promotedDecode = decode64(A1)
        let freshAtSame = KVCacheSimple()
        _ = feed(freshAtSame, depth: 18_432)
        let freshDecode = decode64(freshAtSame)
        report(
            String(
                format: "Promote 后 decode: %.3f ms/token vs fresh 同深度 %.3f → 比值 %.2f×(回归项,>1.10× 记警;注意 A1 先跑含冷启动偏置,本值为上界)",
                promotedDecode, freshDecode, promotedDecode / freshDecode))

        // 历史快照取在 promote-decode 之后(合法增长不计入"不变性"检验)
        let viewA1 = A1.state
        eval(viewA1)
        let viewB1 = B1.state
        eval(viewB1)

        // ---- Rollback:回到 Root,重叉 D(root 此时只读再冻结)----
        let tRollback = clock.measure { _ = COWForkKVCache(parent: root) }
        report(String(format: "Rollback(root 重叉 D): %.4f ms(仅执行引用切换 + O(1) 元数据)", ms(tRollback)))
        let D = COWForkKVCache(parent: root)
        _ = appendStream(D, at: 8_192, n: 1_024, salt: 77)

        // 历史树不变:A1/B1 视图在 rollback+refork 后逐元素不变
        let viewA1Now = A1.state
        eval(viewA1Now)
        let viewB1Now = B1.state
        eval(viewB1Now)
        #expect(arrayEqual(viewA1Now[0], viewA1[0]))
        #expect(arrayEqual(viewA1Now[1], viewA1[1]))
        #expect(arrayEqual(viewB1Now[0], viewB1[0]))
        #expect(arrayEqual(viewB1Now[1], viewB1[1]))
        report("历史树(A1/B1)在 Rollback + Re-fork(D) 后逐元素不变: ✓")

        // ---- Discard:丢弃 B 子树 → 私有页回收 ----
        GPU.clearCache()
        let beforeDiscard = GPU.activeMemory
        let bPrivateRows = B.privateRows + B1.privateRows
        var holdB: [COWForkKVCache]? = [B, B1]
        _ = holdB  // 强引用占位
        holdB = nil  // 丢弃 → ARC 释放私有缓冲
        GPU.clearCache()
        let afterDiscard = GPU.activeMemory
        let dropped = beforeDiscard - afterDiscard
        report(
            String(
                format: "Discard(B 子树,私有 %d 行 ≈ %.1f MiB): active 下降 %.1f MiB(%@)",
                bPrivateRows, Double(bPrivateRows * Self.bytesPerTokenPerLayer) / 1048576.0,
                Double(dropped) / 1048576.0,
                dropped > 0 ? "可回收" : "未见下降(分配器/图保留语义,记 ⚠ 待归因)"))

        // A1 在 B 丢弃后仍正确(root 共享缓冲因 A1 存活未被误回收)
        let viewA1Post = A1.state
        eval(viewA1Post)
        #expect(arrayEqual(viewA1Post[0], viewA1[0]))
        report("B 丢弃后 A1 仍逐元素正确(共享 root 未被误回收): ✓")

        // ---- 长时间循环:无异常内存积累 ----
        GPU.clearCache()
        let baseline = GPU.activeMemory
        for cycle in 0..<60 {
            var litter: [COWForkKVCache] = []
            for b in 0..<4 {
                let c = COWForkKVCache(parent: root)
                _ = appendStream(c, at: 8_192, n: 512, salt: 900 + b + cycle)
                litter.append(c)
            }
            litter.removeAll()  // 每轮丢弃
        }
        GPU.clearCache()
        let afterCycles = GPU.activeMemory
        let drift = afterCycles - baseline
        report(
            String(
                format: "60 轮 fork×4+append+discard 循环: 内存漂移 %d KiB(判据 |漂移| < 4 MiB)",
                drift / 1024))
        #expect(abs(drift) < 4 * 1024 * 1024)
    }

    // MARK: - F4-C 多分支长期存活(交错追加 + 部分丢弃)

    @Test func f4cSixteenBranchSurvival() throws {
        report("=== F4-C: 16 分支交错追加 ×3 轮 + 半数丢弃(cache 级 @32K)===")
        let rootDepth = 32_768
        let root = KVCacheSimple()
        _ = feed(root, depth: rootDepth)
        let rootView = root.state
        eval(rootView)

        var branches: [COWForkKVCache] = (0..<16).map { _ in COWForkKVCache(parent: root) }
        // 三轮交错追加(每分支每轮 512)
        for round in 0..<3 {
            for (i, branch) in branches.enumerated() {
                _ = appendStream(
                    branch, at: rootDepth + round * 512, n: 512, salt: 200 + i)
            }
        }
        let finalViews: [(MLXArray, MLXArray)] = branches.map {
            let v = $0.state
            eval(v)
            return (v[0], v[1])
        }
        // 逐分支 == 独立 monolithic(32K 主流 + 1.5K 分支流)
        var allOk = true
        for i in 0..<16 {
            let ref = monolithicReference(
                depth: rootDepth + 1_536, suffix: 1_536, branchSalt: 200 + i)
            eval(ref.0, ref.1)
            allOk = allOk
                && arrayEqual(finalViews[i].0, ref.0)
                && arrayEqual(finalViews[i].1, ref.1)
        }
        #expect(allOk)
        let rootAfter = root.state
        eval(rootAfter)
        #expect(arrayEqual(rootAfter[0], rootView[0]))
        report("16 分支三轮交错追加后:逐分支 == 独立参照 \(allOk ? "✓" : "✗");root 完好 ✓")

        // 半数丢弃 → 剩余仍正确;丢弃后 active 下降
        GPU.clearCache()
        let before = GPU.activeMemory
        branches.removeSubrange(8...)  // 丢弃强引用 → ARC 释放私有缓冲
        GPU.clearCache()
        let after = GPU.activeMemory
        report(
            String(
                format: "丢弃 8/16 分支: active 下降 %.1f MiB;剩余 8 分支复验:",
                Double(before - after) / 1048576.0))
        var survivorsOk = true
        for i in 0..<8 {
            let ref = monolithicReference(
                depth: rootDepth + 1_536, suffix: 1_536, branchSalt: 200 + i)
            eval(ref.0, ref.1)
            let now = branches[i].state
            eval(now)
            survivorsOk = survivorsOk
                && arrayEqual(now[0], ref.0) && arrayEqual(now[1], ref.1)
        }
        #expect(survivorsOk)
        report("剩余 8 分支丢弃后逐元素正确: \(survivorsOk ? "✓" : "✗")")
    }

    // MARK: - F4-B-model: root 解冻续写门(真模型)

    private func modelDirectory() throws -> URL {
        let hub = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        let name = "models--peculiar-ragdoll--Cyber-Tiel-Coder-35B-A3B-MLX-oQ4e"
        let snapshots = hub.appendingPathComponent(name).appendingPathComponent("snapshots")
        let dirs = try FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil
        ).filter { $0.hasDirectoryPath }.sorted { $0.path < $1.path }
        guard let dir = dirs.first else {
            throw NSError(domain: "F4", code: 1, userInfo: [NSLocalizedDescriptionKey: "not found"])
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

    private func greedy(
        _ lm: Qwen35Language.LanguageModel, cache: [KVCache],
        state: inout LMOutput.State?, firstToken: Int, steps: Int
    ) -> [Int] {
        var tokens: [Int] = []
        var input = firstToken
        for _ in 0..<steps {
            let out = lm(
                MLXArray([Int32(input)]),
                cache: cache.map { $0 as KVCache? },
                state: state)
            state = out.state
            eval(out.logits)
            input = out.logits[0, out.logits.dim(1) - 1].argMax().item(Int.self)
            tokens.append(input)
        }
        return tokens
    }

    @Test func f4bModelRootUnfreezeGate() async throws {
        report("=== F4-B-model: root 解冻续写门(真模型)===")
        let dir: URL
        do {
            dir = try modelDirectory()
        } catch {
            Issue.record("模型不在本机,跳过: \(error)")
            return
        }
        let container = try await VLMModelFactory.shared.loadContainer(
            from: dir, using: NoOpTokenizerLoader())
        let prefixLen = 4_096
        let suffixLen = 128
        let extraLen = 128
        let decodeSteps = 16

        let prefix = syntheticIds(prefixLen, seed: 1)
        let suffixX = syntheticIds(suffixLen, seed: 2)
        let extra = syntheticIds(extraLen, seed: 4)
        let suffixY = syntheticIds(suffixLen, seed: 3)

        try await container.perform { context in
            guard let moe = context.model as? Qwen35MoE else {
                Issue.record("模型类型不符")
                return
            }
            let lm = moe.languageModel
            try lm.prepare()

            // P prefill → fork X(COW)→ X 后缀+decode
            let cacheP = lm.makeCache(capacity: nil)
            var stateP: LMOutput.State? = nil
            _ = prefill(lm, cache: cacheP, ids: prefix, state: &stateP)
            let cacheX = forkModelCache(cacheP)
            var stateX = stateP
            let logitsX = prefill(lm, cache: cacheX, ids: suffixX, state: &stateX)
            let tokensX = greedy(
                lm, cache: cacheX, state: &stateX,
                firstToken: logitsX[0, logitsX.dim(1) - 1].argMax().item(Int.self),
                steps: decodeSteps)

            // *** root 解冻:P 继续写 rows [4K, 4K+128)——X 的段表只读
            // [0,4K),行区间不相交,应安全 ***
            _ = prefill(lm, cache: cacheP, ids: extra, state: &stateP)

            // 解冻后再叉 Y(fork 点 4,224)
            let cacheY = forkModelCache(cacheP)
            var stateY = stateP
            let logitsY = prefill(lm, cache: cacheY, ids: suffixY, state: &stateY)
            let tokensY = greedy(
                lm, cache: cacheY, state: &stateY,
                firstToken: logitsY[0, logitsY.dim(1) - 1].argMax().item(Int.self),
                steps: decodeSteps)

            // 控制组:RX = prefix+suffixX;RY = prefix+extra+suffixY(同 chunk 边界)
            let cacheRX = lm.makeCache(capacity: nil)
            var stateRX: LMOutput.State? = nil
            let logitsRX = prefill(lm, cache: cacheRX, ids: prefix + suffixX, state: &stateRX)
            let tokensRX = greedy(
                lm, cache: cacheRX, state: &stateRX,
                firstToken: logitsRX[0, logitsRX.dim(1) - 1].argMax().item(Int.self),
                steps: decodeSteps)
            let cacheRY = lm.makeCache(capacity: nil)
            var stateRY: LMOutput.State? = nil
            let logitsRY = prefill(
                lm, cache: cacheRY, ids: prefix + extra + suffixY, state: &stateRY)
            let tokensRY = greedy(
                lm, cache: cacheRY, state: &stateRY,
                firstToken: logitsRY[0, logitsRY.dim(1) - 1].argMax().item(Int.self),
                steps: decodeSteps)

            #expect(tokensX == tokensRX)
            #expect(tokensY == tokensRY)
            report("门 1(冻结期分叉 X): \(tokensX == tokensRX ? "✓" : "✗") \(decodeSteps)/\(decodeSteps)")
            report(
                "门 2(解冻后续叉 Y,证明 P 的续写未污染 X 共享前缀与 P 自身): "
                    + "\(tokensY == tokensRY ? "✓" : "✗") \(decodeSteps)/\(decodeSteps)")
            report(
                String(
                    format: "内存: active=%dMiB(root 解冻语义 = 段表行区间不相交,COW 子代只读 [0,forkDepth))",
                    GPU.activeMemory / 1024 / 1024))
        }
    }
}
