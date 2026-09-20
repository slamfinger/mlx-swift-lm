// Copyright © 2026 SimiGo-Lab. F2 门 — 真模型 fork → 双分叉 → 逐 token 一致
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径)
// 记录:SimiGo-Lab trackB-execution-state/(F2,承 F1)
//
// F2b(报告 §5.1 验收指标 4,一票否决):
//   Cyber-Tiel-Coder-35B-A3B-MLX-oQ4e(Qwen3.5 MoE 混合架构)真模型,
//   prefill 公共前缀 → fork 出两个子代(page-split attention + GDN 引用)
//   → 各自不同后缀续跑 → greedy decode,与同 chunk 边界的从零控制组
//   逐 token 比对。附:单发全量 prefill 控制组(信息项,允许数值漂移)。
//
// E5(F2a,视图物化税消解的 fast path):
//   单写者 slack fork——子代直接在父代缓冲余量(offset<capacity)内续写,
//   零私有存储、零 concat;对照 F1 的 page-split(2.36× decode 税)。

import Foundation
import MLX
import MLXNN
import Testing
import BenchmarkHelpers

@testable import MLXVLM
@testable import MLXLMCommon

// MARK: - 共享原型(F1 提升为文件级)

/// F1/F2 原型:共享前缀(父代缓冲只读)+ 私有后缀(子代自增长)。
/// 返回视图 = concatenated(共享前缀, 私有已用行)。
/// 非生产 API —— 只存在于 exp/execution-state-fork。
final class COWForkKVCache: BaseKVCache {
    let sharedKeys: MLXArray  // 父代原始缓冲(含 step 填充),子代只读 [0, forkDepth)
    let sharedValues: MLXArray
    let forkDepth: Int
    var privateKeys: MLXArray?
    var privateValues: MLXArray?
    var step = 256

    init(parent: KVCacheSimple) {
        guard let pk = parent.keys, let pv = parent.values else {
            fatalError("parent has no storage")
        }
        self.sharedKeys = pk
        self.sharedValues = pv
        self.forkDepth = parent.offset
        super.init()
        self.offset = parent.offset
    }

    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = offset - forkDepth
        let reset =
            if let currentKeys = privateKeys, (previous + keys.dim(2)) > currentKeys.dim(2) {
                true
            } else {
                privateKeys == nil
            }
        if reset {
            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [
                keys.dim(0), keys.dim(1), nSteps * step, keys.dim(3),
            ]
            let vShape = [
                values.dim(0), values.dim(1), nSteps * step, values.dim(3),
            ]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)
            if var currentKeys = privateKeys, var currentValues = privateValues {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                privateKeys = concatenated([currentKeys, newK], axis: 2)
                privateValues = concatenated([currentValues, newV], axis: 2)
            } else {
                privateKeys = newK
                privateValues = newV
            }
        }
        offset += keys.dim(2)
        let used = offset - forkDepth
        privateKeys?[.ellipsis, previous..<used, 0...] = keys
        privateValues?[.ellipsis, previous..<used, 0...] = values

        if let pk = privateKeys, let pv = privateValues {
            return (
                concatenated([
                    sharedKeys[.ellipsis, ..<forkDepth, 0...],
                    pk[.ellipsis, ..<used, 0...],
                ], axis: 2),
                concatenated([
                    sharedValues[.ellipsis, ..<forkDepth, 0...],
                    pv[.ellipsis, ..<used, 0...],
                ], axis: 2)
            )
        }
        return (
            sharedKeys[.ellipsis, ..<forkDepth, 0...],
            sharedValues[.ellipsis, ..<forkDepth, 0...]
        )
    }

    override var isTrimmable: Bool { false }

    /// 私有后缀已用行数(不含 step 填充)
    var usedPrivateRows: Int { offset - forkDepth }

    /// 私有后缀存储行数(含 step 填充)
    var privateRows: Int { privateKeys?.dim(2) ?? 0 }
}

/// 模型级 cache fork:attention(KVCacheSimple)→ page-split 子代;
/// GDN(MambaCache)→ 引用子代(两槽函数式更新,零拷贝)。
func forkModelCache(_ cache: [KVCache]) -> [KVCache] {
    cache.map { layer in
        switch layer {
        case let simple as KVCacheSimple:
            return COWForkKVCache(parent: simple)
        case let mamba as MambaCache:
            let child = MambaCache()
            for slot in 0..<mamba.slotCount {
                child[slot] = mamba[slot]
            }
            child.offset = mamba.offset
            return child
        default:
            fatalError("forkModelCache: unhandled cache type \(type(of: layer))")
        }
    }
}

// MARK: - E5 slack-fork fast path(单写者,零 concat 税)

@Suite(.serialized)
struct ExecutionStateForkF2Gate {

    static let kvHeads = 4
    static let headDim = 128
    static let bytesPerTokenPerLayer = 2 * kvHeads * headDim * 2
    static let scale = Float(pow(Float(headDim), -0.5))

    private func report(_ line: @autoclosure () -> String) {
        print("[F2] " + line())
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

    @Test func e5SlackForkZeroTaxFastPath() throws {
        report("=== E5: 单写者 slack fork(父代缓冲余量内续写,零 concat)===")
        // depth 65,400:capacity 取整到 65,536,余量 136 ≥ 128 decode 步
        let depth = 65_400
        let decodeSteps = 128

        // 基线:monolithic 先喂 depth 再 decode(同深度公平)
        let mono = KVCacheSimple()
        _ = feed(mono, depth: depth)
        // slack 子代:直接共享父代缓冲,offset 接续,不建私有存储
        let parent = KVCacheSimple()
        _ = feed(parent, depth: depth)
        #expect(parent.keys!.dim(2) > parent.offset)
        let slack = parent.keys!.dim(2) - parent.offset
        let child = KVCacheSimple()
        child.step = parent.step
        child.keys = parent.keys
        child.values = parent.values
        child.offset = parent.offset

        // page-split 对照(F1 的 2.36× 税路径)
        let cow = COWForkKVCache(parent: parent)

        let q = MLXArray.zeros([1, Self.kvHeads, 1, Self.headDim], dtype: .float16)
        func decodeLoop(_ cache: any KVCache, tag: String) {
            let clock = ContinuousClock()
            let t = clock.measure {
                for i in 0..<decodeSteps {
                    let (k, v) = tokenChunk(chunkIndex: 5000 + i, tokens: 1, salt: 0)
                    let (kk, vv) = cache.update(keys: k, values: v)
                    let out = MLXFast.scaledDotProductAttention(
                        queries: q, keys: kk, values: vv, scale: Self.scale,
                        mask: MLXFast.ScaledDotProductAttentionMaskMode.none)
                    eval(out)
                }
            }
            report(
                String(
                    format: "  %@: %6.3f ms/token(深度 %d)",
                    tag, ms(t) / Double(decodeSteps), cache.offset))
        }
        report("slack=\(slack) 行;decode \(decodeSteps) 步对比:")
        decodeLoop(mono, tag: "baseline mono    ")
        decodeLoop(child, tag: "slack-fork child ")
        decodeLoop(cow, tag: "page-split COW   ")

        // 正确性:slack 子代终视图 == monolithic 同流终视图(深度一致)
        let monoLast = mono.state
        let childLast = child.state
        #expect(mono.offset == child.offset)
        #expect(arrayEqual(childLast[0], monoLast[0]))
        #expect(arrayEqual(childLast[1], monoLast[1]))
        report("slack 子代终视图 == monolithic(同流同深度)逐元素相等: ✓")
        report("父代 offset 保持 \(parent.offset)(冻结),子代已推进到 \(child.offset)")
    }

    // MARK: - F2b 真模型逐 token 一致门

    static let prefixLen = 2_048
    static let suffixLen = 128
    static let decodeSteps = 16
    static let chunk = 512

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
                domain: "F2Gate", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "model snapshot not found: \(snapshots)"])
        }
        return dir
    }

    /// 确定性 token id 流(避开特殊 token 区段;vocab=248,320)
    private func syntheticIds(_ n: Int, seed: Int) -> [Int] {
        (0..<n).map { i in
            2000 + ((i * 7919 + seed * 104729) % 140_000)
        }
    }

    private func idsArray(_ ids: [Int]) -> MLXArray {
        MLXArray(ids.map(Int32.init))
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
            let n = min(Self.chunk, ids.count - start)
            let out = lm(
                idsArray(Array(ids[start..<(start + n)])),
                cache: cache.map { $0 as KVCache? },
                state: state)
            logits = out.logits
            state = out.state
            eval(logits)
            start += n
        }
        return logits
    }

    private func greedyDecode(
        _ lm: Qwen35Language.LanguageModel,
        cache: [KVCache],
        state: inout LMOutput.State?,
        firstToken: Int,
        steps: Int
    ) -> [Int] {
        var tokens: [Int] = []
        var input = firstToken
        for _ in 0..<steps {
            let out = lm(
                idsArray([input]),
                cache: cache.map { $0 as KVCache? },
                state: state)
            state = out.state
            eval(out.logits)
            input = out.logits[0, out.logits.dim(1) - 1].argMax().item(Int.self)
            tokens.append(input)
        }
        return tokens
    }

    /// 末步 logits 的 greedy token
    private func argMaxRow(_ logits: MLXArray) -> Int {
        logits[0, logits.dim(1) - 1].argMax().item(Int.self)
    }

    @Test func f2bRealModelTokenExactGate() async throws {
        report("=== F2b: 真模型 fork → 双分叉 → 逐 token greedy 一致门 ===")
        let dir: URL
        do {
            dir = try modelDirectory()
        } catch {
            Issue.record("模型不在本机,跳过门测试: \(error)")
            return
        }
        report("model: \(dir.lastPathComponent)")

        let container = try await VLMModelFactory.shared.loadContainer(
            from: dir, using: NoOpTokenizerLoader())
        report(String(format: "模型加载后: active=%dMiB", GPU.activeMemory / 1024 / 1024))

        let prefix = syntheticIds(Self.prefixLen, seed: 1)
        let suffixX = syntheticIds(Self.suffixLen, seed: 2)
        let suffixY = syntheticIds(Self.suffixLen, seed: 3)

        try await container.perform { context in
            guard let moe = context.model as? Qwen35MoE else {
                Issue.record("模型类型不符: \(type(of: context.model))")
                return
            }
            let lm = moe.languageModel
            try lm.prepare()

            // ---- P: 公共前缀 prefill ----
            let clock = ContinuousClock()
            let cacheP = lm.makeCache(capacity: nil)
            var stateP: LMOutput.State? = nil
            let tPre = clock.measure {
                _ = prefill(lm, cache: cacheP, ids: prefix, state: &stateP)
            }
            report(String(format: "P prefill %d tokens: %.0f ms", Self.prefixLen, ms(tPre)))

            // ---- fork:双分叉 ----
            var cacheX: [KVCache]!
            var cacheY: [KVCache]!
            let tFork = clock.measure {
                cacheX = forkModelCache(cacheP)
                cacheY = forkModelCache(cacheP)
            }
            report(String(format: "fork ×2(40 层): %.3f ms", ms(tFork)))
            // state 也是 execution state 的一部分(Swift 字典值语义 → 天然 fork)
            var stateX = stateP
            var stateY = stateP

            // ---- X/Y:各自后缀 + greedy decode ----
            let logitsX = prefill(lm, cache: cacheX, ids: suffixX, state: &stateX)
            let tokensX = greedyDecode(
                lm, cache: cacheX, state: &stateX, firstToken: argMaxRow(logitsX),
                steps: Self.decodeSteps)
            let logitsY = prefill(lm, cache: cacheY, ids: suffixY, state: &stateY)
            let tokensY = greedyDecode(
                lm, cache: cacheY, state: &stateY, firstToken: argMaxRow(logitsY),
                steps: Self.decodeSteps)
            report("fork X 续跑 tokens: \(tokensX)")
            report("fork Y 续跑 tokens: \(tokensY)")

            // ---- 控制组 A:同 chunk 边界从零(必须逐 token 一致)----
            let cacheRA = lm.makeCache(capacity: nil)
            var stateRA: LMOutput.State? = nil
            let logitsRA = prefill(lm, cache: cacheRA, ids: prefix + suffixX, state: &stateRA)
            let tokensRA = greedyDecode(
                lm, cache: cacheRA, state: &stateRA, firstToken: argMaxRow(logitsRA),
                steps: Self.decodeSteps)
            report("控制 A tokens:      \(tokensRA)")

            // ---- 门:逐 token 一致(一票否决)----
            #expect(tokensX == tokensRA)
            if tokensX == tokensRA {
                report("*** 正确性门(逐 token greedy 一致,fork vs 从零): ✓ 全部 \(Self.decodeSteps) 步 ***")
            } else {
                report("*** 正确性门失败:fork 与从零分叉 ***")
            }

            // ---- 控制组 B:单发全量 prefill(信息项;允许 chunk 归约数值漂移)----
            let cacheRB = lm.makeCache(capacity: nil)
            var stateRB: LMOutput.State? = nil
            let outRB = lm(
                idsArray(prefix + suffixX),
                cache: cacheRB.map { $0 as KVCache? },
                state: stateRB)
            eval(outRB.logits)
            stateRB = outRB.state
            let tokensRB = greedyDecode(
                lm, cache: cacheRB, state: &stateRB, firstToken: argMaxRow(outRB.logits),
                steps: Self.decodeSteps)
            let driftB = zip(tokensX, tokensRB).filter { $0 != $1 }.count
            report("控制 B(单发 prefill)tokens: \(tokensRB) — 与 fork 漂移 \(driftB)/\(Self.decodeSteps) 步(信息项)")

            // ---- 模型级 fork 记账 ----
            let attn = cacheX.filter { $0 is COWForkKVCache }
            let gdn = cacheX.filter { $0 is MambaCache }
            let privateRows = attn.compactMap { ($0 as? COWForkKVCache)?.privateRows }
                .reduce(0, +)
            let usedRows = attn.compactMap { ($0 as? COWForkKVCache)?.usedPrivateRows }
                .reduce(0, +)
            let sharedRows = Self.prefixLen * attn.count
            // 真实形状:kvHeads=2, headDim=256 → 2KiB/token/层(fp16)
            let bytesPerRow = 2 * 2 * 256 * 2 * 2  // K+V
            report(
                "X 记账: attention \(attn.count) 层(page-split,shared \(sharedRows) 行 + private \(privateRows) 行[used \(usedRows)])"
                    + ";GDN \(gdn.count) 层(引用 fork,零拷贝)")
            report(
                String(
                    format: "模型级物理增长/后缀 = %.2f(判据 ∝ 后缀 %d 行,非 ∝ 前缀 %d 行)",
                    Double(privateRows) / Double(Self.suffixLen + Self.decodeSteps),
                    Self.suffixLen + Self.decodeSteps, Self.prefixLen))
            report(
                String(
                    format: "模型级 shared ratio = %.3f;X 增量内存 ≈ %.1f MiB",
                    Double(sharedRows) / Double(sharedRows + privateRows),
                    Double(privateRows * bytesPerRow) / 1048576))

            // 父代冻结校验:offset 不动
            let parentAttnOffsets = cacheP.compactMap { $0 as? KVCacheSimple }.map(\.offset)
            #expect(parentAttnOffsets.allSatisfy { $0 == Self.prefixLen })
            report("父代 attention 层 offset 全部保持 \(Self.prefixLen)(冻结): ✓")
        }
    }
}
