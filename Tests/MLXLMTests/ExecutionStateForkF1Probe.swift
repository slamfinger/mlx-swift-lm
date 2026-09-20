// Copyright © 2026 SimiGo-Lab. F1 探针 — 2.0 Track B (Execution State Fork)
//
// 分支:exp/execution-state-fork(实验室分支,永不进入 1.7 发布路径)
// 记录:SimiGo-Lab trackB-execution-state/(探针 → 结果 → 决策三段式)
//
// 探针问题(对应 F0 三缺口的第一性量化,报告 §5.1 验收指标 1/3/4):
//   E1  copy() 成本时间线 —— fork 时 / 首次物化 / 稳态各花多少?
//       (补 F0 §5 "copy() 惰性物化未独立计时" 的欠账)
//   E2  引用共享 fork 的就地写语义 —— MLX 散射写是否穿透共享缓冲?
//       (决定 page-split 是否强制;Q2/Q3 语义未知,报告不判红)
//   E3  page-split fork 原型 —— fork 是否 O(1)?物理增长 ∝ 后缀?
//       解码步是否有 concat 附加带宽代价?
//   E4  GDN(MambaCache)引用 fork —— 函数式状态更新是否零拷贝、
//       双方续跑互不污染?(attention 就地写 vs GDN 函数式的对称性验证)
//
// 形状基准:Qwen3.5 35B(oQ4e)单 attention 层 GQA:kvHeads=4, headDim=128,
// fp16 → 2 KiB/token/层;×12 attention 层 = F0 实测 24 KiB/token。
// 本探针在单层上测,报告中标明换算。

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct ExecutionStateForkF1Probe {

    static let kvHeads = 4
    static let headDim = 128
    static let bytesPerTokenPerLayer = 2 /*K+V*/ * kvHeads * headDim * 2 /*fp16*/
    static let attentionLayers = 12  // full_attention_interval=4 量级的 attention 层数
    static let scale = Float(pow(Float(headDim), -0.5))

    private func report(_ line: @autoclosure () -> String) {
        print("[F1] " + line())
    }

    private func ms(_ duration: Duration) -> Double {
        let ns = Double(duration.components.seconds) * 1e9
            + Double(duration.components.attoseconds) / 1e9
        return ns / 1e6
    }

    /// 确定性 token 流:第 chunkIndex 块的 k/v,host 端生成
    private func tokenChunk(
        chunkIndex: Int, tokens: Int, salt: Int
    ) -> (MLXArray, MLXArray) {
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

    /// 向 cache 喂 depth 个 token(chunk 512),返回最后一步返回的 (k, v) 视图
    @discardableResult
    private func feed(
        _ cache: any KVCache, depth: Int, salt: Int = 0
    ) -> (MLXArray, MLXArray) {
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

    /// 独立快照:KVCacheSimple.state 已是惰性切片,eval 后即独立缓冲
    private func snapshot(_ cache: KVCacheSimple) -> [MLXArray] {
        let s = cache.state
        eval(s)
        return s
    }

    /// 独立快照(裸引用集合):必须手动加惰性切片再 eval
    private func snapshotRefs(_ arrays: [MLXArray]) -> [MLXArray] {
        let s = arrays.map { $0[.ellipsis] }
        eval(s)
        return s
    }

    private func arrayEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
        precondition(a.shape == b.shape, "shape mismatch \(a.shape) vs \(b.shape)")
        return sum(a .== b).item(Int.self) == a.size
    }

    private func memLine(_ tag: String) {
        report(
            "\(tag): active=\(GPU.activeMemory / 1024)KiB peak=\(GPU.peakMemory / 1024)KiB cache=\(GPU.cacheMemory / 1024)KiB"
        )
    }

    // MARK: - E1 copy() 成本时间线

    @Test func e1CopyCostTimeline() throws {
        report("=== E1: KVCacheSimple.copy() 成本时间线 ===")
        for depth in [16_384, 65_536, 131_072] {
            GPU.clearCache()
            let parent = KVCacheSimple()
            let (k, v) = feed(parent, depth: depth)
            eval(k, v)
            let prefixBytes = depth * Self.bytesPerTokenPerLayer
            report(
                "depth=\(depth) tokens, 单层 KV=\(prefixBytes / 1024)KiB (×\(Self.attentionLayers) 层 = \(prefixBytes * Self.attentionLayers / 1024 / 1024)MiB)"
            )
            memLine("after prefill")

            let clock = ContinuousClock()
            let tCopy = clock.measure { _ = parent.copy() }
            report(String(format: "  copy() 调用:            %8.3f ms", ms(tCopy)))
            memLine("after copy()")

            // 子代首次 update + eval —— 触发惰性切片物化的真实成本
            let child = parent.copy() as! KVCacheSimple
            let (sk, sv) = tokenChunk(chunkIndex: 999, tokens: 512, salt: 7)
            let tFirst = clock.measure {
                let r = child.update(keys: sk, values: sv)
                eval(r.0, r.1)
            }
            report(String(format: "  子代首次 update+eval:  %8.3f ms  (含 O(depth) 物化)", ms(tFirst)))
            memLine("after child first update")

            let (sk2, sv2) = tokenChunk(chunkIndex: 1000, tokens: 512, salt: 8)
            let tSteady = clock.measure {
                let r = child.update(keys: sk2, values: sv2)
                eval(r.0, r.1)
            }
            report(String(format: "  子代第二次 update+eval: %8.3f ms  (稳态)", ms(tSteady)))

            // 父代不受污染(子代物化在独立缓冲)
            let after = snapshot(parent)
            #expect(arrayEqual(after[0], k))
            #expect(arrayEqual(after[1], v))
            report("  父代 rows[0..<depth] 未被子代写入污染: ✓")
        }
    }

    // MARK: - E2 引用共享的就地写语义

    @Test func e2ReferenceShareInPlaceSemantics() throws {
        report("=== E2: 裸引用共享 fork 的就地写语义 ===")
        // depth 取 8000:capacity 按 step=256 取整到 8192,留出后缀写入余量,
        // 子代首写不触发缓冲增长 → 共享语义可观测
        let depth = 8_000
        let suffixLen = 128
        let parent = KVCacheSimple()
        let (k, v) = feed(parent, depth: depth)
        eval(k, v)
        let before = snapshot(parent)

        // 裸引用 fork:不走 copy() 的惰性切片,直接共享同一 MLXArray 对象
        let child = KVCacheSimple()
        child.step = parent.step
        child.keys = parent.keys
        child.values = parent.values
        child.offset = parent.offset

        // 子代写后缀 rows[D, D+suffix)
        let (sk, sv) = tokenChunk(chunkIndex: 999, tokens: suffixLen, salt: 7)
        _ = child.update(keys: sk, values: sv)
        eval(child.state)

        // Q1: 父代活跃行完好?(行区间不相交 → 共享也安全;或缓冲本就分离)
        let parentNow = snapshot(parent)
        #expect(arrayEqual(parentNow[0], before[0]))
        #expect(arrayEqual(parentNow[1], before[1]))
        report("Q1 子代写后缀后,父代 rows[0..<D) 完好: ✓ (行区间不相交)")

        // Q2: 写入是否落进共享缓冲?直接读父代缓冲的 [D, D+suffix) 行
        let parentRawSuffixK = parent.keys![.ellipsis, depth..<(depth + suffixLen), 0...]
        eval(parentRawSuffixK)
        let shared = arrayEqual(parentRawSuffixK, sk)
        report(
            shared
                ? "Q2 子代后缀写入出现在父代缓冲 rows[D,...): ✓ → MLX 散射写就地、缓冲真共享"
                : "Q2 子代后缀写入未出现在父代缓冲 → 散射写为功能性替换(缓冲分离)")

        // Q3: 父代随后也写 [D, ...) 会怎样?(双写冲突 = 引用共享 fork 的边界)
        let (pk, pv) = tokenChunk(chunkIndex: 1000, tokens: suffixLen, salt: 99)
        _ = parent.update(keys: pk, values: pv)
        let childSuffixAfterParentWrite = child.keys![.ellipsis, depth..<(depth + suffixLen), 0...]
        eval(childSuffixAfterParentWrite)
        let corrupted = !arrayEqual(childSuffixAfterParentWrite, sk)
        report(
            corrupted
                ? "Q3 父代双写 [D,...) 覆盖了子代后缀: ✗ → 双写冲突,page-split(后缀私有化)是强制项"
                : "Q3 父代双写未影响子代视图 → 无冲突,引用共享即安全 fork")
        report(
            "E2 结论:散射写语义 = \(shared ? "就地(共享缓冲)" : "功能性(独立缓冲)");"
                + "\(shared ? "单写者引用 fork 可行,双写需 page-split" : "引用共享即安全")")
    }

    // MARK: - E3 page-split fork 原型

    @Test func e3PageSplitForkPrototype() throws {
        report("=== E3: page-split fork 原型(共享前缀 + 私有后缀)===")
        let depth = 65_536
        let suffix = 2_048
        let clock = ContinuousClock()

        // ---- 基线:monolithic 全深度 ----
        GPU.clearCache()
        let mono = KVCacheSimple()
        let (monoK, monoV) = feed(mono, depth: depth + suffix)
        eval(monoK, monoV)
        memLine("baseline mono @\(depth + suffix) tokens")

        // ---- fork 路线 ----
        GPU.clearCache()
        let parent = KVCacheSimple()
        let (pk, pv) = feed(parent, depth: depth)
        eval(pk, pv)
        memLine("parent @\(depth) tokens")

        var forked: COWForkKVCache!
        let tFork = clock.measure { forked = COWForkKVCache(parent: parent) }
        report(String(format: "fork 调用:                 %8.3f ms", ms(tFork)))
        memLine("after fork")
        let activeAfterFork = GPU.activeMemory

        // 后写入 suffix —— 与 monolithic 同一逻辑 token 流(全局 chunk 编号)
        var fed = 0
        var last: (MLXArray, MLXArray)?
        while fed < suffix {
            let n = min(512, suffix - fed)
            let (k, v) = tokenChunk(chunkIndex: (depth + fed) / 512, tokens: n, salt: 0)
            last = forked.update(keys: k, values: v)
            fed += n
        }
        eval(last!.0, last!.1)
        let suffixBytes = suffix * Self.bytesPerTokenPerLayer
        report(
            "fork 后 active 内存增量:   \(max(0, GPU.activeMemory - activeAfterFork) / 1024)KiB"
                + "(私有存储 \(forked.privateRows * Self.bytesPerTokenPerLayer / 1024)KiB vs 理论后缀 \(suffixBytes / 1024)KiB)"
        )
        report(
            String(
                format: "物理增长/后缀字节 = %.2f (判据:∝ 后缀 %.0f,非 ∝ 全深度 %d)",
                Double(forked.privateRows) / Double(suffix),
                Double(suffixBytes),
                (depth + suffix) * Self.bytesPerTokenPerLayer))
        report(
            "shared page ratio(物理前缀/物理总量)= \(Double(depth) / Double(depth + forked.privateRows)) @ suffix=\(suffix)"
        )

        // ---- 正确性门(数组级):fork 路线的最终视图 == monolithic 最终视图 ----
        #expect(arrayEqual(last!.0, monoK))
        #expect(arrayEqual(last!.1, monoV))
        report("正确性门(数组级逐元素相等,fork 视图 vs monolithic): ✓")

        // 父代不受 fork 后缀写入影响
        let parentAfter = snapshot(parent)
        #expect(arrayEqual(parentAfter[0], pk))
        #expect(arrayEqual(parentAfter[1], pv))
        report("父代 rows[0..<D) 在子代后缀写入后完好: ✓")

        // ---- 解码步成本对比(attention 视图物化代价)----
        let decodeSteps = 128
        let q = MLXArray.zeros(
            [1, Self.kvHeads, 1, Self.headDim], dtype: .float16)

        func decodeLoop(_ cache: any KVCache, tag: String) {
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
                    format: "  %@: %d 步共 %8.1f ms → %6.3f ms/token(全深度 %d)",
                    tag, decodeSteps, ms(t), ms(t) / Double(decodeSteps),
                    cache.offset))
        }
        report("解码步对比(update+SDPA+eval,@深度 \(depth + suffix + decodeSteps)):")
        decodeLoop(mono, tag: "baseline KVCacheSimple")
        decodeLoop(forked, tag: "COWForkKVCache   ")
    }

    // MARK: - E4 GDN(MambaCache)引用 fork

    @Test func e4GDNReferenceFork() throws {
        report("=== E4: GDN 递归态(MambaCache)裸引用 fork ===")
        // 真跑 gatedDeltaUpdate:拿到真实递归态,再 fork 双路续跑
        let inputs = withRandomState(MLXRandom.RandomState(seed: 42)) {
            let B = 1, T = 1, Hk = 4, Dk = 128, Hv = 4, Dv = 128
            let dtype = DType.bfloat16
            return (
                q: MLXRandom.normal([B, T, Hk, Dk]).asType(dtype),
                k: MLXRandom.normal([B, T, Hk, Dk]).asType(dtype),
                v: MLXRandom.normal([B, T, Hv, Dv]).asType(dtype),
                a: MLXRandom.normal([B, T, Hv]).asType(dtype),
                b: MLXRandom.normal([B, T, Hv]).asType(dtype),
                aLog: (MLXRandom.normal([Hv]) * MLXArray(0.1)).asType(dtype),
                dtBias: MLXRandom.normal([Hv]).asType(dtype)
            )
        }
        let (_, parentState) = gatedDeltaUpdate(
            q: inputs.q, k: inputs.k, v: inputs.v,
            a: inputs.a, b: inputs.b,
            aLog: inputs.aLog, dtBias: inputs.dtBias)
        eval(parentState)

        let parent = MambaCache()
        parent[0] = parentState  // 引用,零拷贝
        let parentSnapshot = snapshotRefs(parent.state)

        let clock = ContinuousClock()
        GPU.clearCache()
        var child: MambaCache!
        let tFork = clock.measure {
            child = MambaCache()
            child[0] = parent[0]
        }
        report(String(format: "GDN fork(引用传递):     %8.4f ms", ms(tFork)))

        // 子代续跑一步(真实 kernel)
        let (_, childState2) = gatedDeltaUpdate(
            q: inputs.q, k: inputs.k, v: inputs.v,
            a: inputs.a, b: inputs.b,
            aLog: inputs.aLog, dtBias: inputs.dtBias, state: child[0])
        eval(childState2)
        child[0] = childState2

        // 父代状态被子代续跑污染了吗?
        let parentNow = snapshotRefs(parent.state)
        #expect(arrayEqual(parentNow[0], parentSnapshot[0]))
        report("子代续跑后父代 GDN 态完好: ✓(函数式更新,子代换引用不落原缓冲)")

        // 父代也续跑,子代不受影响;父代用不同输入(扰动 k —— GDN 状态转移
        // 只依赖 k/v/a/b,q 仅影响读出)→ 终态真实分叉
        let (_, parentState2) = withRandomState(MLXRandom.RandomState(seed: 43)) {
            gatedDeltaUpdate(
                q: inputs.q,
                k: inputs.k + MLXArray(0.01).asType(.bfloat16),
                v: inputs.v,
                a: inputs.a, b: inputs.b,
                aLog: inputs.aLog, dtBias: inputs.dtBias, state: parent[0])
        }
        eval(parentState2)
        parent[0] = parentState2
        let childNow = snapshotRefs(child.state)
        #expect(arrayEqual(childNow[0], childState2))
        report("父代续跑后子代 GDN 态完好: ✓ → 双向隔离,fork=纯元数据")

        // 双方终态不同(输入不同 → 真实分叉)
        #expect(!arrayEqual(parent[0]!, child[0]!))
        report("父/子终态互异(真实分叉): ✓")

        memLine("GDN fork 后")
        report(
            "注:GDN 态为 O(1) 尺寸(与序列长度无关),fork 成本恒定 —— 与 attention KV 的 page-split 互补"
        )
    }
}
