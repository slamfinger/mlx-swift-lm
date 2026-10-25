// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation
import MLX

/// An ADDRESS-LEVEL observation of the arrays a realized KV cache is made of.
///
/// Why this exists: `KVCache.copy()` is documented as an independent deep copy,
/// while `KVCacheSimple.copy()` stores `state.map { $0[.ellipsis] }` — a slice
/// expression. The documentation and the implementation do not settle, on their
/// own, whether the underlying allocation is shared. That question has to be
/// answered by sampling the storage of a live parent/child pair, not by reading
/// the source.
///
/// What this is NOT:
/// - Not an ownership proof. Two records with the same `address` read the same
///   bytes at sample time; that is all.
/// - Not a durable identifier. An address is only meaningful while the arrays
///   that produced it are alive, and it must never be stored across processes,
///   runs, or even cache rewrites (MLX reallocates on append, so an address can
///   be reused by an unrelated allocation later).
/// - Not a statement about physical memory. Even a proven last-reference release
///   would only end an allocation's lifetime; the OS reclaiming those pages is a
///   separate claim this probe never makes.
///
/// Value digests are OPT-IN (`includeValues`). Two arrays can read the same
/// address and still be two logically independent states, and one array can be
/// rewritten in place under a reader that kept its address. Only the digest
/// answers whether the bytes a session *sees* changed.
public struct KVStorageSample: Sendable, Hashable {

    /// One array's storage as read while the array was alive.
    public struct ArrayRecord: Sendable, Hashable {
        /// `main`, `draft`, or a caller-supplied label.
        public let tag: String
        /// Stable leaf path through a possibly nested cache topology.
        public let path: [Int]
        /// Runtime class of the leaf cache that owns this array.
        public let cacheKind: String
        /// `key` / `value` for two-array attention leaves, `state[i]` otherwise.
        public let role: String
        public let dtype: String
        public let shape: [Int]
        public let strides: [Int]
        public let byteCount: Int
        /// Host-visible address of the array's first element, in bytes, read from
        /// MLX's buffer (see `observe(_:)` for why not through `Data`).
        /// `0` when the array has no addressable storage.
        public let address: UInt
        /// A second read of the SAME array returned the same address, so this is
        /// the array's storage and not a per-read temporary.
        public let addressStable: Bool
        /// FNV-1a over the array's CONTIGUOUS logical bytes, `nil` unless values
        /// were requested. Equal digests with equal shapes mean the two arrays
        /// present the same values at sample time; nothing about identity.
        ///
        /// The bytes are read through `.copy`, so the digest never depends on the
        /// view's geometry: a strided view's linear byte range is not its logical
        /// contents (see `strides` versus `byteCount`).
        public let valueDigest: UInt64?

        public var isObserved: Bool { address != 0 }
        /// A stable non-zero address is the minimum bar for comparing two
        /// samples; anything less cannot be read as either sharing or separation.
        public var isComparable: Bool { isObserved && addressStable }
    }

    /// Process-level allocator counters, recorded as CONTEXT only.
    ///
    /// They are not attributable to any record above: a drop here cannot prove
    /// that a specific allocation was released, and a rise cannot prove that one
    /// is still held.
    public struct AllocatorCounters: Sendable, Hashable {
        public let activeBytes: Int
        public let cachedBytes: Int
        public let peakBytes: Int
    }

    public let records: [ArrayRecord]
    /// Token progress the cache reported at sample time, when sampled from a
    /// session.
    public let processedTokenCount: Int?
    public let allocator: AllocatorCounters

    public init(
        records: [ArrayRecord], processedTokenCount: Int?, allocator: AllocatorCounters
    ) {
        self.records = records
        self.processedTokenCount = processedTokenCount
        self.allocator = allocator
    }

    public var isComparable: Bool {
        !records.isEmpty && records.allSatisfy(\.isComparable)
    }
}

/// How two samples relate, record by record.
public struct KVStorageOverlap: Sendable, Hashable {

    public struct Pair: Sendable, Hashable {
        public let tag: String
        public let path: [Int]
        public let role: String
        public let firstAddress: UInt
        public let secondAddress: UInt
        public let firstShape: [Int]
        public let secondShape: [Int]
        public let firstDigest: UInt64?
        public let secondDigest: UInt64?

        /// Same non-zero address in both samples. Reads the same bytes at sample
        /// time — not "the same object forever".
        public var isSameStorage: Bool {
            firstAddress != 0 && firstAddress == secondAddress
        }

        /// Same shape and same value digest: the two arrays present the same
        /// values. `nil` digests (values not requested) can never satisfy this.
        public var isSameValue: Bool {
            guard let firstDigest, let secondDigest else { return false }
            return firstShape == secondShape && firstDigest == secondDigest
        }
    }

    public let pairs: [Pair]
    /// Records present in only one sample, as `tag/path/role`, so a topology
    /// change cannot be mistaken for independence.
    public let onlyInFirst: [String]
    public let onlyInSecond: [String]

    public init(pairs: [Pair], onlyInFirst: [String], onlyInSecond: [String]) {
        self.pairs = pairs
        self.onlyInFirst = onlyInFirst
        self.onlyInSecond = onlyInSecond
    }

    public var comparedCount: Int { pairs.count }
    public var sharedCount: Int { pairs.filter(\.isSameStorage).count }
    /// Matched records that present the same shape AND the same value digest.
    public var sameValueCount: Int { pairs.filter(\.isSameValue).count }
    /// How many records could be value-compared at all (values were requested).
    public var valueComparedCount: Int {
        pairs.filter { $0.firstDigest != nil && $0.secondDigest != nil }.count
    }

    /// Every matched record reads the same values. This is the logical-isolation
    /// question: two caches can be "independent" in storage and still look equal,
    /// or share storage and still present different values.
    public var isValueFullyUnchanged: Bool {
        !pairs.isEmpty && onlyInFirst.isEmpty && onlyInSecond.isEmpty
            && valueComparedCount == pairs.count && sameValueCount == pairs.count
    }

    /// Every matched record reads the same bytes. Meaningful only when the
    /// topology matched as well.
    public var isFullyShared: Bool {
        !pairs.isEmpty && onlyInFirst.isEmpty && onlyInSecond.isEmpty
            && sharedCount == pairs.count
    }

    /// Every matched record reads different bytes.
    public var isFullyIndependent: Bool {
        !pairs.isEmpty && sharedCount == 0
    }

    public var summary: String {
        "compared=\(comparedCount) shared=\(sharedCount)"
            + " sameValue=\(sameValueCount)/\(valueComparedCount)"
            + " onlyFirst=\(onlyInFirst.count) onlySecond=\(onlyInSecond.count)"
    }
}

/// Read-only instrument for KV storage. Sampling never mutates the cache, but it
/// does force evaluation of lazy arrays (an address requires realized storage)
/// and therefore synchronizes the cache it is looking at.
public enum KVStorageProbe {

    /// Sample arrays that are not behind a `ChatSession`.
    ///
    /// The instrument's own self-check uses this on a view that shares storage by
    /// construction (`array[.ellipsis]`): if such a pair does not report the same
    /// address, no equality result elsewhere can be read as sharing.
    public static func sample(
        _ arrays: [MLXArray], tag: String = "raw", path: [Int] = [],
        includeValues: Bool = false, valueRows: Int? = nil
    ) -> [KVStorageSample.ArrayRecord] {
        arrays.enumerated().map { index, array in
            record(
                window(array, rows: valueRows), tag: tag, path: path, cacheKind: "MLXArray",
                role: "array[\(index)]", includeValues: includeValues)
        }
    }

    /// Compare two samples by `(tag, path, role)`.
    public static func compare(
        _ first: KVStorageSample, _ second: KVStorageSample
    ) -> KVStorageOverlap {
        compare(first.records, second.records)
    }

    /// Compare two record lists by `(tag, path, role)`.
    ///
    /// The array form exists for callers that already hold bare records — for
    /// example two arrays out of a larger sample, when the question is about a
    /// subset rather than the whole cache.
    public static func compare(
        _ first: [KVStorageSample.ArrayRecord], _ second: [KVStorageSample.ArrayRecord]
    ) -> KVStorageOverlap {
        let secondByKey = Dictionary(
            second.map { (key($0), $0) }, uniquingKeysWith: { a, _ in a })
        var pairs: [KVStorageOverlap.Pair] = []
        var onlyInFirst: [String] = []
        var seen = Set<String>()

        for record in first {
            let name = key(record)
            guard let other = secondByKey[name] else {
                onlyInFirst.append(name)
                continue
            }
            seen.insert(name)
            pairs.append(
                .init(
                    tag: record.tag, path: record.path, role: record.role,
                    firstAddress: record.address, secondAddress: other.address,
                    firstShape: record.shape, secondShape: other.shape,
                    firstDigest: record.valueDigest, secondDigest: other.valueDigest))
        }
        let onlyInSecond = second.map(key).filter { !seen.contains($0) }
        return KVStorageOverlap(
            pairs: pairs, onlyInFirst: onlyInFirst, onlyInSecond: onlyInSecond)
    }

    // MARK: - Sampling from a realized cache

    /// Internal entry point: every leaf of every cache in the list.
    static func sample(
        main: [KVCache], draft: [KVCache], processedTokenCount: Int?,
        includeValues: Bool = false, valueRows: Int? = nil
    ) -> KVStorageSample {
        var records: [KVStorageSample.ArrayRecord] = []
        records += sample(
            cache: main, tag: "main", includeValues: includeValues, valueRows: valueRows)
        records += sample(
            cache: draft, tag: "draft", includeValues: includeValues, valueRows: valueRows)
        return KVStorageSample(
            records: records, processedTokenCount: processedTokenCount,
            allocator: KVStorageSample.AllocatorCounters(
                activeBytes: Memory.activeMemory,
                cachedBytes: Memory.cacheMemory,
                peakBytes: Memory.peakMemory))
    }

    private static func sample(
        cache: [KVCache], tag: String, includeValues: Bool, valueRows: Int?
    ) -> [KVStorageSample.ArrayRecord] {
        guard !cache.isEmpty else { return [] }
        var records: [KVStorageSample.ArrayRecord] = []
        for leaf in KVCacheTree.leaves(in: cache) {
            let arrays = leaf.cache.state
            let kind = String(describing: type(of: leaf.cache))
            for (index, array) in arrays.enumerated() {
                records.append(
                    record(
                        window(array, rows: valueRows), tag: tag, path: leaf.path,
                        cacheKind: kind,
                        role: role(index: index, count: arrays.count, isAttention: leaf.isAttentionCache),
                        includeValues: includeValues))
            }
        }
        return records
    }

    /// The first `rows` rows of the sequence axis, as a VIEW.
    ///
    /// Nothing is copied here — the slice shares the cache's storage, and only the
    /// digest reads bytes. A window exists because a write changes the number of
    /// visible rows: whole-array digests taken before and after such a write have
    /// different shapes and therefore stop being comparable on the rows they do
    /// share. Truncating both to the same window makes "did the values on THOSE
    /// rows change" answerable, which is what separates a real write from a
    /// write-shaped no-op (an offset that advanced and an array that rebound).
    private static func window(_ array: MLXArray, rows: Int?) -> MLXArray {
        guard let rows, rows > 0, array.shape.count == 4, array.shape[2] > rows else {
            return array
        }
        return array[.ellipsis, ..<rows, 0...]
    }

    private static func role(index: Int, count: Int, isAttention: Bool) -> String {
        guard isAttention, count == 2 else { return "state[\(index)]" }
        return index == 0 ? "key" : "value"
    }

    private static func key(_ record: KVStorageSample.ArrayRecord) -> String {
        "\(record.tag)/\(record.path.map(String.init).joined(separator: "."))/\(record.role)"
    }

    private static func record(
        _ array: MLXArray, tag: String, path: [Int], cacheKind: String, role: String,
        includeValues: Bool
    ) -> KVStorageSample.ArrayRecord {
        let observed = observe(array)
        return .init(
            tag: tag, path: path, cacheKind: cacheKind, role: role,
            dtype: "\(array.dtype)", shape: array.shape, strides: array.strides,
            byteCount: array.nbytes, address: observed.address,
            addressStable: observed.stable,
            valueDigest: includeValues ? digest(array) : nil)
    }

    /// FNV-1a over the array's contiguous logical bytes.
    ///
    /// `.copy` on purpose: the digest must describe the array's values, not the
    /// linear extent of a strided view. Digesting is O(bytes) and opt-in, because
    /// a production cache is hundreds of megabytes.
    private static func digest(_ array: MLXArray) -> UInt64? {
        guard array.nbytes > 0 else { return nil }
        let data = array.asData(access: .copy).data
        guard !data.isEmpty else { return nil }
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        data.withUnsafeBytes { buffer in
            for byte in buffer {
                hash = (hash ^ UInt64(byte)) &* 0x0000_0100_0000_01b3
            }
        }
        return hash
    }

    /// Read the address of an array's first element.
    ///
    /// The pointer is read from `mlx_array_data_uint8` directly -- the same call the
    /// digest above reads through -- and NOT from `asData(access: .noCopy)`'s
    /// `Data.baseAddress`.
    ///
    /// That distinction is not cosmetic. The first version of this file used the
    /// `Data` wrapper because it looks like "an array's own bytes, no copy". Swift
    /// `Data` keeps payloads under ~14 bytes **inline in the value**, so
    /// `baseAddress` for a 4- or 8-byte array points at the `Data`'s own storage, not
    /// at MLX's buffer. Measured consequence: four different 8-byte
    /// `scales`/`biases` arrays of one quantized cache all reported the SAME address
    /// (one stack slot) while presenting four different value digests. At 16 bytes
    /// and above the two agree, which is why the multi-kilobyte attention arrays
    /// never exposed it.
    ///
    /// With the raw pointer, an address and the digest that was taken from that same
    /// address describe the same bytes, which is the minimum bar for reading an equal
    /// address as "reads the same storage".
    private static func observe(_ array: MLXArray) -> (address: UInt, stable: Bool) {
        guard array.nbytes > 0 else { return (0, false) }
        array.eval()
        func read() -> UInt {
            guard let base = mlx_array_data_uint8(array.ctx) else { return 0 }
            return UInt(bitPattern: UnsafeRawPointer(base))
        }
        let first = read()
        guard first != 0 else { return (0, false) }
        return (first, read() == first)
    }
}
