// Copyright © 2026 Apple Inc.

import Cmlx
import MLX

/// Why this file exists.
///
/// `ChatSession.fork()` is documented as an independent deep copy and is not one:
/// every `copy()` in this module writes `state.map { $0[.ellipsis] }`, a slice
/// expression that keeps reading the source's buffer. For two live branches that
/// is enough -- the probes in `KVCacheStorageIdentityProbeTests` show writes on
/// either side stay out of the sibling's visible rows -- but it is not enough for
/// a host that wants to *retire* the source: the survivor still holds the
/// source's KV allocations, so those pages cannot go away while it lives.
///
/// `forkDetached` is the narrower promise: the returned session's KV arrays own
/// storage the source does not read, verified before the session is handed back.
/// It is deliberately not the default `fork()`: it allocates and copies, and that
/// cost (and the transient peak it creates) has to be bought deliberately.
///
/// The boundary of the promise, stated so it cannot be read as more:
/// - Verified: the survivor's visible KV arrays do not read bytes the source
///   reads, and they present the same values, shapes, dtypes, offsets and cache
///   metadata as the source did.
/// - NOT verified: that the source's last reference is released, that the
///   allocator returns those bytes to the OS, or anything about physical pages.
///   Retiring the source is still a separate step with its own evidence.

/// A strict storage-isolation continuation that could not be established.
///
/// Every case is a refusal. There is no "fall back to a shared fork" case on
/// purpose: a caller that asked for detached storage and silently got a shared
/// one would be holding an isolation claim it cannot support.
public enum DetachedForkError: Error, Equatable, CustomStringConvertible {
    /// The source has no realized KV cache (never generated, or holds a transcript).
    case noRealizedCache(String)
    /// A cache kind has no detached copy. `CacheList` descends into children; a
    /// leaf that reaches here fails the whole fork instead of sharing storage.
    case unsupportedCacheKind(String)
    /// There was nothing to copy or nothing to verify.
    case emptyState(String)
    /// The copy could not be made (the native primitive returned an error).
    case copyFailed(String)
    /// The copy was made but does not satisfy the isolation contract.
    case verificationFailed(String)
    /// The cache is in a state that cannot be snapshotted faithfully.
    case invalidState(String)

    public var description: String {
        switch self {
        case .noRealizedCache(let detail):
            "detached fork has no realized KV cache: \(detail)"
        case .unsupportedCacheKind(let kind):
            "detached fork does not know how to copy \(kind) without sharing storage"
        case .emptyState(let detail):
            "detached fork had nothing to verify: \(detail)"
        case .copyFailed(let detail):
            "detached fork could not copy a KV array: \(detail)"
        case .verificationFailed(let detail):
            "detached fork failed verification: \(detail)"
        case .invalidState(let detail):
            "detached fork refused an inconsistent cache: \(detail)"
        }
    }
}

/// What one detached fork did, in the terms the gate actually checked.
///
/// A caller is expected to look at ``storageSeparated`` and ``valuesPreserved``
/// before letting the source retire; `forkDetached` only returns when both hold,
/// so seeing a report at all is already the positive case.
public struct DetachedForkReport: Sendable, Hashable {

    /// One cache list's share of the fork (`main`, or the speculative `draft`).
    public struct Section: Sendable, Hashable {
        public let tag: String
        public let leaves: Int
        public let arrays: Int
        public let bytes: Int
        /// Arrays whose values were compared element-wise and matched.
        public let valuesMatched: Int
        /// Arrays whose storage address differs from the source's.
        public let addressesSeparated: Int
        public let processedTokenCount: Int
        public let cacheKinds: [String]

        public var storageSeparated: Bool { arrays > 0 && addressesSeparated == arrays }
        public var valuesPreserved: Bool { arrays > 0 && valuesMatched == arrays }
    }

    public let main: Section
    public let draft: Section?

    public var sections: [Section] { [main] + (draft.map { [$0] } ?? []) }
    public var leaves: Int { sections.reduce(0) { $0 + $1.leaves } }
    public var arrays: Int { sections.reduce(0) { $0 + $1.arrays } }
    public var bytes: Int { sections.reduce(0) { $0 + $1.bytes } }
    public var storageSeparated: Bool { sections.allSatisfy(\.storageSeparated) }
    public var valuesPreserved: Bool { sections.allSatisfy(\.valuesPreserved) }
    public var cacheKinds: [String] { Array(Set(sections.flatMap(\.cacheKinds))).sorted() }

    public var summary: String {
        sections.map {
            "\($0.tag): leaves=\($0.leaves) arrays=\($0.arrays) bytes=\($0.bytes)"
                + " separated=\($0.addressesSeparated)/\($0.arrays)"
                + " values=\($0.valuesMatched)/\($0.arrays)"
                + " tokens=\($0.processedTokenCount)"
        }.joined(separator: " | ")
    }
}

/// Eager, verified deep copy of a realized KV cache.
///
/// Public because the host has to be able to run the same check per cache kind:
/// ``ChatSession/forkDetached(history:)`` uses this on the leaves a session holds,
/// and a caller that builds caches directly can detach and verify them here
/// without a model. Everything in it refuses rather than degrade.
public enum KVCacheDetachment {

    // MARK: - The native copy

    /// A copy that owns its own buffer.
    ///
    /// Which MLX op actually allocates was not assumed here, it was read out of
    /// the pinned backend and then checked by the gate:
    /// - `copy` / `mlx_copy` does **not** copy. `Copy::eval` is
    ///   `out.copy_shared_buffer(in)` (backend/common/common.cpp): a logical
    ///   alias. The first version of this file used it, and the gate rejected the
    ///   result at `main/0/state[0]: still reads the source's storage` -- which is
    ///   the whole reason the gate compares addresses instead of trusting the op.
    /// - `contiguous` copies only when it has to: `Contiguous::eval_gpu` shares
    ///   when `buffer_size() <= out.nbytes() + 16384 && row_contiguous`, which is
    ///   exactly the shape a full cache buffer has.
    /// - `positive` is `array(a)` in mlx: the same buffer, twice.
    /// - `concatenate` always materializes: `concatenate_gpu` starts with
    ///   `out.set_data(allocator::malloc(out.nbytes()))` (backend/metal/slicing.cpp)
    ///   and copies each input into it, whatever the inputs are.
    ///
    /// So the copy is `concatenated([empty, array], axis: 0)`: one allocation of
    /// exactly the array's size, and a byte copy with no arithmetic in between
    /// (an elementwise identity would be exact for floats, but it is not defined
    /// for every dtype a cache state can hold).
    ///
    /// The copy is lazy: it is a graph node until evaluated. Callers that are
    /// about to retire the source must `evaluate(_:)` first, otherwise the copy
    /// is still a dependency on the source's buffer.
    public static func owningCopy(_ array: MLXArray) throws -> MLXArray {
        // A zero-byte array has no storage to separate. Keep the value as it is
        // rather than asking the allocator for nothing.
        guard array.nbytes > 0 else { return array }
        // Rank 0 has no axis to concatenate along; give it one and take it back.
        let flat = array.ndim > 0 ? array : array.reshaped([1])
        let copied = concatenated([flat[0 ..< 0], flat], axis: 0)
        return array.ndim > 0 ? copied : copied.reshaped([])
    }

    static func owningCopies(of arrays: [MLXArray]) throws -> [MLXArray] {
        try arrays.map(owningCopy)
    }

    /// Copy one leaf of the cache tree, or refuse.
    ///
    /// `BaseKVCache` carries the per-kind implementations; anything that is only
    /// a `KVCache` conformance has no detached copy and fails here instead of
    /// being passed through as a shared one.
    public static func detachedCopy(of cache: KVCache) throws -> KVCache {
        guard let base = cache as? BaseKVCache else {
            throw DetachedForkError.unsupportedCacheKind(String(describing: type(of: cache)))
        }
        return try base.detachedCopy()
    }

    public static func detachedCopy(of cache: [KVCache]) throws -> [KVCache] {
        try cache.map(detachedCopy(of:))
    }

    /// Force every copied array to be realized.
    ///
    /// Until this runs, each copy is a `Copy` node whose input is the source's
    /// buffer. The whole point of the detached fork is that the survivor does not
    /// depend on the source, so the dependency has to be consumed *before* the
    /// source is allowed to go away -- not left for the next write to notice.
    public static func evaluate(_ cache: [KVCache]) {
        let arrays = KVCacheTree.leaves(in: cache).flatMap { $0.cache.state }
        guard !arrays.isEmpty else { return }
        eval(arrays)
    }

    // MARK: - The gate

    /// Verify that `detached` is a faithful, non-sharing copy of `source`.
    ///
    /// Checks, in the order of what would invalidate a continuation fastest:
    ///   - the logical timeline (processed token count) changed
    ///   - the cache tree has the same leaves, at the same paths, of the same kinds
    ///   - each leaf kept its offset, capacity and metadata
    ///   - each visible state array kept its shape, dtype and byte count
    ///   - each visible state array reads a DIFFERENT address than the source's
    ///   - each visible state array presents the SAME values as the source's
    ///
    /// The last two are the contract. The address check alone would accept a copy
    /// that landed the wrong rows; the value check alone would accept a slice,
    /// which is what `copy()` already does.
    public static func verify(
        source: [KVCache], detached: [KVCache], tag: String,
        sourceTokenCount: Int, detachedTokenCount: Int
    ) throws -> DetachedForkReport.Section {
        guard sourceTokenCount == detachedTokenCount else {
            throw DetachedForkError.verificationFailed(
                "\(tag): processed token count changed (\(sourceTokenCount) -> \(detachedTokenCount))")
        }
        guard !source.isEmpty else {
            throw DetachedForkError.emptyState("\(tag): the source has no realized cache leaves")
        }

        let sourceLeaves = KVCacheTree.leaves(in: source)
        let detachedLeaves = KVCacheTree.leaves(in: detached)
        guard sourceLeaves.count == detachedLeaves.count else {
            throw DetachedForkError.verificationFailed(
                "\(tag): leaf count changed (\(sourceLeaves.count) -> \(detachedLeaves.count))")
        }

        var arrays = 0
        var bytes = 0
        var valuesMatched = 0
        var addressesSeparated = 0
        var kinds = Set<String>()

        for (sourceLeaf, detachedLeaf) in zip(sourceLeaves, detachedLeaves) {
            let where_ = "\(tag)/\(sourceLeaf.path.map(String.init).joined(separator: "."))"
            let sourceKind = String(describing: type(of: sourceLeaf.cache))
            let detachedKind = String(describing: type(of: detachedLeaf.cache))
            // The path is part of a leaf's identity: a copy that reordered the
            // tree would compare unrelated arrays against each other.
            guard sourceLeaf.path == detachedLeaf.path, sourceKind == detachedKind else {
                throw DetachedForkError.verificationFailed(
                    "\(where_): cache leaf changed (\(sourceKind) -> \(detachedKind) at \(detachedLeaf.path))")
            }
            kinds.insert(sourceKind)

            guard sourceLeaf.cache.offset == detachedLeaf.cache.offset else {
                throw DetachedForkError.verificationFailed(
                    "\(where_): offset changed (\(sourceLeaf.cache.offset) -> \(detachedLeaf.cache.offset))")
            }
            guard sourceLeaf.cache.maxSize == detachedLeaf.cache.maxSize else {
                throw DetachedForkError.verificationFailed(
                    "\(where_): maxSize changed (\(String(describing: sourceLeaf.cache.maxSize)) -> \(String(describing: detachedLeaf.cache.maxSize)))")
            }
            guard sourceLeaf.cache.metaState == detachedLeaf.cache.metaState else {
                throw DetachedForkError.verificationFailed(
                    "\(where_): metaState changed (\(sourceLeaf.cache.metaState) -> \(detachedLeaf.cache.metaState))")
            }

            let sourceArrays = sourceLeaf.cache.state
            let detachedArrays = detachedLeaf.cache.state
            guard sourceArrays.count == detachedArrays.count else {
                throw DetachedForkError.verificationFailed(
                    "\(where_): state array count changed (\(sourceArrays.count) -> \(detachedArrays.count))")
            }

            for index in sourceArrays.indices {
                let sourceArray = sourceArrays[index]
                let detachedArray = detachedArrays[index]
                let label = "\(where_)/state[\(index)]"

                guard sourceArray.shape == detachedArray.shape else {
                    throw DetachedForkError.verificationFailed(
                        "\(label): shape changed (\(sourceArray.shape) -> \(detachedArray.shape))")
                }
                guard sourceArray.dtype == detachedArray.dtype else {
                    throw DetachedForkError.verificationFailed(
                        "\(label): dtype changed (\(sourceArray.dtype) -> \(detachedArray.dtype))")
                }
                guard sourceArray.nbytes == detachedArray.nbytes else {
                    throw DetachedForkError.verificationFailed(
                        "\(label): byte count changed (\(sourceArray.nbytes) -> \(detachedArray.nbytes))")
                }

                guard let sourceAddress = address(of: sourceArray),
                    let detachedAddress = address(of: detachedArray)
                else {
                    throw DetachedForkError.verificationFailed(
                        "\(label): storage is not addressable, so separation cannot be shown")
                }
                guard sourceAddress != detachedAddress else {
                    throw DetachedForkError.verificationFailed(
                        "\(label): still reads the source's storage (address 0x\(String(sourceAddress, radix: 16)))")
                }
                guard valuesMatch(detachedArray, sourceArray) else {
                    throw DetachedForkError.verificationFailed(
                        "\(label): values differ between the source and its copy")
                }

                arrays += 1
                bytes += detachedArray.nbytes
                valuesMatched += 1
                addressesSeparated += 1
            }
        }

        return DetachedForkReport.Section(
            tag: tag, leaves: sourceLeaves.count, arrays: arrays, bytes: bytes,
            valuesMatched: valuesMatched, addressesSeparated: addressesSeparated,
            processedTokenCount: detachedTokenCount, cacheKinds: kinds.sorted())
    }

    // MARK: - Observation

    /// Host-visible address of an array's first element, in bytes.
    ///
    /// Read through `mlx_array_data_uint8`, i.e. the array's actual buffer, and not
    /// through `asData(access: .noCopy).data.baseAddress`. The wrapper looked better
    /// and was wrong: Swift `Data` stores payloads under ~14 bytes inline, so every
    /// 8-byte array reported one and the same address. That made the gate refuse
    /// `QuantizedKVCache` (`.../state[1]: still reads the source's storage`) for a
    /// copy that was in fact separate -- a false alarm only visible because the same
    /// arrays carried four different value digests.
    ///
    /// `.copy` and `.noCopyIfContiguous` were never options: the first makes every
    /// address fresh, the second silently copies a strided view, which is precisely
    /// the shape a forked cache tends to have.
    private static func address(of array: MLXArray) -> UInt? {
        guard array.nbytes > 0 else { return nil }
        array.eval()
        guard let base = mlx_array_data_uint8(array.ctx) else { return nil }
        return UInt(bitPattern: UnsafeRawPointer(base))
    }

    /// Element-wise equality, evaluated.
    ///
    /// This is the check a slice cannot fake: `array[.ellipsis]` reads the very
    /// same bytes, so it passes -- but it is caught by the address check above.
    /// Conversely, a copy that landed the wrong rows passes the address check and
    /// fails here. Both are needed.
    ///
    /// A NaN in the cache would compare unequal and fail the fork closed; that is
    /// the intended reading, since a cache containing NaN is already not a state
    /// worth continuing from.
    private static func valuesMatch(_ lhs: MLXArray, _ rhs: MLXArray) -> Bool {
        guard lhs.shape == rhs.shape, lhs.dtype == rhs.dtype else { return false }
        guard lhs.nbytes > 0 else { return true }
        return all(lhs .== rhs).item(Bool.self)
    }
}
