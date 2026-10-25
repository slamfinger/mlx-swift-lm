// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// A NON-HOLDING record of which `Data` object owns each array of a realized KV
/// cache, taken at one moment.
///
/// This is the ownership-level counterpart of ``KVStorageSample``: the sample
/// answers "do these two arrays read the same bytes now", this answers "is the
/// allocation record behind those bytes still there", and it keeps answering
/// after every `MLXArray` that produced it is gone.
///
/// Nothing in a snapshot holds MLX storage. Each entry carries an opaque
/// `MLXStorageObserver` (a `std::weak_ptr` behind a C handle) plus plain
/// metadata; the arrays are read once at capture and immediately dropped.
///
/// What an expired entry proves: the last strong reference to that `Data` is
/// gone. What it does NOT prove:
/// - that the bytes left MLX's allocator -- released buffers go to a reusable
///   pool that MLX still manages;
/// - that the OS reclaimed pages;
/// - anything about an array that was *rebound* after capture. A cache that
///   appends replaces its arrays, so re-capture after a write instead of
///   reading a stale entry as "the child still uses the old storage".
public struct KVOwnershipSnapshot: @unchecked Sendable {

    /// One array's ownership as observed while the array was alive.
    public struct Entry: @unchecked Sendable {
        /// `main`, `draft`, or a caller-supplied label.
        public let tag: String
        /// Stable leaf path through a possibly nested cache topology.
        public let path: [Int]
        /// Runtime class of the leaf cache that owns this array.
        public let cacheKind: String
        /// `key` / `value` for two-array attention leaves, `state[i]` otherwise.
        public let role: String
        /// Address of the `Data` object at capture time. A diagnostic label.
        public let ownerIdentity: UInt
        /// The non-owning observation itself.
        public let observer: MLXStorageObserver

        /// Stable within one snapshot; the identity of an array, not of its
        /// storage. `tag` plus the leaf path plus the role, e.g. `main/3.0/key`
        /// or `raw/array[2]` for bare arrays.
        public var key: String {
            ([tag] + path.map(String.init) + [role]).joined(separator: "/")
        }
        public var isAlive: Bool { observer.isAlive }
        public var strongCount: Int { observer.strongCount }
    }

    public let entries: [Entry]
    /// `processedTokenCount` of the main cache when the snapshot was taken, so
    /// two snapshots can be told apart even when their keys match.
    public let processedTokenCount: Int?

    public var isEmpty: Bool { entries.isEmpty }
    public var aliveKeys: [String] { entries.filter(\.isAlive).map(\.key) }
    public var expiredKeys: [String] { entries.filter { !$0.isAlive }.map(\.key) }
    public var allAlive: Bool { !entries.isEmpty && aliveKeys.count == entries.count }
    public var allExpired: Bool { !entries.isEmpty && expiredKeys.count == entries.count }
    /// Distinct owner identities, alive only. Two entries with one identity are
    /// two views of one allocation.
    public var liveIdentities: Set<UInt> {
        Set(entries.filter(\.isAlive).map(\.ownerIdentity))
    }

    public func entry(_ key: String) -> Entry? {
        entries.first { $0.key == key }
    }

    /// True when the two named arrays are still alive and share one owner.
    public func sharesOwner(_ first: String, _ second: String) -> Bool? {
        guard let a = entry(first)?.observer, let b = entry(second)?.observer else {
            return nil
        }
        return a.sharesOwner(with: b)
    }

    /// Release every observation held by this snapshot. Optional: `deinit` of
    /// each observer does the same. Call it before a measurement that must not
    /// keep `Data` alive -- the observers never do, but an explicit release
    /// makes that failure mode impossible instead of merely intended.
    public func releaseObservers() {
        for entry in entries { entry.observer.release() }
    }

    /// Structured evidence. `allocation_final_reference` is the only claim this
    /// structure supports; `physical_pages` is deliberately reported as
    /// `not_observed` rather than inferred.
    public func summary() -> [String: Any] {
        [
            "entries": entries.count,
            "alive": aliveKeys.count,
            "expired": expiredKeys.count,
            "all_alive": allAlive,
            "all_expired": allExpired,
            "processed_token_count": processedTokenCount.map { $0 as Any } ?? NSNull(),
            "cache_kinds": Array(Set(entries.map(\.cacheKind))).sorted(),
            "live_identities": liveIdentities.count,
            "alive_keys": aliveKeys,
            "expired_keys": expiredKeys,
            "strong_counts": Dictionary(
                entries.map { ($0.key, $0.strongCount) }, uniquingKeysWith: { a, _ in a }),
            "claims": [
                "allocation_final_reference": allExpired ? "observed" : "not_observed",
                "buffer_left_mlx_pool": "not_observed",
                "physical_pages": "not_observed",
            ],
        ]
    }
}

/// The comparison of two snapshots that describe the same arrays at two times.
public struct KVOwnershipOverlap: Sendable {

    public struct Pair: Sendable {
        public let childKey: String
        public let parentKey: String
        public let sharesOwner: Bool
        public let parentAlive: Bool
        public let childAlive: Bool
    }

    public let pairs: [Pair]
    public let onlyInChild: [String]
    public let onlyInParent: [String]
    public let childOwners: Int
    public let parentOwners: Int

    public var shared: Int { pairs.filter(\.sharesOwner).count }
    public var separated: Int { pairs.filter { !$0.sharesOwner }.count }
    public var allShared: Bool { !pairs.isEmpty && shared == pairs.count }
    public var allSeparated: Bool { !pairs.isEmpty && separated == pairs.count }

    public func summary() -> [String: Any] {
        [
            "pairs": pairs.count,
            "shared": shared,
            "separated": separated,
            "all_shared": allShared,
            "all_separated": allSeparated,
            "parent_owners": parentOwners,
            "child_owners": childOwners,
            "only_in_child": onlyInChild,
            "only_in_parent": onlyInParent,
        ]
    }
}

/// Ownership capture and comparison for a realized KV cache.
public enum KVOwnershipProbe {

    /// Observe every array of every cache leaf. Arrays are read once and
    /// dropped; only weak references survive.
    static func snapshot(
        _ cache: [KVCache], tag: String, processedTokenCount: Int?
    ) -> [KVOwnershipSnapshot.Entry] {
        guard !cache.isEmpty else { return [] }
        var entries: [KVOwnershipSnapshot.Entry] = []
        for leaf in KVCacheTree.leaves(in: cache) {
            let arrays = leaf.cache.state
            let kind = String(describing: type(of: leaf.cache))
            for (index, array) in arrays.enumerated() {
                // An unrealized array has no `Data` to observe yet; evaluating
                // first is what makes the observation describe the storage the
                // cache actually holds.
                array.eval()
                guard let observer = StorageOwnership.observe(array) else { continue }
                entries.append(
                    .init(
                        tag: tag, path: leaf.path, cacheKind: kind,
                        role: role(index: index, count: arrays.count, isAttention: leaf.isAttentionCache),
                        ownerIdentity: observer.identity, observer: observer))
            }
        }
        return entries
    }

    static func snapshot(
        main: [KVCache], draft: [KVCache], processedTokenCount: Int?
    ) -> KVOwnershipSnapshot {
        var entries: [KVOwnershipSnapshot.Entry] = []
        entries += snapshot(main, tag: "main", processedTokenCount: processedTokenCount)
        entries += snapshot(draft, tag: "draft", processedTokenCount: processedTokenCount)
        return .init(entries: entries, processedTokenCount: processedTokenCount)
    }

    /// Observe bare arrays, for the probe's own self-check.
    public static func snapshot(
        _ arrays: [MLXArray], tag: String = "raw"
    ) -> KVOwnershipSnapshot {
        let entries = arrays.enumerated().compactMap { index, array -> KVOwnershipSnapshot.Entry? in
            array.eval()
            guard let observer = StorageOwnership.observe(array) else { return nil }
            return .init(
                tag: tag, path: [], cacheKind: "MLXArray", role: "array[\(index)]",
                ownerIdentity: observer.identity, observer: observer)
        }
        return .init(entries: entries, processedTokenCount: nil)
    }

    /// Pair two snapshots by `(tag, path, role)` and ask, per pair, whether the
    /// two observations still share one owner.
    public static func compare(
        parent: KVOwnershipSnapshot, child: KVOwnershipSnapshot
    ) -> KVOwnershipOverlap {
        let childByKey = Dictionary(
            child.entries.map { ($0.key, $0) }, uniquingKeysWith: { a, _ in a })
        var pairs: [KVOwnershipOverlap.Pair] = []
        var onlyInParent: [String] = []
        var seen = Set<String>()
        for entry in parent.entries {
            guard let other = childByKey[entry.key] else {
                onlyInParent.append(entry.key)
                continue
            }
            seen.insert(entry.key)
            pairs.append(
                .init(
                    childKey: entry.key, parentKey: other.key,
                    sharesOwner: entry.observer.sharesOwner(with: other.observer),
                    parentAlive: entry.isAlive, childAlive: other.isAlive))
        }
        let onlyInChild = child.entries.map(\.key).filter { !seen.contains($0) }
        return .init(
            pairs: pairs, onlyInChild: onlyInChild, onlyInParent: onlyInParent,
            childOwners: child.entries.count, parentOwners: parent.entries.count)
    }

    private static func role(index: Int, count: Int, isAttention: Bool) -> String {
        guard isAttention, count == 2 else { return "state[\(index)]" }
        return index == 0 ? "key" : "value"
    }
}
