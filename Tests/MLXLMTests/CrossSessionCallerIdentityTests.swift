// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// FORK-HOOK-CALLER-IDENTITY-01 (SimiGo authorization c5c92ecc): the
/// context-aware cross-session hook must receive the CALLING session's own
/// model directory at the consultation point, must be preferred over the
/// legacy slot when installed, and must leave legacy-only behavior intact.
/// Refusals (nil return) must take the ordinary cold path.
final class CrossSessionCallerIdentityTests: XCTestCase {

    /// Strict-prefix boundary used by every seeding/adoption leg: well
    /// inside the rendered test prompt, so adoption always leaves tokens
    /// to prefill.
    private static let boundary = 4

    /// Concurrency-safe recorders (Swift 6: the hook closures are
    /// @Sendable and must not capture the test case).
    private final class ContextRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var origins: [URL?] = []
        func record(_ origin: URL?) {
            lock.lock()
            origins.append(origin)
            lock.unlock()
        }
        var values: [URL?] {
            lock.lock()
            defer { lock.unlock() }
            return origins
        }
    }

    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        func increment() {
            lock.lock()
            count += 1
            lock.unlock()
        }
        var value: Int {
            lock.lock()
            defer { lock.unlock() }
            return count
        }
    }

    private final class SnapshotBox: @unchecked Sendable {
        let snapshot: PromptCacheSnapshot
        init(_ snapshot: PromptCacheSnapshot) { self.snapshot = snapshot }
    }

    private var snapshotURL: URL?

    override func tearDown() async throws {
        ChatSession.crossSessionPrefixLookup = nil
        ChatSession.crossSessionPrefixLookupWithContext = nil
        if let snapshotURL {
            try? FileManager.default.removeItem(at: snapshotURL)
        }
        try await super.tearDown()
    }

    /// A full synthetic model whose configuration carries a LOCAL DIRECTORY
    /// origin — the identity the consultation point must derive.
    private static func directoryModel(_ directory: URL) -> ModelContext {
        let configuration = ModelConfiguration(directory: directory)
        return ChatSessionTests.makeModel(
            processor: TestInputProcessor(
                tokenizer: TestTokenizer(),
                configuration: configuration,
                messageGenerator: DefaultMessageGenerator()),
            configuration: configuration,
            tokenizer: TestTokenizer())
    }

    @discardableResult
    private func respond(
        _ context: ModelContext, prompt: String
    ) async throws -> (text: String, reuseMode: String?) {
        let session = ChatSession(
            ModelContainer(context: context),
            generateParameters: GenerateParameters(maxTokens: 8, temperature: 0))
        var reuseMode: String?
        var text = ""
        for try await generation in session.streamDetails(to: prompt) {
            if let chunk = generation.chunk { text += chunk }
            if let info = generation.info { reuseMode = info.cacheReuseMode }
        }
        return (text, reuseMode)
    }

    /// Seed a durable snapshot through the PUBLIC save/load round trip:
    /// synthetic simple caches for the synthetic model's layers (the
    /// session-private realized caches are not inspectable, and
    /// savePrefixSnapshot requires an all-simple cache which the sliding
    /// synthetic Gemma does not produce). The disk mechanics under test —
    /// save, load, adoption guards — are the real public path.
    private func seedSnapshot(layerCount: Int = 8) throws -> SnapshotBox {
        var caches: [KVCache] = []
        for _ in 0..<layerCount {
            let cache = KVCacheSimple()
            _ = cache.update(
                keys: MLXArray.zeros([1, 4, Self.boundary, 64]),
                values: MLXArray.zeros([1, 4, Self.boundary, 64]))
            caches.append(cache)
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("caller-identity-\(UUID().uuidString).safetensors")
        try savePromptCache(url: url, cache: caches)
        snapshotURL = url
        return SnapshotBox(
            try loadPromptCacheSnapshot(url: url, materializeArrays: true))
    }

    /// The consultation point derives the caller's origin from the calling
    /// session's own configuration — verified by capturing what the context
    /// hook actually receives.
    func testContextHookReceivesCallingSessionDirectory() async throws {
        let directory = URL(fileURLWithPath: "/models/caller-A", isDirectory: true)
        let recorder = ContextRecorder()
        ChatSession.crossSessionPrefixLookupWithContext = { _, caller in
            recorder.record(caller.modelDirectory)
            return nil
        }
        _ = try await respond(Self.directoryModel(directory), prompt: "hello world")
        XCTAssertEqual(recorder.values, [directory])
    }

    /// An unresolved (hub-id) origin arrives as nil: the host decides, and
    /// the fork must not fabricate an origin.
    func testHubOriginCallerArrivesAsNil() async throws {
        let configuration = ModelConfiguration(id: "hub/model")
        let context = ChatSessionTests.makeModel(
            processor: TestInputProcessor(
                tokenizer: TestTokenizer(),
                configuration: configuration,
                messageGenerator: DefaultMessageGenerator()),
            configuration: configuration,
            tokenizer: TestTokenizer())
        let recorder = ContextRecorder()
        ChatSession.crossSessionPrefixLookupWithContext = { _, caller in
            recorder.record(caller.modelDirectory)
            return nil
        }
        _ = try await respond(context, prompt: "hello world")
        XCTAssertEqual(recorder.values, [nil])
    }

    /// A refused context hook (foreign directory) must take the ordinary
    /// cold path: generation completes and the completion carries no
    /// cross-session reuse mode.
    func testForeignDirectoryRefusalTakesColdPath() async throws {
        let foreign = URL(fileURLWithPath: "/models/caller-A", isDirectory: true)
        let cacheModel = URL(fileURLWithPath: "/models/cache-B", isDirectory: true)
        let box = try seedSnapshot()

        ChatSession.crossSessionPrefixLookupWithContext = { _, caller in
            guard let origin = caller.modelDirectory, origin == cacheModel else {
                return nil
            }
            return (box.snapshot, Self.boundary)
        }
        let result = try await respond(
            Self.directoryModel(foreign), prompt: "seed prompt content")
        XCTAssertFalse(result.text.isEmpty)
        XCTAssertNotEqual(result.reuseMode, "cross-session")
    }

    /// Same-directory caller with a matching snapshot adopts: the completion
    /// reports cross-session reuse.
    func testSameDirectoryCallerAdoptsSnapshot() async throws {
        let directory = URL(fileURLWithPath: "/models/caller-A", isDirectory: true)
        let box = try seedSnapshot()

        ChatSession.crossSessionPrefixLookupWithContext = { _, caller in
            guard caller.modelDirectory == directory else { return nil }
            return (box.snapshot, Self.boundary)
        }
        let result = try await respond(
            Self.directoryModel(directory), prompt: "seed prompt content")
        XCTAssertEqual(result.reuseMode, "cross-session")
    }

    /// When both slots are installed, the consultation prefers the context
    /// slot and never calls the legacy one.
    func testContextSlotTakesPrecedenceOverLegacy() async throws {
        let directory = URL(fileURLWithPath: "/models/precedence", isDirectory: true)
        let recorder = ContextRecorder()
        let legacy = Counter()
        ChatSession.crossSessionPrefixLookup = { _ in
            legacy.increment()
            return nil
        }
        ChatSession.crossSessionPrefixLookupWithContext = { _, caller in
            recorder.record(caller.modelDirectory)
            return nil
        }
        _ = try await respond(Self.directoryModel(directory), prompt: "hello world")
        XCTAssertEqual(recorder.values.count, 1)
        XCTAssertEqual(legacy.value, 0)
    }

    /// Legacy-only installation keeps the prior behavior exactly: adoption
    /// through the legacy slot still reports cross-session reuse.
    func testLegacyOnlyInstallationPreservesBehavior() async throws {
        let directory = URL(fileURLWithPath: "/models/legacy", isDirectory: true)
        let box = try seedSnapshot()

        ChatSession.crossSessionPrefixLookup = { _ in
            (box.snapshot, Self.boundary)
        }
        let result = try await respond(
            Self.directoryModel(directory), prompt: "legacy seed prompt")
        XCTAssertEqual(result.reuseMode, "cross-session")
    }
}
