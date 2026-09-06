// DS4 check commands: the synthetic math self-tests and the tokenizer
// round-trip smoke, both weights-free (GGUF header only), for the
// Tools/ds4_check.sh gate. `DS4SelfTest` and `DS4TokenizerSmoke` carry the
// checks themselves; this is the CLI adapter, in the style of RuntimeCheck.

import ArgumentParser
import Foundation
import Slotstream

/// Sync root (`Slotstream`) + async subcommand is a silent no-op in
/// ArgumentParser, so like TemplateCheck the async work runs on a Task behind
/// a semaphore.
struct DS4Check: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ds4-check",
        abstract: "Run the DeepSeek-V4-Flash math self-tests and the tokenizer smoke (no tensor bytes read)")

    @Option(
        name: .long,
        help: "Model directory holding exactly one deepseek4 GGUF (header only).")
    var model: String?

    func run() throws {
        let sem = DispatchSemaphore(value: 0)
        var err: Error?
        Task {
            do { try await execute() } catch { err = error }
            sem.signal()
        }
        sem.wait()
        if let e = err { throw e }
    }

    func execute() async throws {
        let dir = try Self.resolveModelDir(model)
        let ggufURL = try Engine.ds4GGUF(in: dir)
        guard let ggufURL else {
            throw ModelError("no deepseek4 GGUF found in \(dir.path) — check --model")
        }

        print("DS4 self-tests (synthetic tensors only):")
        guard DS4SelfTest.runAll() else {
            throw ModelError("DS4SelfTest FAILED")
        }

        print("\nDS4 tokenizer smoke (\(ggufURL.lastPathComponent)):")
        let tmp = NSTemporaryDirectory() + "slotstream-ds4-tokenizer-smoke-\(UUID().uuidString)"
        defer { try? FileManager.default.removeItem(atPath: tmp) }
        let report = try await DS4TokenizerSmoke.run(ggufPath: ggufURL.path, modelDir: tmp)
        print(report)

        print("\nDS4 CHECK PASS")
    }

    /// A bare `ds4-check` runs against the first deepseek4 GGUF directory it
    /// can find, so the gate does not need the path spelled out on machines
    /// that keep the model in one of the usual places.
    static func resolveModelDir(_ model: String?) throws -> URL {
        if let model {
            return URL(fileURLWithPath: model).resolvingSymlinksInPath()
        }
        let candidates = [
            URL(fileURLWithPath: "/opt/common/models/text/antirez/deepseek-v4-gguf"),
            ModelLocator.repoLocalDir.deletingLastPathComponent()
                .appendingPathComponent("deepseek-v4-gguf"),
            ModelLocator.userModelsDir.appendingPathComponent("deepseek-v4-gguf"),
        ]
        for c in candidates where (try? Engine.ds4GGUF(in: c)) != nil {
            return c
        }
        throw ModelError(
            "no deepseek4 GGUF directory found — pass --model <dir containing the DeepSeek-V4-Flash GGUF>")
    }
}
