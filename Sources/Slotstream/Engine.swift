// High-level engine: model + tokenizer + chat templating, shared by CLI/server.

import CoreGraphics
import Foundation
import MLX
import Tokenizers

public struct ChatMessage {
    public var role: String
    public var content: String
    /// An assistant turn's reasoning, rendered as `reasoning_content`. Clients
    /// that keep reasoning in history can replay it; fx does not send any.
    public var reasoning: String?
    /// Calls this assistant turn made.
    public var toolCalls: [ParsedToolCall]
    /// For a `tool` message: which call it answers.
    public var toolCallId: String?
    public var toolName: String?
    /// Pictures this turn carries, as inline bytes (a `data:` URL or bare
    /// base64) in the order the template should render them. Text-only paths
    /// leave it empty and behave exactly as before.
    public var images: [String] = []

    public init(role: String, content: String) {
        self.role = role
        self.content = content
        self.reasoning = nil
        self.toolCalls = []
        self.toolCallId = nil
        self.toolName = nil
    }

    public init(
        role: String, content: String, reasoning: String? = nil,
        toolCalls: [ParsedToolCall] = [], toolCallId: String? = nil, toolName: String? = nil
    ) {
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.toolCallId = toolCallId
        self.toolName = toolName
    }

    /// The dictionary the chat template consumes.
    ///
    /// Tool-call arguments are bridged as an unordered dictionary because
    /// swift-jinja accepts nothing else, so the template's `arguments|items`
    /// follows Swift's hash order. That is why a generated assistant turn is
    /// spliced back as raw ids rather than re-rendered (`PrefixCache`): a
    /// re-render is semantically identical but not byte-identical, and the
    /// prefix cache matches on bytes.
    public var templateValue: [String: any Sendable] {
        var m: [String: any Sendable] = ["role": role, "content": content]
        // The template checks each content part for an `image`/`image_url`
        // key, so a turn with pictures has to arrive as parts rather than a
        // string. Images first, then the text: that is the order the template
        // numbers them in ("Picture 1: ..."), and the order
        // `Engine.imageSources` reads them back in.
        if !images.isEmpty {
            var parts: [[String: any Sendable]] = images.map {
                ["type": "image_url", "image_url": ["url": $0] as [String: any Sendable]]
            }
            if !content.isEmpty { parts.append(["type": "text", "text": content]) }
            m["content"] = parts
        }
        if let r = reasoning, !r.isEmpty { m["reasoning_content"] = r }
        if !toolCalls.isEmpty {
            m["tool_calls"] = toolCalls.map { call in
                [
                    "type": "function",
                    "function": [
                        "name": call.name,
                        "arguments": call.arguments.mapValues { $0.any },
                    ] as [String: any Sendable],
                ] as [String: any Sendable]
            }
        }
        return m
    }
}

/// The one model an Engine runs.
///
/// **Design note — why an enum box and not a protocol.** The alternative was a
/// protocol with associated types (model/state), which would have put
/// existentials on the generate hot path or forced a huge generic rewrite of
/// Generator and PrefixCache. The enum keeps every hot path concrete: the
/// Qwen branch of `Generator.generate` still runs against the concrete
/// `Qwen4ExpModel` (bound once, then a local), and the DS4 branch runs
/// against the concrete `DS4Model`. State never crosses this seam in this
/// cut — DS4 has no prefix cache, no disk tier, and no MTP (see the scope
/// notes in `Engine.init`), so no matching state box exists; when DS4 gains a
/// prefix cache, that is the moment an `EngineState` box earns its place.
public enum EngineModel {
    case qwen(Qwen4ExpModel)
    case ds4(DS4Model)

    /// The shared expert pool (both models stream experts through one).
    public var pool: SlotPool {
        switch self {
        case .qwen(let m): return m.pool
        case .ds4(let m): return m.pool
        }
    }

    /// Routed experts per layer, for the startup banner and /api/show.
    public var expertsPerLayer: Int {
        switch self {
        case .qwen(let m): return m.cfg.numExperts
        case .ds4(let m): return m.cfg.expertCount
        }
    }

    /// The architecture string /api/show reports.
    public var architecture: String {
        switch self {
        case .qwen: return "qwen4_exp"
        case .ds4: return "deepseek4"
        }
    }

    public var hasMTPDraftHead: Bool {
        if case .qwen(let m) = self { return m.mtpHead != nil }
        return false
    }

    public var isQwen: Bool {
        if case .qwen = self { return true }
        return false
    }

    /// The concrete Qwen model. The parity/inspection commands are Qwen-only
    /// tools and use this directly; a DS4 engine makes the misuse loud
    /// instead of quietly returning wrong-shaped state.
    public var qwenModel: Qwen4ExpModel {
        guard case .qwen(let m) = self else {
            fatalError("this command needs the Qwen model; the loaded engine is a DS4 (deepseek4) engine")
        }
        return m
    }

    public var qwen: Qwen4ExpModel? {
        if case .qwen(let m) = self { return m }
        return nil
    }

    public var ds4: DS4Model? {
        if case .ds4(let m) = self { return m }
        return nil
    }
}

/// What /api/show says about a DS4 GGUF beyond its geometry, read once from
/// the header at boot. nil on a Qwen engine.
public struct DS4ModelInfo: Sendable {
    public let name: String?
    public let sizeLabel: String?
    public let sourceRevision: String?
    public let sourceURL: String?
    /// The deepseek4 text-model GGUF this engine runs from.
    public let ggufPath: String
}

public final class Engine {
    public let modelDir: URL
    /// The loaded model: `.qwen` for the pinned Qwen4Exp checkpoint, `.ds4`
    /// for a directory whose GGUF declares `general.architecture == deepseek4`.
    public let model: EngineModel
    /// The model geometry the memory plan and the pool speak in.
    public let profile: GeometryProfile
    /// DS4 header metadata for /api/show; nil on a Qwen engine.
    public let ds4Info: DS4ModelInfo?
    public let generator: Generator
    public let tokenizer: any Tokenizers.Tokenizer
    public let eosIds: Set<Int>
    public let modelName: String
    /// Lazily-loaded vision tower (VLM). Loaded on the first request that
    /// carries an image and then cached; see `ensureVisionTower`.
    public private(set) var visionTower: VisionTower?
    /// Whether this process will accept images at all (`--vision`). False
    /// makes every image request a 400 that says so, rather than a surprise
    /// gigabyte.
    public var visionAllowed = true
    /// Whether the checkpoint carries a tower at all, read once at startup so
    /// the fx catalogue and `/api/show` can answer without touching it.
    public private(set) var visionAvailable = false
    /// Longest prompt accepted, at most `ContextPolicy.maxTokens` (the largest
    /// context that has been measured, see Context.swift). Unbounded prompts
    /// are not free: KV plus indexer state costs ~27 KiB per token, and a
    /// prompt is read in full before the first token, so a huge prompt is a
    /// long, memory-growing stall rather than a fast failure.
    public var maxContextTokens = ContextPolicy.maxTokens {
        didSet {
            let capped = min(prefixCache.maxTokens, maxContextTokens)
            prefixCache.configure(maxTokens: capped)
            // Keep /api/show's memory plan aligned with the allocation control
            // that actually changed; otherwise --max-context 1024 reported the
            // startup cache ceiling even though it had already been reduced.
            if let p = currentPlan, p.prefixCacheTokens != capped || p.maxContextTokens != maxContextTokens {
                updatePlan(MemoryPlan(
                    source: p.source, slots: p.slots, targetGB: p.targetGB,
                    ramGB: p.ramGB, workingSetGB: p.workingSetGB,
                    ramPercent: p.ramPercent, availableGB: p.availableGB,
                    clamped: p.clamped, prefillChunk: p.prefillChunk,
                    prefixCacheTokens: capped, mtpEnabled: p.mtpEnabled,
                    visionEnabled: p.visionEnabled,
                    maxContextTokens: maxContextTokens, notes: p.notes,
                    simulated: p.simulated, profile: p.profile))
            }
        }
    }

    /// Retained conversation state, so a follow-up turn re-prefills only what
    /// is new. See PrefixCache for the extend-only rule and the memory story.
    public let prefixCache: PrefixCache

    /// Release the retained conversation state. Takes the generation lock, so
    /// never call it from inside `generate`.
    public func dropPrefixCache() {
        withExclusive { prefixCache.drop() }
    }

    /// nil when `promptTokens` fits, otherwise the message to return to the client.
    ///
    /// The message names the cap for what it is. It used to tell people to
    /// raise --max-context, which cannot go past the ceiling the server was
    /// already at.
    public func contextError(promptTokens: Int) -> String? {
        guard promptTokens > maxContextTokens else { return nil }
        let wait = PrefillSchedule.describe(seconds: PrefillSchedule.estSeconds(
            tokens: promptTokens, maxChunk: generator.prefillChunk, profile: profile))
        let ceiling = maxContextTokens < ContextPolicy.maxTokens
            ? "this server was started with --max-context \(maxContextTokens); "
                + "the ceiling is \(ContextPolicy.maxTokens)"
            : "\(ContextPolicy.maxTokens) is the largest context slotstream has measured, "
                + "not a memory limit (context state costs ~"
                + String(
                    format: "%.0f KiB per token)",
                    Double(profile.contextBytesPerToken) / 1024)
        return "prompt is \(promptTokens) tokens, over this server's limit of "
            + "\(maxContextTokens) for prompt plus reply. \(ceiling). Reading a prompt "
            + "this long would take ~\(wait) before the first token here. Send less, or "
            + "split the material across turns of one conversation so each follow-up "
            + "reads only what is new."
    }
    /// The live memory plan (updated by the elastic governor on resize; nil
    /// for internal fixed-size uses). Guarded by its own lock so /api reads
    /// never block behind a running generation.
    private var _plan: MemoryPlan?
    private let planLock = NSLock()
    public var currentPlan: MemoryPlan? {
        planLock.lock()
        defer { planLock.unlock() }
        return _plan
    }
    public func updatePlan(_ p: MemoryPlan) {
        planLock.lock()
        _plan = p
        planLock.unlock()
    }

    private let lock = NSLock()

    /// Run `body` with the generation lock held — the governor uses this to
    /// resize the pool strictly between requests.
    public func withExclusive<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    /// Pool numbers for the metadata endpoints, published rather than read
    /// live. Reading SlotPool's mutable Swift arrays while the governor
    /// resizes is a data race, but taking the *generation* lock to avoid it
    /// made /api/tags and /api/ps block for the whole of a running request, so
    /// a client that polls either one saw a generating server as a hung one.
    private var _poolSnapshot: (slots: Int, slotsPerLayer: Double, poolBytes: Int) = (0, 0, 0)
    private let poolSnapshotLock = NSLock()

    public func poolSnapshot() -> (slots: Int, slotsPerLayer: Double, poolBytes: Int) {
        poolSnapshotLock.lock()
        defer { poolSnapshotLock.unlock() }
        return _poolSnapshot
    }

    /// Re-read the pool and publish it. **Call with the generation lock held**
    /// (inside `withExclusive`), which is where every resize already happens.
    public func publishPoolSnapshot() {
        let s = (model.pool.slots, model.pool.slotsPerLayer, model.pool.poolBytes)
        poolSnapshotLock.lock()
        _poolSnapshot = s
        poolSnapshotLock.unlock()
    }

    public convenience init(modelDir: URL, plan: MemoryPlan) async throws {
        try await self.init(modelDir: modelDir, poolSlots: plan.slots, plan: plan)
    }

    public init(modelDir: URL, poolSlots: Int, plan: MemoryPlan? = nil) async throws {
        // A plan made for a simulated machine may be printed and compared,
        // never loaded. Simulating memory the machine does not have still
        // allocates for real: on 2026-08-30 a simulated 60 GB drove a 25.4 GB
        // allocation and 39 GB of swap. The flag travels on the plan so this
        // cannot be forgotten at a call site.
        if plan?.simulated == true { throw SlotstreamError.simulatedDeviceCannotLoad }
        self.modelDir = modelDir
        self._plan = plan
        // Boot milestones wrap the whole init; the per-action pairs below nest
        // inside, so a polled log shows both where boot is and how long it all
        // took. A throw mid-boot leaves the last START in the log — which is
        // exactly where boot died.
        let tBoot = Milestone.start("engine ready")
        // Model dispatch: a directory whose GGUF declares the deepseek4
        // architecture builds the DS4 path (config → weights → expert store →
        // pool, all from the GGUF); everything else is the pinned Qwen
        // checkpoint exactly as before. Sidecar GGUFs in the same directory
        // (the vision encoder, the DSpark head) declare other architectures
        // and are tolerated; more than one deepseek4 text model is an error.
        let ds4URL = try Self.ds4GGUF(in: modelDir)
        self.profile = ds4URL != nil ? .ds4 : .qwen
        // Sized from the same budget as the pool; SLOTSTREAM_PREFIX_CACHE=0
        // (or --no-prefix-cache) pins it off for parity work.
        //
        // DS4 first cut: the conversation prefix cache is NOT built for DS4.
        // PrefixCache holds `Qwen4ExpModel.State`, and a DS4 state box does
        // not exist yet; wiring DS4State in without a prefix-check
        // equivalent would trade a measured invariant for an unmeasured one.
        // The cache is therefore constructed disabled and `generate` hands
        // the generator nil, so no DS4 state is ever stored or reused.
        // Follow-up: an EngineState box + a DS4 prefix-check gate.
        let env = ProcessInfo.processInfo.environment["SLOTSTREAM_PREFIX_CACHE"]
        switch profile.kind {
        case .ds4:
            self.prefixCache = PrefixCache(maxTokens: 0, enabled: false)
        case .qwen:
            self.prefixCache = PrefixCache(
                maxTokens: plan?.prefixCacheTokens
                    ?? Planner.prefixCacheTokensFor(
                        poolBudgetGB: Geometry.gb(poolSlots)),
                enabled: env != "0")
        }
        // MLX's allocator otherwise retains freed transients (KV caches,
        // activations) in an unbounded internal cache — measured ~5 GB of RSS
        // above the memory plan after a few dozen requests. 2 GB keeps
        // per-token reallocation churn away while making real process memory
        // track the announced plan.
        MLX.Memory.cacheLimit = 2 << 30
        let t0 = Date()
        if let ggufURL = ds4URL {
            // ---- DS4 (DeepSeek-V4-Flash from GGUF)
            self.modelName = "deepseek-v4-flash:gguf"
            let boot = try Self.bootDS4(modelDir: modelDir, ggufPath: ggufURL.path, poolSlots: poolSlots)
            self.model = .ds4(boot.model)
            self.ds4Info = boot.info
            // MTP/speculation is not built for DS4 in this cut. auto never
            // gets here (the GGUF dir has no mtp.safetensors), so this only
            // fires for an explicit --mtp on, which is refused rather than
            // silently ignored.
            if plan?.mtpEnabled == true {
                throw PlanError(
                    "--mtp is not supported for deepseek-v4-flash:gguf — no draft head is "
                        + "built for this architecture yet; use --mtp off/auto")
            }
            self.generator = Generator(model: self.model)
            if let p = plan, ProcessInfo.processInfo.environment["SLOTSTREAM_PREFILL_CHUNK"] == nil {
                generator.prefillChunk = p.prefillChunk
            }
            if let mb = Int(ProcessInfo.processInfo.environment["SLOTSTREAM_PREFILL_CACHE_MB"] ?? "") {
                generator.prefillCacheLimit = max(0, mb) << 20
            } else if let p = plan, p.expectedPeakGB <= 12 {
                generator.prefillCacheLimit = 512 << 20
            }
            // One milestone covers the GGUF tokenizer export (or its reuse
            // check) and the swift-transformers load: to a poller it is one
            // wait, and the boot struct no longer needs to carry the folder.
            let tTokenizer = Milestone.start("load tokenizer")
            let tokenizerDir = try Self.ensureDS4Tokenizer(
                modelDir: modelDir, gguf: boot.gguf, cfg: boot.cfg)
            self.tokenizer = try await AutoTokenizer.from(
                modelFolder: URL(fileURLWithPath: tokenizerDir))
            Milestone.end("load tokenizer", tTokenizer)
            var eos: Set<Int> = []
            if let e = boot.gguf.kv("tokenizer.ggml.eos_token_id")?.intValue { eos.insert(e) }
            if let e = tokenizer.eosTokenId { eos.insert(e) }
            self.eosIds = eos
            // The vision sidecar is not supported in this cut: the variant is
            // named vision-exp and the encoder is a separate GGUF this engine
            // does not load, so an image request must be refused clearly
            // rather than answered wrong.
            self.visionAvailable = false
            self.visionAllowed = false
        } else {
            // ---- Qwen (the pinned checkpoint; this path is unchanged)
            self.modelName = "qwen3.8-flash-next:4bit"
            self.ds4Info = nil
            let tIndex = Milestone.start("read checkpoint index")
            let index = try CheckpointIndex(dir: modelDir)
            Milestone.end("read checkpoint index", tIndex)
            let qwen = try Qwen4ExpModel(index: index, poolSlots: poolSlots)
            self.model = .qwen(qwen)
            try qwen.validate()
            // Read from the index that is already open — no tensor is touched, and
            // nothing is allocated until an image actually arrives.
            self.visionAvailable = VisionTower.present(index: index)
            self.visionAllowed = plan?.visionEnabled ?? visionAvailable
            if plan?.mtpEnabled == true {
                try qwen.enableMTP(modelDir: modelDir)
            }
            self.generator = Generator(model: self.model)
            if let p = plan, ProcessInfo.processInfo.environment["SLOTSTREAM_PREFILL_CHUNK"] == nil {
                generator.prefillChunk = p.prefillChunk
            }
            if let mb = Int(ProcessInfo.processInfo.environment["SLOTSTREAM_PREFILL_CACHE_MB"] ?? "") {
                generator.prefillCacheLimit = max(0, mb) << 20
            } else if let p = plan, p.expectedPeakGB <= 12 {
                generator.prefillCacheLimit = 512 << 20
            }
            let tTokenizer = Milestone.start("load tokenizer")
            self.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
            Milestone.end("load tokenizer", tTokenizer)
            var eos: Set<Int> = [index.config.eosTokenId]
            if let e = tokenizer.eosTokenId { eos.insert(e) }
            // generation_config may list several
            if let d = try? Data(contentsOf: modelDir.appendingPathComponent("generation_config.json")),
                let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any]
            {
                if let list = o["eos_token_id"] as? [Int] { list.forEach { eos.insert($0) } }
                if let one = o["eos_token_id"] as? Int { eos.insert(one) }
            }
            self.eosIds = eos
        }
        publishPoolSnapshot()
        Milestone.end("engine ready", tBoot)
        let banner = "engine ready in \(String(format: "%.1f", -t0.timeIntervalSinceNow))s: "
            + "expert cache ~\(String(format: "%.0f", model.pool.slotsPerLayer))/\(model.expertsPerLayer) per layer "
            + "(\(model.pool.slots) global slots = \(String(format: "%.1f", Double(model.pool.poolBytes) / 1e9)) GB), "
            + (model.hasMTPDraftHead ? "mtp draft head on, " : "")
            + "eos \(eosIds.sorted())\n"
        FileHandle.standardError.write(banner.data(using: .utf8)!)
    }

    // MARK: model dispatch helpers

    /// The deepseek4 text-model GGUF in `dir`, if any. Sidecars (vision
    /// encoder, DSpark head) declare other `general.architecture` values and
    /// are skipped; two deepseek4 GGUFs in one directory is an error because
    /// "pick one" would be a guess.
    public static func ds4GGUF(in dir: URL) throws -> URL? {
        let fm = FileManager.default
        let entries: [URL]
        do {
            entries = try fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        } catch {
            // Not a listable directory: nothing to find here. The Qwen path
            // produces the "no model at ..." error the caller expects.
            return nil
        }
        var matches: [URL] = []
        for e in entries where e.pathExtension.lowercased() == "gguf" {
            guard let gguf = try? GGUFFile(path: e.path),
                gguf.kv("general.architecture")?.stringValue == DS4Config.prefix
            else { continue }
            matches.append(e)
        }
        switch matches.count {
        case 0: return nil
        case 1: return matches[0]
        default:
            throw ModelError(
                "\(dir.path) holds \(matches.count) deepseek4 GGUF files — pass a directory "
                    + "with exactly one text model: "
                    + matches.map { ($0 as NSURL).lastPathComponent ?? $0.path }.joined(separator: ", "))
        }
    }

    /// Everything the DS4 boot needs, built without touching `self` (the init
    /// is async and must not read partially initialized state). The tokenizer
    /// export lives with the tokenizer load in the init's own milestone, so
    /// the boot carries the config the export needs instead of a folder.
    private struct DS4Boot {
        let model: DS4Model
        let info: DS4ModelInfo
        let cfg: DS4Config
        let gguf: GGUFFile
    }

    private static func bootDS4(modelDir: URL, ggufPath: String, poolSlots: Int) throws -> DS4Boot {
        let total = GeometryProfile.ds4.totalRecords
        guard poolSlots >= 1, poolSlots <= total else {
            throw ModelError(
                "expert-pool slot count must be between 1 and \(total), got \(poolSlots)")
        }
        // Header + metadata only; tensor bytes are read by the loaders below.
        let gguf = try GGUFFile(path: ggufPath)
        // The Geometry.check analog: DS4Config.init validates the exact Flash
        // geometry (and, with the gguf argument, the tensor directory) and
        // rejects anything else with a --model-style message.
        let tConfig = Milestone.start("ds4 config")
        let cfg = try DS4Config(gguf: gguf)
        Milestone.end("ds4 config", tConfig)
        // Neither loader prints a byte total up front, so the trunk milestone
        // carries no size (a static "8.8 GB" would be a guess).
        let tTrunk = Milestone.start("load resident trunk")
        let weights = try DS4Weights(ggufPath: ggufPath, cfg: cfg)
        Milestone.end("load resident trunk", tTrunk)
        let tExperts = Milestone.start("open expert store")
        let experts = try DS4ExpertStore(ggufPath: ggufPath, cfg: cfg)
        Milestone.end("open expert store", tExperts)
        // The pool target the plan chose, in the same figure the banner prints.
        let poolGB = String(format: "%.1f", Double(poolSlots) * Double(experts.recordBytes) / 1e9)
        let tPool = Milestone.start("build slot pool (\(poolGB) GB)")
        let pool = SlotPool(slots: poolSlots, source: .ds4(experts))
        Milestone.end("build slot pool (\(poolGB) GB)", tPool)
        let model = try DS4Model(cfg: cfg, weights: weights, experts: experts, pool: pool)
        let info = DS4ModelInfo(
            name: gguf.kv("general.name")?.stringValue,
            sizeLabel: gguf.kv("general.size_label")?.stringValue,
            sourceRevision: gguf.kv("general.source.revision")?.stringValue,
            sourceURL: gguf.kv("general.source.url")?.stringValue,
            ggufPath: ggufPath)
        return DS4Boot(model: model, info: info, cfg: cfg, gguf: gguf)
    }

    /// The folder swift-transformers loads the DS4 tokenizer from.
    ///
    /// swift-transformers needs a folder of HF files, and the weights live in
    /// a GGUF (and may sit on a read-only volume), so the tokenizer is
    /// exported from the GGUF metadata into `~/.slotstream/ds4/<dir>/`. A
    /// previous export is reused only when all three files exist and the
    /// exported vocabulary still matches the GGUF's token count — the cheap
    /// validity check that keeps a stale or partial export from serving.
    private static func ensureDS4Tokenizer(modelDir: URL, gguf: GGUFFile, cfg: DS4Config) throws
        -> String
    {
        let home = ModelLocator.home
        let safe = modelDir.lastPathComponent.map {
            $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" || $0 == "." ? $0 : "_"
        }
        let dir = home.appendingPathComponent(".slotstream/ds4/\(String(safe))/tokenizer")
        let path = dir.path
        let fm = FileManager.default
        let files = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
        var reusable = files.allSatisfy {
            ((try? fm.attributesOfItem(atPath: path + "/" + $0))?[.size] as? Int ?? 0) > 0
        }
        if reusable {
            // The vocab-count check: parse tokenizer.json and compare its
            // vocab size with the GGUF. A mismatched export would tokenize
            // prompts into ids the model cannot read.
            if let data = fm.contents(atPath: path + "/tokenizer.json"),
                let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let model = obj["model"] as? [String: Any],
                let vocab = model["vocab"] as? [String: Any]
            {
                reusable = vocab.count == cfg.vocabSize
            } else {
                reusable = false
            }
        }
        if !reusable {
            try exportTokenizer(from: gguf, to: path)
        }
        return path
    }

    public func encodeChat(_ messages: [ChatMessage], thinking: Bool) throws -> [Int] {
        try encodeChat(messages, tools: [], thinking: thinking, effort: nil)
    }

    /// Render a conversation that may declare tools and replay tool calls.
    ///
    /// `tools` empty renders no `<tools>` block at all, which is what the
    /// Ollama and OpenAI dialects pass, so their bytes are unchanged.
    public func encodeChat(
        _ messages: [ChatMessage], tools: [ToolDefinition], thinking: Bool, effort: String?
    ) throws -> [Int] {
        try tokenizer.applyChatTemplate(
            messages: messages.map { $0.templateValue },
            tools: tools.isEmpty ? nil : tools.map { $0.templateValue },
            additionalContext: Self.additionalContext(thinking: thinking, effort: effort))
    }

    /// Encode a conversation, substituting the exact ids this server generated
    /// for any assistant turn it can still prove it produced.
    ///
    /// Why this exists. The prefix cache matches on bytes, and it must: the GDN
    /// recurrent state is a fold over the tokens it consumed, with no inverse,
    /// so a state may only be extended by the very ids that built it. A client
    /// replaying history does not send those ids — it sends its own view of the
    /// turn, which the template then re-renders. Whenever that re-render
    /// differs by a single byte, the next turn rebuilds the whole prompt.
    ///
    /// With reasoning ON that is not an edge case, it is every turn: fx (and
    /// most clients) never echo reasoning back, so the re-render is missing the
    /// `<think>` block the model actually produced, and the state cannot match.
    /// Measured on this machine, a two-turn tool loop reused 303 of 325 tokens
    /// with reasoning off and 0 of 349 with it on — three and a half times the
    /// wall time for the identical second turn.
    ///
    /// The splice closes that. For each assistant turn, ask the cache whether it
    /// still holds a state whose ids begin with exactly the prompt that turn was
    /// generated from; if it does, the remainder of those ids *is* that turn,
    /// verbatim. Check that the remainder really describes the turn the client
    /// sent (same calls, same arguments, same text) and then use the held ids in
    /// place of the re-render, tokenizing only the conversation after it.
    ///
    /// Splitting the text at `<|im_end|>` is safe because it is an added token
    /// and therefore a hard tokenizer boundary: the suffix tokenizes identically
    /// whether or not the text before it is present. That is measured, not
    /// assumed — see the `chat-splice` check.
    ///
    /// Any mismatch anywhere falls back to the plain render, which is the
    /// behaviour that existed before. The splice can make a turn cheaper; it can
    /// never make one wrong.
    public func encodeChatSpliced(
        _ messages: [ChatMessage], tools: [ToolDefinition], thinking: Bool, effort: String?
    ) throws -> [Int] {
        let full = try encodeChat(messages, tools: tools, thinking: thinking, effort: effort)
        // The splice replays ids the prefix cache held, which only exists for
        // Qwen in this cut (see Engine.init); DS4 takes the plain render.
        guard case .qwen = model, prefixCache.enabled,
            messages.contains(where: { $0.role == "assistant" })
        else { return full }
        let fullText = tokenizer.decode(tokens: full, skipSpecialTokens: false)

        var spliced: [Int] = []  // ids exactly as the model saw or produced them
        var consumed = 0  // characters of fullText those ids already cover
        var didSplice = false

        func index(_ offset: Int) -> String.Index {
            fullText.index(fullText.startIndex, offsetBy: offset)
        }

        for k in messages.indices where messages[k].role == "assistant" {
            guard
                let headIds = try? encodeChat(
                    Array(messages[0..<k]), tools: tools, thinking: thinking, effort: effort)
            else { break }
            let headText = tokenizer.decode(tokens: headIds, skipSpecialTokens: false)
            guard fullText.hasPrefix(headText), headText.count >= consumed else { break }
            // The ids that produced turn k: what is already spliced, plus the
            // conversation between there and this turn's generation prompt.
            let bridge = String(fullText[index(consumed)..<index(headText.count)])
            let producer =
                spliced + (bridge.isEmpty ? [] : tokenizer.encode(text: bridge, addSpecialTokens: false))
            guard let entry = prefixCache.peek(extending: producer) else { break }
            let generated = Array(entry[producer.count...])
            let genText = tokenizer.decode(tokens: generated, skipSpecialTokens: false)
            guard Self.spliceDescribes(genText, messages[k], tools: tools) else { break }
            guard
                let end = fullText.range(
                    of: "<|im_end|>", range: index(headText.count)..<fullText.endIndex)
            else { break }
            spliced = entry
            consumed = fullText.distance(from: fullText.startIndex, to: end.lowerBound)
            didSplice = true
        }

        guard didSplice else { return full }
        let tail = String(fullText[index(consumed)...])
        return spliced + tokenizer.encode(text: tail, addSpecialTokens: false)
    }

    /// Does this generated text describe the assistant turn the client sent?
    ///
    /// Deliberately compares meaning rather than bytes: the client's copy has
    /// been through its own JSON round trip, so whitespace and argument order
    /// may differ, but the calls it reports must be the calls that were made.
    /// Reasoning is ignored — the client dropping it is the whole reason the
    /// splice is needed.
    public static func spliceDescribes(
        _ generated: String, _ message: ChatMessage, tools: [ToolDefinition]
    ) -> Bool {
        let (_, body) = ThinkSplitter.split(generated)
        let visible = body.isEmpty && !generated.contains("</think>") ? generated : body
        let events = ToolCallSplitter.parseAll(visible, tools: tools.map { $0.schema })
        var calls: [ParsedToolCall] = []
        var text = ""
        for e in events {
            switch e {
            case .toolCall(let c): calls.append(c)
            case .text(let t): text += t
            case .malformed: return false
            default: break
            }
        }
        guard calls.count == message.toolCalls.count else { return false }
        for (a, b) in zip(calls, message.toolCalls) {
            guard a.name == b.name, a.arguments == b.arguments else { return false }
        }
        // The text is compared after trimming only. A client that rewrites the
        // assistant's prose is describing a different turn, and re-rendering it
        // is then the correct answer.
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
            == message.content.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func additionalContext(thinking: Bool, effort: String?) -> [String: any Sendable] {
        var ctx: [String: any Sendable] = ["enable_thinking": thinking]
        if let e = effort, thinking { ctx["reasoning_effort"] = e }
        return ctx
    }

    /// Render a template without constructing the multi-GB model. Installer
    /// and API acceptance checks run this while a server is already live; the
    /// old implementation built a second Engine merely to load the tokenizer,
    /// so the singleton guard correctly rejected the check it was meant to run.
    public static func encodeChatWithoutModel(
        modelDir: URL, messages: [ChatMessage], thinking: Bool,
        tools: [ToolDefinition] = [], effort: String? = nil
    ) async throws -> [Int] {
        // A DS4 directory keeps its tokenizer inside a GGUF, so the same
        // export-or-reuse path the engine boot uses points the loader at the
        // exported folder.
        let folder: URL
        if let gguf = try ds4GGUF(in: modelDir) {
            let ggufFile = try GGUFFile(path: gguf.path)
            let cfg = try DS4Config(gguf: ggufFile)
            folder = URL(fileURLWithPath: try ensureDS4Tokenizer(
                modelDir: modelDir, gguf: ggufFile, cfg: cfg))
        } else {
            folder = modelDir
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: folder)
        return try tokenizer.applyChatTemplate(
            messages: messages.map { $0.templateValue },
            tools: tools.isEmpty ? nil : tools.map { $0.templateValue },
            additionalContext: additionalContext(thinking: thinking, effort: effort))
    }

    /// OpenAI path: messages already contain image_url parts, and content may
    /// be String or [[String: Any]] (vision). The nested arrays must be
    /// bridged to the tokenizer's `[String: any Sendable]` messages or the
    /// vision parts are silently dropped before the Jinja template can render
    /// them as <|image_pad|>.
    public func encodeChatOpenAI(
        messages: [[String: Any]], tools: [[String: Any]]?, thinking: Bool = false
    ) throws -> [Int] {
        func toSendable(_ v: Any) -> any Sendable {
            if let arr = v as? [[String: Any]] {
                return arr.map { d -> [String: any Sendable] in
                    var out: [String: any Sendable] = [:]
                    for (k, vv) in d { out[k] = toSendable(vv) }
                    return out
                } as any Sendable
            }
            if let d = v as? [String: Any] {
                var out: [String: any Sendable] = [:]
                for (k, vv) in d { out[k] = toSendable(vv) }
                return out as any Sendable
            }
            if let a = v as? [Any] {
                return a.map { toSendable($0) } as any Sendable
            }
            return v as any Sendable
        }
        let msgs: [[String: any Sendable]] = messages.map { dict in
            var m: [String: any Sendable] = [:]
            for (k, v) in dict { m[k] = toSendable(v) }
            return m
        }
        let toolSpecs: [[String: any Sendable]]? = tools?.map { dict in
            var t: [String: any Sendable] = [:]
            for (k, v) in dict { t[k] = toSendable(v) }
            return t
        }
        return try tokenizer.applyChatTemplate(
            messages: msgs, tools: toolSpecs, additionalContext: ["enable_thinking": thinking])
    }

    // MARK: Vision

    /// Load the vision tower on first use, and only if the machine can spare
    /// it right now.
    ///
    /// **Under the generation lock, not a lock of its own.** Loading is
    /// ~0.9 GB of MLX arrays plus an `eval`; a private lock let that run on a
    /// connection thread while another request was mid-generation, which is
    /// exactly the concurrent GPU work every other allocation path in this
    /// file serializes. `withExclusive` is that serialization, and it also
    /// makes the availability reading below meaningful: nothing else can
    /// allocate between reading it and taking the memory.
    ///
    /// The check itself is the promise from `Planner.visionResidentGB` kept.
    /// The plan the user was shown does not include the tower, so the tower
    /// may only load when it genuinely fits on top; a busy machine gets a
    /// refusal it can act on instead of a gigabyte of swap.
    public func ensureVisionTower() throws -> VisionTower {
        if let vt = visionTower { return vt }
        // The DS4 variant is "vision-exp" and ships its tower as a separate
        // GGUF this engine does not load. Refuse with that sentence rather
        // than the generic --vision-off one, so nobody goes hunting a flag.
        guard case .qwen = model else {
            throw SlotstreamError.vision(
                "deepseek-v4-flash:gguf is text-only in this build — the vision sidecar GGUF "
                    + "is not supported; send text only")
        }
        guard visionAllowed else {
            throw SlotstreamError.vision(
                "this server was started with --vision off; images are not accepted")
        }
        return try withExclusive {
            if let vt = visionTower { return vt }
            let idx = try CheckpointIndex(dir: modelDir)
            guard VisionTower.present(index: idx) else {
                throw SlotstreamError.vision(
                    "this checkpoint has no vision tower — it is a text-only model")
            }
            let needGB = Double(VisionTower.residentBytes(index: idx)) / 1e9
            if let avail = Planner.deviceAvailableGB(), avail.isFinite,
                avail < needGB + Planner.visionLoadMarginGB
            {
                throw SlotstreamError.vision(
                    String(
                        format: "the vision tower needs %.1f GB and only %.1f GB is reclaimable "
                            + "right now — close other apps and retry, or restart with a lower "
                            + "--memory-gb so the tower fits",
                        needGB, avail))
            }
            let vt = try VisionTower(index: idx)
            self.visionTower = vt
            return vt
        }
    }

    /// Tokenize with vision expansion: each template image_pad is worth
    /// N_merged real tokens, so the template's single pad is expanded to a run
    /// of pads that the tower's embeddings will fill. Returns the expanded ids
    /// and a `VisionPrompt` when the request carries images, nil otherwise.
    ///
    /// The tower does not run here. The run lengths come from each image's
    /// dimensions, so the ids — and with them the prefix cache key — are ready
    /// before any pixels are read. `Generator.generate` asks the cache first
    /// and then encodes only the images that the reused state does not cover.
    public func encodeWithVision(
        messages: [[String: Any]], tools: [[String: Any]]?, thinking: Bool = false
    ) throws -> ([Int], VisionPrompt?) {
        let baseIds = try encodeChatOpenAI(messages: messages, tools: tools, thinking: thinking)
        return try withImages(baseIds: baseIds, sources: Self.imageSources(in: messages))
    }

    /// The typed path (`ChatMessage`), for the fx gateway and the CLI. Renders
    /// through the same template as `encodeChat` and then expands the same
    /// placeholders.
    public func encodeChatWithVision(
        _ messages: [ChatMessage], tools: [ToolDefinition] = [], thinking: Bool = false,
        effort: String? = nil
    ) throws -> ([Int], VisionPrompt?) {
        let baseIds = try encodeChat(messages, tools: tools, thinking: thinking, effort: effort)
        return try withImages(baseIds: baseIds, sources: messages.flatMap { $0.images })
    }

    /// Expand each `<|image_pad|>` the template rendered into the run of
    /// placeholders its image is worth, and describe the images for the tower
    /// and the prefix cache. Shared by every surface so they cannot drift.
    private func withImages(baseIds: [Int], sources: [String]) throws -> ([Int], VisionPrompt?) {
        if sources.isEmpty { return (baseIds, nil) }
        // Decode and hash first: it needs no tower, it is cheap next to one,
        // and a malformed picture should be a 400 before the process commits
        // 0.9 GB to a tower it may not otherwise need.
        var decoded: [(cg: CGImage, hash: ImageHash)] = []
        decoded.reserveCapacity(sources.count)
        for (i, source) in sources.enumerated() {
            do {
                let data = try VisionPreprocess.loadImageData(from: source)
                decoded.append((try VisionPreprocess.decodeCGImage(data), ImageHash(hashing: data)))
            } catch {
                throw SlotstreamError.vision("image \(i + 1): \(error)")
            }
        }
        let vt = try ensureVisionTower()
        var items: [VisionPrompt.Item] = []
        items.reserveCapacity(decoded.count)
        for (i, d) in decoded.enumerated() {
            do { items.append(VisionPrompt.Item(image: d.cg, plan: try vt.plan(for: d.cg))) }
            catch { throw SlotstreamError.vision("image \(i + 1): \(error)") }
        }
        // The template renders one `<|image_pad|>` per image; the tower
        // produces `mergedTokens` rows for it. Expanding the pad into a run of
        // that length is what makes the two line up, and it moves every token
        // after the first image — ids and segment offsets alike, in one sweep,
        // so a later prompt that extends this one keys identically.
        // Unreachable for DS4 — ensureVisionTower threw above — but the
        // image-token config only exists on Qwen, so the shape says so.
        guard let qcfg = model.qwen?.cfg else {
            throw SlotstreamError.vision("vision is not supported by this model")
        }
        let imageId = qcfg.imageTokenId
        let perImage = items.map { $0.plan.mergedTokens }
        var expanded: [Int] = []
        var segments: [ImageSegment] = []
        expanded.reserveCapacity(baseIds.count + perImage.reduce(0, +) - perImage.count)
        var imgIdx = 0
        for tok in baseIds {
            if tok == imageId, imgIdx < perImage.count {
                segments.append(
                    ImageSegment(
                        start: expanded.count, count: perImage[imgIdx], hash: decoded[imgIdx].hash))
                expanded.append(contentsOf: repeatElement(imageId, count: perImage[imgIdx]))
                imgIdx += 1
            } else {
                expanded.append(tok)
            }
        }
        // Both directions are checked. Too few placeholders means the template
        // did not render an image this code found; too many means something
        // else in the prompt tokenized to the placeholder id — a user who
        // typed the literal `<|image_pad|>`, for instance. Either way the rows
        // and the runs would not correspond, so the request stops here rather
        // than putting embeddings under the wrong tokens.
        guard imgIdx == items.count else {
            throw SlotstreamError.vision(
                "the chat template rendered \(imgIdx) image placeholders for \(items.count) "
                    + "images; slotstream cannot place the rest")
        }
        let placeholders = expanded.reduce(0) { $0 + ($1 == imageId ? 1 : 0) }
        guard placeholders == perImage.reduce(0, +) else {
            throw SlotstreamError.vision(
                "the prompt carries \(placeholders) image placeholder tokens but the images "
                    + "account for \(perImage.reduce(0, +)); remove any literal <|image_pad|> "
                    + "from the text")
        }
        return (
            expanded,
            VisionPrompt(
                tower: vt, items: items, segments: segments, hiddenSize: qcfg.hiddenSize)
        )
    }

    /// Every image a request carries, in the order the chat template will
    /// render them: message by message, part by part, and Ollama's per-message
    /// `images` array after that message's content parts — which is where the
    /// template puts them too.
    public static func imageSources(in messages: [[String: Any]]) -> [String] {
        var out: [String] = []
        for m in messages {
            if let content = m["content"] as? [[String: Any]] {
                for part in content {
                    if let iu = part["image_url"] as? [String: Any], let u = iu["url"] as? String {
                        out.append(u)
                    } else if let u = part["image_url"] as? String {
                        out.append(u)
                    } else if let u = part["image"] as? String {
                        out.append(u)
                    }
                }
            }
            for b64 in (m["images"] as? [String] ?? []) { out.append(b64) }
        }
        return out
    }

    /// Earliest position at which any stop sequence occurs, or nil.
    private static func stopIndex(_ text: String, _ stops: [String]) -> String.Index? {
        var best: String.Index?
        for s in stops {
            if let r = text.range(of: s), best == nil || r.lowerBound < best! {
                best = r.lowerBound
            }
        }
        return best
    }

    /// Serialized generation (single-flight; callers queue on the lock).
    ///
    /// Incremental detokenization consumes bounded groups of token ids, keeping
    /// incomplete UTF-8 bytes at the group boundary. Two rules matter:
    ///
    /// - Emission and stop holdback are by Unicode scalar, never Character. A
    ///   later token can contribute a scalar that merges into the grapheme
    ///   already sent (an emoji plus U+FE0F is still one Character).
    /// - While stop sequences are active, the last `maxStopLength - 1` scalars
    ///   are withheld, so the prefix of a stop sequence that straddles a token
    ///   boundary is never emitted before the rest of it arrives. Whatever is
    ///   still held back is flushed once generation ends.
    ///
    /// The invariant the tests hold this to: concatenating every streamed delta
    /// reproduces the non-streamed text exactly.
    public func generate(
        promptIds: [Int], params: SampleParams, vision: VisionPrompt? = nil,
        shouldContinue: (() -> Bool)? = nil,
        onToken: ((Int, String) -> Bool)? = nil,
        onNewToken: ((Int) -> Void)? = nil
    ) -> (text: String, ids: [Int], stats: GenStats) {
        lock.lock()
        defer { lock.unlock() }
        var params = params.sanitized()
        let room = max(0, maxContextTokens - promptIds.count)
        if room == 0 {
            var stats = GenStats()
            stats.promptTokens = promptIds.count
            stats.finishReason = "length"
            stats.peakMemoryGB = ProcessMemory.peakResidentGB
            return ("", [], stats)
        }
        // Context is prompt + completion, not two independent 32k allowances.
        params.maxTokens = min(params.maxTokens, room)
        let stops = params.stop
        let holdBack = stops.isEmpty
            ? 0 : max(0, (stops.map { $0.unicodeScalars.count }.max() ?? 1) - 1)
        var pendingIds: [Int] = []
        var withheld = ""
        var delivered = ""
        var lastTok = -1
        var clientGone = false
        var stopFound = false

        func emit(_ delta: String, _ tok: Int) -> Bool {
            if delta.isEmpty { return true }
            delivered += delta
            guard let cb = onToken else { return true }
            return cb(tok, delta)
        }

        /// Feed a stable decoded piece through the stop-sequence holdback.
        func feed(_ piece: String, final: Bool, tok: Int) -> Bool {
            withheld += piece
            if !stops.isEmpty, let cut = Self.stopIndex(withheld, stops) {
                _ = emit(String(withheld[..<cut]), tok)
                withheld = ""
                stopFound = true
                return false
            }
            let scalars = withheld.unicodeScalars
            let n = final ? scalars.count : max(0, scalars.count - holdBack)
            let delta = String(String.UnicodeScalarView(scalars.prefix(n)))
            withheld = String(String.UnicodeScalarView(scalars.dropFirst(n)))
            return emit(delta, tok)
        }

        /// Qwen's ByteLevel decoder is concatenative once a UTF-8 scalar is
        /// complete. Decode small bounded groups and retain four token bytes at
        /// the boundary; if the candidate still ends in U+FFFD, retain more.
        /// This makes streaming decode O(n), rather than decoding tokens 1...n
        /// after every generated token.
        func flushStablePrefix(_ tok: Int) -> Bool {
            guard !pendingIds.isEmpty else { return true }
            // Start from everything buffered and hand back one token at a time
            // while the decode still ends mid-scalar. Waiting for eight tokens
            // before the first flush and holding four back after it gave
            // clients one delta per four tokens, and no delta at all for a
            // reply shorter than eight; the byte-exactness this protects rests
            // on the replacement-character check below, not on the backlog.
            var n = pendingIds.count
            var piece = ""
            while n > 0 {
                piece = tokenizer.decode(
                    tokens: Array(pendingIds.prefix(n)), skipSpecialTokens: true)
                if !piece.hasSuffix("\u{FFFD}") { break }
                n -= 1
            }
            guard n > 0 else { return true }
            pendingIds.removeFirst(n)
            return feed(piece, final: false, tok: tok)
        }

        let needsIncrementalDecode = onToken != nil || onNewToken != nil || !stops.isEmpty
        let tokenHandler: ((Int) -> Bool)? = needsIncrementalDecode ? { tok in
            lastTok = tok
            // Exactly one event per sampled token, before detokenization can
            // buffer it or merge it into a later delta — `onToken` sees
            // deltas, which do not map one-to-one onto tokens.
            onNewToken?(tok)
            pendingIds.append(tok)
            let ok = flushStablePrefix(tok)
            if !ok, !stopFound { clientGone = true }
            return ok
        } : nil

        // No prefix cache for DS4 in this cut (see Engine.init): the cache
        // would never hold a DS4State, so the generator is handed nil rather
        // than a cache it could only ever miss on.
        let (ids, stats) = generator.generate(
            promptIds: promptIds, params: params, eosIds: eosIds,
            cache: model.isQwen ? prefixCache : nil,
            vision: vision,
            shouldContinue: {
                guard !clientGone, !stopFound else { return false }
                return shouldContinue?() ?? true
            }, onToken: tokenHandler)

        var text = tokenizer.decode(tokens: ids, skipSpecialTokens: true)
        if !stops.isEmpty, let cut = Self.stopIndex(text, stops) {
            text = String(text[text.startIndex ..< cut])
        }
        // The one full decode is both the non-streamed result and an exact final
        // reconciliation for the bounded incremental decoder.
        if !clientGone, onToken != nil {
            let target = text.unicodeScalars
            let sent = delivered.unicodeScalars
            if target.count >= sent.count, target.starts(with: sent) {
                _ = emit(String(String.UnicodeScalarView(target.dropFirst(sent.count))), lastTok)
            }
        }
        return (text, ids, stats)
    }
}
