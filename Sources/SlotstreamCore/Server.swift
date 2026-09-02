// Minimal dependency-free HTTP/1.1 server exposing the Ollama API surface
// (/api/*) plus the OpenAI-compatible /v1/chat/completions. Localhost,
// single-flight generation, chunked streaming (NDJSON for /api, SSE for /v1).

import CoreFoundation
import Foundation
import MLX

public struct ServerError: Error, CustomStringConvertible {
    public let description: String
    public init(_ s: String) { description = s }
}

public final class Server {
    let engine: Engine
    let port: UInt16
    /// Total on-disk size of the weights, reported by /api/tags and /api/ps.
    /// Supplied by the caller so it can come from the pinned manifest rather
    /// than a second hand-maintained copy of the number.
    let weightsBytes: Int
    var listenFD: Int32 = -1
    /// When set, every generation appends one JSON line here with the raw
    /// prompt and the raw model response, before any processing.
    let outputPath: String?
    private let outputQueue = DispatchQueue(label: "slotstream.rawlog")
    /// Touched only from `outputQueue`.
    private var outputFile: FileHandle?

    public init(
        engine: Engine, port: UInt16, weightsBytes: Int = 0, listenFD: Int32 = -1,
        outputPath: String? = nil
    ) {
        self.engine = engine
        self.port = port
        self.weightsBytes = weightsBytes
        self.listenFD = listenFD
        self.outputPath = outputPath
    }

    /// Claim the port. Callers bind *before* loading the model so "address
    /// already in use" — running `serve` twice is the common case — costs a
    /// second and one sentence instead of a full load and a fatalError.
    public static func bindPort(_ port: UInt16) throws -> Int32 {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw ServerError("cannot create a socket: \(String(cString: strerror(errno)))")
        }
        var yes: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        let rc = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard rc == 0 else {
            let why = String(cString: strerror(errno))
            close(fd)
            throw ServerError(
                "cannot listen on 127.0.0.1:\(port): \(why)"
                    + (errno == EADDRINUSE
                        ? " — another slotstream (or Ollama) is already there; "
                            + "stop it or pass --port" : ""))
        }
        return fd
    }

    /// Bounded so a client that opens sockets and never speaks cannot exhaust
    /// the thread pool (each connection costs one thread).
    static let maxConcurrentConnections = 8
    private let connSlots = DispatchSemaphore(value: maxConcurrentConnections)

    public func run() throws -> Never {
        // A client that disappears mid-stream makes write() raise SIGPIPE, whose
        // default action kills the process. Ignoring it turns that into EPIPE,
        // which is what `send` already handles by returning false.
        signal(SIGPIPE, SIG_IGN)
        if listenFD < 0 { listenFD = try Self.bindPort(port) }
        listen(listenFD, 16)
        print("slotstream listening on http://127.0.0.1:\(port)")
        print("""
        try it:
          curl localhost:\(port)/api/chat -d '{"model": "\(engine.modelName)", "messages": [{"role": "user", "content": "hello"}]}'
        or point any Ollama or OpenAI client at http://localhost:\(port)
        """)
        fflush(stdout)  // visible immediately even when stdout is a file/pipe
        while true {
            let fd = accept(listenFD, nil, nil)
            if fd < 0 { continue }
            // A stalled client must not pin a thread forever: give reads a
            // deadline, and cap how many connections can be in flight.
            var tv = timeval(tv_sec: 30, tv_usec: 0)
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
            var st = timeval(tv_sec: 120, tv_usec: 0)
            setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &st, socklen_t(MemoryLayout<timeval>.size))
            var one: Int32 = 1
            setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
            connSlots.wait()
            Thread.detachNewThread { [weak self] in
                defer { self?.connSlots.signal() }
                self?.handle(fd)
            }
        }
    }

    // MARK: connection handling

    struct Request {
        var method = ""
        var path = ""
        var body = Data()
        var headers: [String: String] = [:]
    }

    /// Largest request body accepted. Prompts are text; anything past this is
    /// a mistake or an attack, and reading it unbounded is how a local process
    /// gets OOM-killed.
    static let maxBodyBytes = 4 << 20

    private func readRequest(_ fd: Int32) -> Request? {
        var buf = Data()
        var tmp = [UInt8](repeating: 0, count: 65536)
        var headerEnd: Range<Data.Index>? = nil
        while headerEnd == nil {
            let n = read(fd, &tmp, tmp.count)
            if n <= 0 { return nil }
            buf.append(contentsOf: tmp[0 ..< n])
            headerEnd = buf.range(of: Data("\r\n\r\n".utf8))
            if headerEnd == nil, buf.count > 64 << 10 { return nil }
        }
        let headData = buf[..<headerEnd!.lowerBound]
        let head = String(data: headData, encoding: .utf8) ?? ""
        var req = Request()
        let lines = head.split(separator: "\r\n", omittingEmptySubsequences: false)
        let parts = lines.first?.split(separator: " ") ?? []
        if parts.count >= 2 {
            req.method = String(parts[0])
            req.path = String(parts[1])
        }
        var contentLength = 0
        for l in lines.dropFirst() {
            let kv = l.split(separator: ":", maxSplits: 1)
            if kv.count == 2 {
                let key = kv[0].trimmingCharacters(in: .whitespaces).lowercased()
                let value = kv[1].trimmingCharacters(in: .whitespaces)
                req.headers[key] = value
                if key == "content-length" { contentLength = Int(value) ?? -1 }
            }
        }
        if contentLength < 0 || contentLength > Self.maxBodyBytes { return nil }
        var body = Data(buf[headerEnd!.upperBound...].prefix(contentLength))
        while body.count < contentLength {
            let n = read(fd, &tmp, min(tmp.count, contentLength - body.count))
            if n <= 0 { break }
            body.append(contentsOf: tmp[0 ..< n])
            if body.count > Self.maxBodyBytes { return nil }
        }
        guard body.count == contentLength else { return nil }
        req.body = body
        return req
    }

    private func send(_ fd: Int32, _ data: Data) -> Bool {
        var sent = 0
        return data.withUnsafeBytes { raw -> Bool in
            while sent < data.count {
                let n = write(fd, raw.baseAddress! + sent, data.count - sent)
                if n < 0 && errno == EINTR { continue }
                if n <= 0 { return false }
                sent += n
            }
            return true
        }
    }

    /// Browser clients need CORS, but a wildcard turns any website the user
    /// visits into an unauthenticated caller of this expensive local service.
    /// Echo only loopback origins, including their arbitrary development port.
    private static func corsHeaders(origin: String?) -> String? {
        guard let origin, !origin.isEmpty else { return "" }
        guard let u = URL(string: origin), let host = u.host?.lowercased(),
            u.scheme == "http" || u.scheme == "https",
            host == "localhost" || host == "127.0.0.1" || host == "0.0.0.0" || host == "::1"
        else { return nil }
        return "Access-Control-Allow-Origin: \(origin)\r\nVary: Origin\r\n"
    }

    private func respondJSON(
        _ fd: Int32, _ obj: Any, status: String = "200 OK", cors: String = ""
    ) {
        let body = (try? JSONSerialization.data(withJSONObject: obj)) ?? Data("{}".utf8)
        var head = "HTTP/1.1 \(status)\r\nContent-Type: application/json\r\n" + cors
        head += "Content-Length: \(body.count)\r\nConnection: close\r\n\r\n"
        _ = send(fd, Data(head.utf8) + body)
    }

    private func startChunked(_ fd: Int32, contentType: String, cors: String) -> Bool {
        let head = "HTTP/1.1 200 OK\r\nContent-Type: \(contentType)\r\n" + cors
            + "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        return send(fd, Data(head.utf8))
    }

    @discardableResult
    private func chunk(_ fd: Int32, _ payload: Data) -> Bool {
        var d = Data(String(format: "%x\r\n", payload.count).utf8)
        d += payload
        d += Data("\r\n".utf8)
        return send(fd, d)
    }

    private func endChunked(_ fd: Int32) {
        _ = send(fd, Data("0\r\n\r\n".utf8))
    }

    /// A nonblocking peek distinguishes an idle connected peer (EAGAIN) from
    /// EOF. Generation checks it before every prefill chunk and decode token,
    /// including non-streaming requests that otherwise would not write until
    /// all work had already been done.
    private func peerAlive(_ fd: Int32) -> Bool {
        var byte: UInt8 = 0
        let n = recv(fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT)
        if n == 0 { return false }
        if n > 0 { return true }
        return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR
    }

    // MARK: routing

    private func handle(_ fd: Int32) {
        defer { close(fd) }
        guard let req = readRequest(fd) else { return }
        // Raw log: the request body exactly as received, before any parsing,
        // validation, or templating. Only when --output was given.
        var logId = ""
        if outputPath != nil, !req.body.isEmpty {
            logId = "req-\(UUID().uuidString.prefix(8))"
            logRequest(
                id: logId, method: req.method, path: req.path,
                body: String(data: req.body, encoding: .utf8)
                    ?? "<non-utf8 \(req.body.count) bytes>")
        }
        let origin = req.headers["origin"]
        guard let cors = Self.corsHeaders(origin: origin) else {
            respondJSON(
                fd, ["error": "browser origin is not allowed"],
                status: "403 Forbidden")
            return
        }
        if req.method == "OPTIONS" {  // CORS preflight
            let head = "HTTP/1.1 204 No Content\r\n" + cors
                + "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
                + "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                + "Access-Control-Allow-Private-Network: true\r\n"
                + "Access-Control-Max-Age: 86400\r\nConnection: close\r\n\r\n"
            _ = send(fd, Data(head.utf8))
            return
        }
        let parsed = (try? JSONSerialization.jsonObject(with: req.body)) as? [String: Any]
        if req.method == "POST", !req.body.isEmpty, parsed == nil {
            respondJSON(
                fd, ["error": "invalid JSON body"], status: "400 Bad Request", cors: cors)
            return
        }
        let json = parsed ?? [:]
        // Heat matrix: handle with or without ?full=1 query
        let pathBase = String(req.path.split(separator: "?").first ?? Substring(req.path))
        if req.method == "GET" && (pathBase == "/api/heat" || pathBase == "/v1/heat") {
            let heat = engine.withExclusive { engine.model.pool.heatSparse() }
            let wantDense = req.path.contains("full")
            var resp: [String: Any] = [
                "layers": 48, "experts_per_layer": 512,
                "sparse": heat,
                "total_accesses": heat.reduce(0) { $0 + $1[2] },
                "slots": engine.withExclusive { engine.model.pool.slots },
            ]
            if wantDense {
                let dense = engine.withExclusive { engine.model.pool.heatMatrix() }
                resp["dense"] = dense
            }
            respondJSON(fd, resp, cors: cors)
            return
        }
        switch (req.method, req.path) {
        case ("GET", "/api/version"):
            respondJSON(fd, ["version": SlotstreamBuild.version], cors: cors)
        case ("GET", "/api/tags"), ("GET", "/api/tags/"):
            respondJSON(fd, ["models": [modelCard()]], cors: cors)
        case ("GET", "/api/ps"):
            respondJSON(fd, ["models": [modelCard(loaded: true)]], cors: cors)
        case ("POST", "/api/show"):
            if let e = modelError(json) {
                respondJSON(fd, ["error": e], status: "404 Not Found", cors: cors)
                return
            }
            if let e = Self.unsupportedKey(json, allowed: ["model", "verbose"]) {
                respondJSON(fd, ["error": e], status: "400 Bad Request", cors: cors)
                return
            }
            if json["verbose"] != nil, Self.bool(json["verbose"]) == nil {
                respondJSON(
                    fd, ["error": "verbose must be true or false"],
                    status: "400 Bad Request", cors: cors)
                return
            }
            respondJSON(
                fd,
                [
                    "modelfile": "# slotstream: SSD-streamed qwen4_exp",
                    "parameters": "",
                    "template": "{{ .Prompt }}",
                    "details": modelDetails(live: true),
                    "model_info": [
                        "general.architecture": "qwen4_exp",
                        "general.parameter_count": 176_000_000_000,
                    ],
                ], cors: cors)
        case ("POST", "/api/chat"):
            apiChat(fd, json, cors: cors, logId: logId)
        case ("POST", "/api/generate"):
            apiGenerate(fd, json, cors: cors, logId: logId)
        case ("POST", "/v1/chat/completions"):
            v1Chat(fd, json, cors: cors, logId: logId)
        case ("GET", "/v1/models"):
            respondJSON(
                fd,
                [
                    "object": "list",
                    "data": [["id": engine.modelName, "object": "model", "owned_by": "slotstream"]],
                ], cors: cors)
        case ("POST", "/api/embed"), ("POST", "/api/embeddings"):
            respondJSON(
                fd, ["error": "model does not support embeddings"],
                status: "400 Bad Request", cors: cors)
        case ("POST", "/api/pull"), ("POST", "/api/create"):
            respondJSON(
                fd, ["error": "use `slotstream pull` on the host"],
                status: "501 Not Implemented", cors: cors)
        case ("HEAD", _):
            // A HEAD response carries headers only; sending a body is a protocol error.
            let head = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n" + cors
                + "Content-Length: 0\r\nConnection: close\r\n\r\n"
            _ = send(fd, Data(head.utf8))
        case ("GET", "/"):
            respondJSON(fd, ["status": "ok", "engine": "slotstream"], cors: cors)
        default:
            respondJSON(
                fd, ["error": "not found: \(req.method) \(req.path)"],
                status: "404 Not Found", cors: cors)
        }
    }

    /// `live: false` for /api/tags and /api/ps, which are *listings* — a
    /// client may cache or diff them, so per-request counters do not belong
    /// there. /api/show is the endpoint that reports runtime state.
    private func modelDetails(live: Bool = false) -> [String: Any] {
        let pool = engine.poolSnapshot()
        var d: [String: Any] = [
            "format": "safetensors", "family": "qwen4_exp",
            "parameter_size": "176B-A6B", "quantization_level": "4bit",
            "expert_cache_per_layer": Int(pool.slotsPerLayer.rounded()),
            "experts_per_layer": engine.model.cfg.numExperts,
        ]
        if let plan = engine.currentPlan { d["memory_plan"] = plan.json() }
        if live { d["prefix_cache"] = engine.prefixCache.json() }
        return d
    }

    private func modelCard(loaded: Bool = false) -> [String: Any] {
        let pool = engine.poolSnapshot()
        var c: [String: Any] = [
            "name": engine.modelName, "model": engine.modelName,
            "modified_at": iso(Date()), "size": weightsBytes,
            "digest": "slotstream-qwen38-flash-next-4bit",
            "details": modelDetails(),
        ]
        if loaded {
            c["expires_at"] = iso(Date().addingTimeInterval(3600))
            c["size_vram"] = pool.poolBytes
        }
        return c
    }

    private func iso(_ d: Date) -> String {
        ISO8601DateFormatter().string(from: d)
    }

    // MARK: raw request/response log (--output)

    /// Append one JSON line per event to the --output file:
    ///   {"ts","id","kind":"request","method","path","body"}  — the raw POST body verbatim, before any parsing or validation
    ///   {"ts","id","kind":"delta","seq","tok","text"}        — one per decoded text piece, as the model emits it
    ///   {"ts","id","kind":"response","endpoint","response"}  — the assembled raw model text at the end of generation
    /// The `id` associates a request with its deltas and response. Delta text
    /// and the final response are the raw model output exactly as decoded —
    /// before tool-call parsing, reasoning splitting, or any other
    /// handler-level processing. Best-effort: a logging failure must never
    /// fail the request being logged.
    private func logRawLine(_ entry: [String: Any]) {
        guard outputPath != nil else { return }
        guard let line = try? JSONSerialization.data(withJSONObject: entry) else { return }
        outputQueue.sync {
            if outputFile == nil {
                if let path = outputPath,
                    !FileManager.default.fileExists(atPath: path)
                {
                    FileManager.default.createFile(atPath: path, contents: nil)
                }
                outputFile = outputPath.flatMap { try? FileHandle(forWritingTo: URL(fileURLWithPath: $0)) }
            }
            guard let fh = outputFile else { return }
            fh.seekToEndOfFile()
            fh.write(line)
            fh.write(Data("\n".utf8))
        }
    }

    func logRequest(id: String, method: String, path: String, body: String?) {
        guard outputPath != nil else { return }
        logRawLine([
            "ts": iso(Date()), "id": id, "kind": "request",
            "method": method, "path": path, "body": body as Any,
        ])
    }

    func logDelta(id: String, seq: Int, tok: Int, text: String) {
        guard outputPath != nil else { return }
        logRawLine([
            "ts": iso(Date()), "id": id, "kind": "delta",
            "seq": seq, "tok": tok, "text": text,
        ])
    }

    func logResponse(
        id: String, endpoint: String, response: String,
        finish: String? = nil, stopTokenId: Int? = nil, stopTokenHex: String? = nil
    ) {
        guard outputPath != nil else { return }
        var entry: [String: Any] = [
            "ts": iso(Date()), "id": id, "kind": "response",
            "endpoint": endpoint, "response": response,
        ]
        if let finish = finish { entry["finish"] = finish }
        if let stopTokenId = stopTokenId { entry["stop_token_id"] = stopTokenId }
        if let stopTokenHex = stopTokenHex { entry["stop_token_hex"] = stopTokenHex }
        logRawLine(entry)
    }

    // MARK: params

    /// OpenAI clients may send `content` as a string or as an array of typed
    /// parts. Taking `as? String` alone silently drops the whole message, so
    /// the text parts are joined here instead.
    static func contentText(_ v: Any?) -> String {
        if let s = v as? String { return s }
        if let parts = v as? [[String: Any]] {
            return parts.compactMap { part -> String? in
                if let t = part["text"] as? String { return t }
                return nil
            }.joined()
        }
        return ""
    }

    private static func messages(_ json: [String: Any]) -> [ChatMessage] {
        (json["messages"] as? [[String: Any]] ?? []).map {
            ChatMessage(role: $0["role"] as? String ?? "user", content: contentText($0["content"]))
        }
    }

    /// JSON numbers arrive as NSNumber; accept ints where a float is expected.
    private static func num(_ v: Any?) -> Double? {
        guard let n = v as? NSNumber,
            CFGetTypeID(n) != CFBooleanGetTypeID()
        else { return nil }
        let d = n.doubleValue
        // Every current numeric API field is ultimately stored as Float.
        // Accepting 1e300 only to turn it into infinity made a syntactically
        // valid request silently select a different sampler mode.
        guard d.isFinite, Float(d).isFinite else { return nil }
        return d
    }

    /// Swift's NSNumber bridge reports JSON `1` as `is Bool`, and JSON `true`
    /// as `as? Int == 1`. CoreFoundation's type id is the only reliable way to
    /// keep JSON booleans, integers, and ordinary numbers distinct here.
    private static func int(_ v: Any?) -> Int? {
        guard let n = v as? NSNumber,
            CFGetTypeID(n) != CFBooleanGetTypeID()
        else { return nil }
        return v as? Int
    }

    private static func bool(_ v: Any?) -> Bool? {
        guard let n = v as? NSNumber,
            CFGetTypeID(n) == CFBooleanGetTypeID()
        else { return nil }
        return n.boolValue
    }

    private static func stopList(_ v: Any?) -> [String]? {
        if let a = v as? [String] { return a.filter { !$0.isEmpty } }
        if let s = v as? String, !s.isEmpty { return [s] }
        return nil
    }

    private func modelError(_ json: [String: Any]) -> String? {
        guard let raw = json["model"] else { return nil }
        guard let requested = raw as? String else { return "model must be text" }
        guard !requested.isEmpty else { return "model must not be empty" }
        return nil
    }

    private func openAIModelError(_ json: [String: Any]) -> String? {
        guard let raw = json["model"] else { return nil }
        guard let s = raw as? String, !s.isEmpty else { return "model must be text" }
        return nil // permissive: any non-empty model id is accepted for OpenAI compat (opencode sends provider/model)
    }

    private static func unsupportedKey(
        _ json: [String: Any], allowed: Set<String>
    ) -> String? {
        let extras = Set(json.keys).subtracting(allowed).sorted()
        return extras.isEmpty ? nil
            : "unsupported request field(s): \(extras.joined(separator: ", "))"
    }

    private static func messageError(_ json: [String: Any]) -> String? {
        // Ollama path: now accepts images (vision) alongside text
        guard let raw = json["messages"] as? [[String: Any]] else {
            return "messages must be an array"
        }
        for (i, m) in raw.enumerated() {
            let extra = Set(m.keys).subtracting(["role", "content", "images", "tool_calls", "tool_call_id"])
            if !extra.isEmpty {
                return "messages[\(i)] has unsupported field(s): "
                    + extra.sorted().joined(separator: ", ")
            }
            if m["tool_calls"] != nil || m["tool_call_id"] != nil {
                return "messages[\(i)] uses tools, which this server does not support on /api (use /v1)"
            }
            if let images = m["images"] {
                guard images is [String] || images is [Any] else {
                    return "messages[\(i)].images must be an array of base64 strings"
                }
            }
            guard let role = m["role"] as? String,
                ["system", "user", "assistant"].contains(role)
            else { return "messages[\(i)].role must be system, user, or assistant" }
            if let parts = m["content"] as? [[String: Any]] {
                for (j, part) in parts.enumerated() {
                    let extra = Set(part.keys).subtracting(["type", "text", "image_url", "image"])
                    if !extra.isEmpty {
                        return "messages[\(i)].content[\(j)] has unsupported field(s): "
                            + extra.sorted().joined(separator: ", ")
                    }
                    let kind = (part["type"] as? String) ?? "text"
                    if kind == "text" || kind == "input_text" {
                        if part["text"] as? String == nil {
                            return "messages[\(i)] has a text part without text"
                        }
                    } else if kind == "image_url" || kind == "image" {
                        continue
                    } else if part["image_url"] != nil || part["image"] != nil {
                        continue
                    } else {
                        return "messages[\(i)] contains unsupported content type '\(kind)'"
                    }
                }
            } else if m["content"] as? String == nil {
                return "messages[\(i)].content must be text"
            }
        }
        return nil
    }

    /// OpenAI path: permissive — allows tool_calls, tool role, and tool_call_id
    private static func openAIMessageError(_ json: [String: Any]) -> String? {
        guard let raw = json["messages"] as? [[String: Any]] else {
            return "messages must be an array"
        }
        for (i, m) in raw.enumerated() {
            let allowedKeys: Set<String> = ["role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content"]
            let extra = Set(m.keys).subtracting(allowedKeys)
            if !extra.isEmpty {
                return "messages[\(i)] has unsupported field(s): "
                    + extra.sorted().joined(separator: ", ")
            }
            if m["images"] != nil {
                return "messages[\(i)] uses images, which this server does not support"
            }
            guard let role = m["role"] as? String else {
                return "messages[\(i)].role must be a string"
            }
            guard ["system", "user", "assistant", "tool", "developer"].contains(role) else {
                return "messages[\(i)].role must be system, user, assistant, tool, or developer"
            }
            if role == "tool" {
                // tool response: content must be text, tool_call_id recommended
                if let c = m["content"], !(c is String) { return "messages[\(i)].content must be text" }
                continue
            }
            // assistant may have tool_calls and null content
            if let tc = m["tool_calls"] {
                guard tc is [[String: Any]] || tc is [Any] else {
                    return "messages[\(i)].tool_calls must be an array"
                }
            }
            if let content = m["content"] {
                if content is NSNull { continue }  // allowed when tool_calls present
                if content is String { continue }
                if let parts = content as? [[String: Any]] {
                    for (j, part) in parts.enumerated() {
                        let extra = Set(part.keys).subtracting(["type", "text", "image_url"])
                        if !extra.isEmpty {
                            return "messages[\(i)].content[\(j)] has unsupported field(s): "
                                + extra.sorted().joined(separator: ", ")
                        }
                        let kind = (part["type"] as? String) ?? "text"
                        if kind != "text" && kind != "input_text" && kind != "image_url" {
                            // image_url ignored gracefully — not supported but don't trap
                            continue
                        }
                        if kind == "text" || kind == "input_text" {
                            if part["text"] as? String == nil {
                                return "messages[\(i)] has a text part without text"
                            }
                        }
                    }
                } else {
                    return "messages[\(i)].content must be text or array"
                }
            }
        }
        return nil
    }

    private static func optionsError(_ json: [String: Any]) -> String? {
        guard let options = json["options"] else { return nil }
        guard let o = options as? [String: Any] else { return "options must be an object" }
        let allowed: Set<String> = [
            "temperature", "top_p", "top_k", "min_p", "presence_penalty",
            "num_predict", "seed", "stop",
        ]
        let extras = Set(o.keys).subtracting(allowed).sorted()
        if !extras.isEmpty {
            return "unsupported options field(s): \(extras.joined(separator: ", "))"
        }
        for key in ["temperature", "top_p", "min_p", "presence_penalty"]
        where o[key] != nil && num(o[key]) == nil {
            return "options.\(key) must be a number"
        }
        for key in ["top_k", "num_predict", "seed"]
        where o[key] != nil && int(o[key]) == nil {
            return "options.\(key) must be an integer"
        }
        if let stop = o["stop"], !(stop is String) && !(stop is [String]) {
            return "options.stop must be text or an array of text"
        }
        return nil
    }

    private func ollamaValidationError(
        _ json: [String: Any], allowed: Set<String>, messages: Bool = false
    ) -> String? {
        if let e = modelError(json) { return e }
        if let e = Self.unsupportedKey(json, allowed: allowed) { return e }
        if let e = Self.optionsError(json) { return e }
        if json["think"] != nil, Self.bool(json["think"]) == nil {
            return "think must be true or false; named reasoning levels are not supported"
        }
        if json["stream"] != nil, Self.bool(json["stream"]) == nil {
            return "stream must be true or false"
        }
        return messages ? Self.messageError(json) : nil
    }

    private func openAIValidationError(_ json: [String: Any]) -> String? {
        if let e = openAIModelError(json) { return e }
        let allowed: Set<String> = [
            "model", "messages", "stream", "temperature", "top_p", "top_k",
            "presence_penalty", "frequency_penalty", "max_tokens", "max_completion_tokens",
            "seed", "stop", "stream_options",
            "tools", "tool_choice", "parallel_tool_calls",
            "response_format", "n", "user", "logit_bias", "logprobs", "top_logprobs",
            "reasoning_effort", "verbosity",
            "enable_thinking",
        ]
        if let e = Self.unsupportedKey(json, allowed: allowed) { return e }
        if let e = Self.openAIMessageError(json) { return e }
        for key in ["temperature", "top_p", "presence_penalty"]
        where json[key] != nil && Self.num(json[key]) == nil {
            return "\(key) must be a number"
        }
        for key in ["top_k", "max_tokens", "max_completion_tokens", "seed"]
        where json[key] != nil && Self.int(json[key]) == nil {
            return "\(key) must be an integer"
        }
        for key in ["max_tokens", "max_completion_tokens"] {
            if let value = Self.int(json[key]), value <= 0 {
                return "\(key) must be greater than zero"
            }
        }
        if let stop = json["stop"], !(stop is String) && !(stop is [String]) {
            return "stop must be text or an array of text"
        }
        if json["stream"] != nil, Self.bool(json["stream"]) == nil {
            return "stream must be true or false"
        }
        if json["enable_thinking"] != nil, Self.bool(json["enable_thinking"]) == nil {
            return "enable_thinking must be true or false"
        }
        // Qwen's chat template only recognises xhigh/medium/low. Reject the
        // OpenAI "high" level explicitly so a client mistake surfaces as a 400
        // rather than a tokenizer exception mid-prefill.
        if let r = json["reasoning_effort"] {
            guard let s = r as? String else { return "reasoning_effort must be a string" }
            if !["low", "medium", "xhigh"].contains(s) {
                return "reasoning_effort must be one of: low, medium, xhigh"
            }
        }
        if json["stream_options"] != nil,
            json["stream_options"] as? [String: Any] == nil
        {
            return "stream_options must be an object"
        }
        if let o = json["stream_options"] as? [String: Any] {
            let extras = Set(o.keys).subtracting(["include_usage"])
            if !extras.isEmpty {
                return "unsupported stream_options field(s): \(extras.sorted().joined(separator: ", "))"
            }
            if o["include_usage"] != nil, Self.bool(o["include_usage"]) == nil {
                return "stream_options.include_usage must be true or false"
            }
        }
        if let tools = json["tools"], !(tools is [[String: Any]]) && !(tools is [Any]) {
            return "tools must be an array"
        }
        if let tc = json["tool_choice"] {
            if !(tc is String) && !(tc is [String: Any]) {
                return "tool_choice must be a string or object"
            }
        }
        if json["parallel_tool_calls"] != nil, Self.bool(json["parallel_tool_calls"]) == nil {
            return "parallel_tool_calls must be true or false"
        }
        return nil
    }

    // MARK: OpenAI helpers

    private static func openAITools(_ json: [String: Any]) -> [[String: Any]]? {
        guard let arr = json["tools"] as? [[String: Any]] else { return nil }
        return arr
    }

    private static func openAIMessagesForTemplate(_ json: [String: Any]) -> [[String: Any]] {
        guard let raw = json["messages"] as? [[String: Any]] else { return [] }
        return raw.map { m in
            var out: [String: Any] = [:]
            out["role"] = m["role"] as? String ?? "user"
            // Preserve structured content for vision (array with image_url) — tokenizer's
            // render_content handles it and emits <|vision_start|><|image_pad|><|vision_end|>
            if let c = m["content"] {
                if c is NSNull { out["content"] = "" }
                else if let s = c as? String { out["content"] = s }
                else if let parts = c as? [[String: Any]] {
                    let hasImage = parts.contains { $0["image_url"] != nil || $0["image"] != nil || ($0["type"] as? String) == "image_url" || ($0["type"] as? String) == "image" }
                    out["content"] = hasImage ? parts : contentText(parts as Any?)
                } else { out["content"] = "" }
            } else { out["content"] = "" }
            // Ollama-style images field -> synthesize image_url content entries for template
            if let images = m["images"] as? [String], !images.isEmpty {
                var existing: [[String: Any]] = []
                if let s = out["content"] as? String, !s.isEmpty {
                    existing.append(["type": "text", "text": s])
                } else if let arr = out["content"] as? [[String: Any]] {
                    existing = arr
                }
                for b64 in images {
                    // Use data URI so template sees an image; actual bytes not needed for tokenization phase
                    let url = b64.hasPrefix("data:") ? b64 : "data:image/jpeg;base64,\(b64)"
                    existing.append(["type": "image_url", "image_url": ["url": url]])
                }
                out["content"] = existing
            }
            if let tcs = m["tool_calls"] as? [[String: Any]] {
                // OpenAI sends function.arguments as a JSON string; swift-jinja's
                // |items filter returns [] for non-dicts, which would render every
                // historical tool call without its parameters and teach the model
                // to emit argument-less calls. Parse to a dict before templating.
                out["tool_calls"] = tcs.map { tc in
                    var t = tc
                    if let fn = t["function"] as? [String: Any], let a = fn["arguments"] as? String,
                        let d = (try? JSONSerialization.jsonObject(with: Data(a.utf8))) as? [String: Any]
                    {
                        var f = fn
                        f["arguments"] = d
                        t["function"] = f
                    }
                    return t
                }
            }
            if let tcid = m["tool_call_id"] as? String { out["tool_call_id"] = tcid }
            if let name = m["name"] as? String { out["name"] = name }
            // Forward prior assistant reasoning so the template can re-emit it as <think>...
            // (chat_template.jinja reads message.reasoning_content and wraps it as
            //   <think>\n<reasoning>\n</think>\n<content> when present.)
            if let rc = m["reasoning_content"] as? String, !rc.isEmpty {
                out["reasoning_content"] = rc
            }
            return out
        }
    }

    /// tool name -> parameter name -> declared JSON type, from the request's tools.
    private static func paramTypes(from tools: [[String: Any]]?) -> [String: [String: String]] {
        guard let tools else { return [:] }
        var out: [String: [String: String]] = [:]
        for t in tools {
            guard let fn = (t["function"] as? [String: Any]) ?? (t["name"] != nil ? t : nil),
                let name = fn["name"] as? String
            else { continue }
            var types: [String: String] = [:]
            if let props = (fn["parameters"] as? [String: Any])?["properties"] as? [String: Any] {
                for (k, v) in props {
                    if let d = v as? [String: Any], let ty = d["type"] as? String { types[k] = ty }
                }
            }
            out[name] = types
        }
        return out
    }

    private static func coerceParam(_ rawVal: String, declared: String?) -> Any {
        switch declared {
        case "number", "integer":
            if let i = Int(rawVal) { return i }
            if let d = Double(rawVal) { return d }
            return rawVal
        default:
            return rawVal
        }
    }

    /// Split a raw model response into (reasoning, content). With thinking on,
    /// Qwen wraps its reasoning in `<think>...</think>`; with thinking off the
    /// template closes the think block immediately (`<think>\n\n</think>`) and
    /// the whole visible response is content. A tool-only reply may omit
    /// `</think>` entirely — then the pre-tool prefix is reasoning.
    private static func splitReasoning(
        _ raw: String, thinking: Bool
    ) -> (reasoning: String, content: String) {
        if let r = raw.range(of: "</think>") {
            let reasoning = String(raw[..<r.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let content = String(raw[r.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return (reasoning, content)
        }
        if raw.contains("<tool_call>") {
            let prefix = String(raw[..<raw.range(of: "<tool_call>")!.lowerBound])
            let rest = String(raw[raw.range(of: "<tool_call>")!.lowerBound...])
            return (
                prefix.trimmingCharacters(in: .whitespacesAndNewlines),
                rest.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }
        // No think markers at all: with thinking off the response is plain
        // content. With thinking on this is a truncated think block (token
        // limit) — surfacing half-thoughts as the answer would be worse than
        // reporting them as reasoning.
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return (thinking ? trimmed : "", thinking ? "" : trimmed)
    }

    /// Parse <tool_call> XML produced by the Qwen template into OpenAI tool_calls.
    private static func parseToolCalls(from text: String, paramTypes: [String: [String: String]] = [:]) -> (content: String, calls: [[String: Any]]?) {
        // Model emits: <tool_call><function=NAME><parameter=ARG>VAL</parameter>...</function></tool_call>
        guard text.contains("<tool_call>") else { return (text, nil) }
        let pattern = "<tool_call>\\s*<function=([^>]+)>\\s*(.*?)\\s*</function>\\s*</tool_call>"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) else {
            return (text, nil)
        }
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))
        if matches.isEmpty { return (text, nil) }
        var calls: [[String: Any]] = []
        // content before first tool_call is treated as reasoning/content
        let firstRange = matches[0].range
        let prefix = ns.substring(with: NSRange(location: 0, length: firstRange.location)).trimmingCharacters(in: .whitespacesAndNewlines)
        for (idx, m) in matches.enumerated() {
            let name = ns.substring(with: m.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
            let inner = ns.substring(with: m.range(at: 2))
            // inner contains <parameter=name>value</parameter> repeats
            let pPattern = "<parameter=([^>]+)>\\s*(.*?)\\s*</parameter>"
            let pRegex = try? NSRegularExpression(pattern: pPattern, options: [.dotMatchesLineSeparators])
            var args: [String: Any] = [:]
            if let pr = pRegex {
                let pMatches = pr.matches(in: inner, range: NSRange(location: 0, length: (inner as NSString).length))
                for pm in pMatches {
                    let key = (inner as NSString).substring(with: pm.range(at: 1))
                    let rawVal = (inner as NSString).substring(with: pm.range(at: 2)).trimmingCharacters(in: .whitespacesAndNewlines)
                    // Structured JSON objects/arrays were emitted via tojson — restore them.
                    if (rawVal.hasPrefix("{") && rawVal.hasSuffix("}"))
                        || (rawVal.hasPrefix("[") && rawVal.hasSuffix("]"))
                    {
                        if let d = rawVal.data(using: .utf8),
                            let obj = try? JSONSerialization.jsonObject(with: d, options: [.allowFragments])
                        {
                            args[key] = obj
                            continue
                        }
                    }
                    if rawVal == "true" {
                        args[key] = true
                    } else if rawVal == "false" {
                        args[key] = false
                    } else if rawVal == "null" {
                        args[key] = NSNull()
                    } else {
                        // Keep strings as strings unless the schema declares the
                        // parameter numeric (e.g. ui_tap x/y). Blind Int() here
                        // previously broke opencode's string-typed tools.
                        args[key] = Self.coerceParam(rawVal, declared: paramTypes[name]?[key])
                    }
                }
            }
            let argsJSON: String
            if let data = try? JSONSerialization.data(withJSONObject: args, options: [.sortedKeys]), let s = String(data: data, encoding: .utf8) {
                argsJSON = s
            } else { argsJSON = "{}" }
            calls.append([
                "id": "call_\(idx)",
                "type": "function",
                "function": ["name": name, "arguments": argsJSON],
            ])
        }
        return (prefix, calls.isEmpty ? nil : calls)
    }

    private func sampleParams(_ json: [String: Any]) -> SampleParams {
        let thinking = Self.bool(json["think"]) ?? false
        var p: SampleParams = thinking ? .thinking : .instruct
        if let o = json["options"] as? [String: Any] {
            if let v = Self.num(o["temperature"]) { p.temperature = Float(v) }
            if let v = Self.num(o["top_p"]) { p.topP = Float(v) }
            if let v = Self.int(o["top_k"]) { p.topK = v }
            if let v = Self.num(o["min_p"]) { p.minP = Float(v) }
            if let v = Self.num(o["presence_penalty"]) { p.presencePenalty = Float(v) }
            if let v = Self.int(o["num_predict"]) { p.maxTokens = v }
            // Ollama uses -1 for "random seed"; UInt64(-1) would trap.
            if let v = Self.int(o["seed"]) { p.seed = v < 0 ? nil : UInt64(v) }
            if let s = Self.stopList(o["stop"]) { p.stop = s }
        }
        return p.sanitized()
    }

    // MARK: /api/chat

    private func apiChat(
        _ fd: Int32, _ json: [String: Any], cors: String, logId: String
    ) {
        if let e = ollamaValidationError(
            json, allowed: ["model", "messages", "stream", "think", "options"],
            messages: true)
        {
            respondJSON(fd, ["error": e], status: "400 Bad Request", cors: cors)
            return
        }
        let stream = Self.bool(json["stream"]) ?? true
        let thinking = Self.bool(json["think"]) ?? false
        let params = sampleParams(json)
        let tmplMessages = Self.openAIMessagesForTemplate(json)
        guard !tmplMessages.isEmpty else {
            respondJSON(
                fd, ["error": "messages must not be empty"],
                status: "400 Bad Request", cors: cors)
            return
        }
        let ids: [Int]
        let visionEmbeds: MLXArray?
        do {
            (ids, visionEmbeds) = try engine.encodeWithVision(messages: tmplMessages, tools: nil, thinking: thinking)
        } catch {
            respondJSON(fd, ["error": "\(error)"], status: "400 Bad Request", cors: cors)
            return
        }
        if visionEmbeds != nil {
            FileHandle.standardError.write("[vision] \(visionEmbeds!.dim(0)) image tokens spliced\n".data(using: .utf8)!)
        }
        if let e = engine.contextError(promptTokens: ids.count) {
            respondJSON(fd, ["error": e], status: "400 Bad Request", cors: cors)
            return
        }
        let t0 = Date()
        if stream, !startChunked(fd, contentType: "application/x-ndjson", cors: cors) { return }
        var alive = true
        // Delta logging exists only under --output; without it a non-streaming
        // request passes no callback at all and nothing is accumulated.
        let logDeltas = outputPath != nil
        var deltaSeq = 0
        let callback: ((Int, String) -> Bool)? = (stream || logDeltas) ? { tok, delta in
            if logDeltas {
                self.logDelta(id: logId, seq: deltaSeq, tok: tok, text: delta)
                deltaSeq += 1
            }
            guard stream else { return true }
            guard alive, !delta.isEmpty else { return alive }
            let obj: [String: Any] = [
                "model": self.engine.modelName, "created_at": self.iso(Date()),
                "message": ["role": "assistant", "content": delta], "done": false,
            ]
            alive = self.chunk(
                fd, (try! JSONSerialization.data(withJSONObject: obj)) + Data("\n".utf8))
            return alive
        } : nil
        let (text, _, stats) = engine.generate(
            promptIds: ids, params: params, visionEmbeds: visionEmbeds,
            shouldContinue: { self.peerAlive(fd) }, onToken: callback)
        if logDeltas {
            logResponse(
                id: logId, endpoint: "/api/chat", response: text,
                finish: stats.finishReason, stopTokenId: stats.stopTokenId,
                stopTokenHex: stats.stopTokenHex)
        }
        let final: [String: Any] = [
            "model": engine.modelName, "created_at": iso(Date()),
            "message": ["role": "assistant", "content": stream ? "" : text],
            "done": true, "done_reason": stats.finishReason,
            "total_duration": Int(-t0.timeIntervalSinceNow * 1e9),
            "prompt_eval_count": stats.promptTokens,
            "prompt_eval_duration": Int(stats.prefillSeconds * 1e9),
            "eval_count": stats.decodeTokens,
            "eval_duration": Int(stats.decodeSeconds * 1e9),
        ]
        if stream, alive {
            chunk(fd, (try! JSONSerialization.data(withJSONObject: final)) + Data("\n".utf8))
            endChunked(fd)
        } else {
            if alive { respondJSON(fd, final, cors: cors) }
        }
    }

    // MARK: /api/generate

    private func apiGenerate(
        _ fd: Int32, _ json: [String: Any], cors: String, logId: String
    ) {
        if let e = ollamaValidationError(
            json,
            allowed: ["model", "prompt", "system", "raw", "stream", "think", "options"])
        {
            respondJSON(fd, ["error": e], status: "400 Bad Request", cors: cors)
            return
        }
        if json["prompt"] != nil, json["prompt"] as? String == nil {
            respondJSON(
                fd, ["error": "prompt must be text"],
                status: "400 Bad Request", cors: cors)
            return
        }
        if json["system"] != nil, json["system"] as? String == nil {
            respondJSON(
                fd, ["error": "system must be text"],
                status: "400 Bad Request", cors: cors)
            return
        }
        if json["raw"] != nil, Self.bool(json["raw"]) == nil {
            respondJSON(
                fd, ["error": "raw must be true or false"],
                status: "400 Bad Request", cors: cors)
            return
        }
        let prompt = json["prompt"] as? String ?? ""
        let raw = Self.bool(json["raw"]) ?? false
        let stream = Self.bool(json["stream"]) ?? true
        let thinking = Self.bool(json["think"]) ?? false
        guard !prompt.isEmpty else {
            respondJSON(
                fd, ["error": "prompt must not be empty"],
                status: "400 Bad Request", cors: cors)
            return
        }
        if raw, thinking || json["system"] != nil {
            respondJSON(
                fd, ["error": "raw generation cannot apply system or think; remove raw or those fields"],
                status: "400 Bad Request", cors: cors)
            return
        }
        let params = sampleParams(json)
        let ids: [Int]
        if raw {
            ids = engine.tokenizer.encode(text: prompt)
        } else {
            var messages: [ChatMessage] = []
            if let system = json["system"] as? String, !system.isEmpty {
                messages.append(ChatMessage(role: "system", content: system))
            }
            messages.append(ChatMessage(role: "user", content: prompt))
            ids = (try? engine.encodeChat(messages, thinking: thinking)) ?? []
        }
        // An empty prompt would leave the first logits uninitialized and make
        // the sampler invent a token out of nothing.
        guard !ids.isEmpty else {
            respondJSON(
                fd, ["error": "prompt must not be empty"],
                status: "400 Bad Request", cors: cors)
            return
        }
        if let e = engine.contextError(promptTokens: ids.count) {
            respondJSON(fd, ["error": e], status: "400 Bad Request", cors: cors)
            return
        }
        let t0 = Date()
        if stream, !startChunked(fd, contentType: "application/x-ndjson", cors: cors) { return }
        var alive = true
        let logDeltas = outputPath != nil
        var deltaSeq = 0
        let callback: ((Int, String) -> Bool)? = (stream || logDeltas) ? { tok, delta in
            if logDeltas {
                self.logDelta(id: logId, seq: deltaSeq, tok: tok, text: delta)
                deltaSeq += 1
            }
            guard stream else { return true }
            guard alive, !delta.isEmpty else { return alive }
            let obj: [String: Any] = [
                "model": self.engine.modelName, "created_at": self.iso(Date()),
                "response": delta, "done": false,
            ]
            alive = self.chunk(
                fd, (try! JSONSerialization.data(withJSONObject: obj)) + Data("\n".utf8))
            return alive
        } : nil
        let (text, _, stats) = engine.generate(
            promptIds: ids, params: params,
            shouldContinue: { self.peerAlive(fd) }, onToken: callback)
        if logDeltas {
            logResponse(
                id: logId, endpoint: "/api/generate", response: text,
                finish: stats.finishReason, stopTokenId: stats.stopTokenId,
                stopTokenHex: stats.stopTokenHex)
        }
        let final: [String: Any] = [
            "model": engine.modelName, "created_at": iso(Date()),
            "response": stream ? "" : text, "done": true, "done_reason": stats.finishReason,
            "total_duration": Int(-t0.timeIntervalSinceNow * 1e9),
            "prompt_eval_count": stats.promptTokens,
            "prompt_eval_duration": Int(stats.prefillSeconds * 1e9),
            "eval_count": stats.decodeTokens,
            "eval_duration": Int(stats.decodeSeconds * 1e9),
        ]
        if stream, alive {
            chunk(fd, (try! JSONSerialization.data(withJSONObject: final)) + Data("\n".utf8))
            endChunked(fd)
        } else {
            if alive { respondJSON(fd, final, cors: cors) }
        }
    }

    // MARK: /v1/chat/completions (OpenAI, SSE streaming)

    private func v1Chat(
        _ fd: Int32, _ json: [String: Any], cors: String, logId: String
    ) {
        if let e = openAIValidationError(json) {
            respondJSON(
                fd, ["error": ["message": e, "type": "invalid_request_error"]],
                status: "400 Bad Request", cors: cors)
            return
        }
        let stream = Self.bool(json["stream"]) ?? false
        var params = SampleParams.instruct
        if let v = Self.num(json["temperature"]) { params.temperature = Float(v) }
        if let v = Self.num(json["top_p"]) { params.topP = Float(v) }
        if let v = Self.int(json["top_k"]) { params.topK = v }
        if let v = Self.num(json["presence_penalty"]) { params.presencePenalty = Float(v) }
        if let v = Self.int(json["max_tokens"]) { params.maxTokens = v }
        if let v = Self.int(json["max_completion_tokens"]) { params.maxTokens = v }
        if let v = Self.int(json["seed"]) {
            params.seed = UInt64(bitPattern: Int64(v))
        }
        if let v = Self.stopList(json["stop"]) { params.stop = v }
        params = params.sanitized()
        let wantUsage = Self.bool(
            (json["stream_options"] as? [String: Any])?["include_usage"]) ?? false
        // Qwen's chat template defaults `enable_thinking` to true and reads
        // `reasoning_effort` to tailor the system instruction. We mirror the
        // template default so OpenAI clients (opencode) see reasoning_content
        // deltas instead of silence; clients that want to skip thinking send
        // `enable_thinking: false`.
        let thinking = Self.bool(json["enable_thinking"]) ?? true
        let reasoningEffort = json["reasoning_effort"] as? String
        let tmplMessages = Self.openAIMessagesForTemplate(json)
        let tmplTools = Self.openAITools(json)
        guard !tmplMessages.isEmpty else {
            respondJSON(
                fd, ["error": ["message": "messages must not be empty"]],
                status: "400 Bad Request", cors: cors)
            return
        }
        let ids: [Int]
        let visionEmbeds: MLXArray?
        do {
            (ids, visionEmbeds) = try engine.encodeWithVision(
                messages: tmplMessages, tools: tmplTools,
                thinking: thinking, reasoningEffort: reasoningEffort)
        } catch {
            respondJSON(fd, ["error": ["message": "\(error)"]], status: "400 Bad Request", cors: cors)
            return
        }
        if let e = engine.contextError(promptTokens: ids.count) {
            respondJSON(
                fd, ["error": ["message": e]], status: "400 Bad Request", cors: cors)
            return
        }
        if visionEmbeds != nil {
            FileHandle.standardError.write("[vision] \(visionEmbeds!.dim(0)) tokens for \(tmplMessages.count) messages\n".data(using: .utf8)!)
        }
        let rid = "chatcmpl-\(UUID().uuidString.prefix(8))"
        if stream, !startChunked(fd, contentType: "text/event-stream", cors: cors) { return }
        var alive = true
        // OpenAI streaming clients expect the assistant role to be announced
        // before any delta arrives. Emit it once, immediately, so the client
        // starts rendering the turn before the first token.
        if stream, alive {
            let open: [String: Any] = [
                "id": rid, "object": "chat.completion.chunk",
                "created": Int(Date().timeIntervalSince1970), "model": engine.modelName,
                "choices": [["index": 0, "delta": ["role": "assistant"], "finish_reason": NSNull()]],
            ]
            alive = self.chunk(
                fd, Data("data: ".utf8)
                    + (try! JSONSerialization.data(withJSONObject: open))
                    + Data("\n\n".utf8))
        }
        // Translate Qwen <tool_call> XML to OpenAI tool_calls incrementally.
        // The model emits XML; we stream reasoning/content between tag
        // boundaries, the tool_call header (id + name) the moment <function=NAME>
        // opens, and the arguments JSON when </function></tool_call> closes.
        // That way opencode sees "calling tool X..." as soon as the model commits
        // to a tool, instead of waiting for every parameter to be filled in.
        var streamBuf = ""
        var reasoningSent = 0
        var contentSent = 0
        var toolHeadersEmitted = 0   // count of tool_calls headers streamed
        // toolCalls state: idx -> (name, args dict, full block emitted?)
        var toolState: [Int: (name: String, args: [String: Any], closed: Bool)] = [:]
        let toolOpenRegex = try? NSRegularExpression(
            pattern: "<tool_call>\\s*<function=([^>]+)>",
            options: [])
        let toolCloseRegex = try? NSRegularExpression(
            pattern: "<tool_call>\\s*<function=([^>]+)>\\s*(.*?)\\s*</function>\\s*</tool_call>",
            options: [.dotMatchesLineSeparators])
        let paramPattern = "<parameter=([^>]+)>\\s*(.*?)\\s*</parameter>"
        let paramRegex = try? NSRegularExpression(pattern: paramPattern, options: [.dotMatchesLineSeparators])
        let paramTypes = Self.paramTypes(from: tmplTools)
        let now = { Int(Date().timeIntervalSince1970) }
        // When thinking is disabled, the model is told to skip reasoning and
        // the response has no `<think>...</think>` wrapping. In that mode any
        // pre-tool text is real content, not reasoning — stream it under
        // `delta.content` instead of `delta.reasoning_content`.
        let streamAsReasoning = thinking

        func sse(_ obj: [String: Any]) -> Bool {
            let data = try! JSONSerialization.data(withJSONObject: obj)
            return self.chunk(fd, Data("data: ".utf8) + data + Data("\n\n".utf8))
        }

        /// Returns the substring of `buf` starting at `after`, minus any
        /// <tool_call>...</tool_call> block ranges that fall at or after
        /// `after`, and truncated at the first unclosed <tool_call> opener
        /// in any remaining gap. Used so content deltas after </think> don't
        /// leak the raw tool-call XML alongside the parsed tool_calls deltas.
        func cleanContentTail(
            _ buf: String, after: String.Index,
            toolCloseRegex: NSRegularExpression?
        ) -> String {
            guard let regex = toolCloseRegex else { return String(buf[after...]) }
            let ns = buf as NSString
            let allMatches = regex.matches(
                in: buf, range: NSRange(location: 0, length: ns.length))
            // Tool_call blocks entirely at or after the start position. Anything
            // overlapping the reasoning boundary is left untouched.
            let afterUtf16 = after.utf16Offset(in: buf)
            let blocks = allMatches
                .map { $0.range }
                .filter { $0.location >= afterUtf16 }
                .sorted { $0.location < $1.location }
            var pieces: [String] = []
            var cursor = afterUtf16
            let total = ns.length
            for tr in blocks {
                if tr.location > cursor {
                    pieces.append(ns.substring(
                        with: NSRange(location: cursor, length: tr.location - cursor)))
                }
                cursor = tr.location + tr.length
            }
            if cursor < total {
                // If the gap after the last closed block still contains an
                // unclosed <tool_call>, stop at its opener so the raw XML
                // (the opening tag, <function=NAME>, <parameter=...>...</parameter>
                // contents, and the </function></tool_call> close) does not
                // get streamed as content. Once that block closes the next
                // pass through the callback will pick up the bytes after
                // </tool_call> as content.
                let gap = ns.substring(
                    with: NSRange(location: cursor, length: total - cursor))
                let endOfSafeContent: String.Index
                if let r = gap.range(of: "<tool_call>") {
                    endOfSafeContent = r.lowerBound
                } else {
                    endOfSafeContent = gap.endIndex
                }
                if endOfSafeContent > gap.startIndex {
                    pieces.append(String(gap[..<endOfSafeContent]))
                }
            }
            return pieces.joined()
        }

        /// Build the args dict for one tool_call block from its inner XML.
        /// Returns the canonical sorted JSON string of those args.
        func parseArgs(inner: String, name: String) -> String {
            var args: [String: Any] = [:]
            if let pr = paramRegex {
                let innerNS = inner as NSString
                let pMatches = pr.matches(
                    in: inner, range: NSRange(location: 0, length: innerNS.length))
                for pm in pMatches {
                    let key = innerNS.substring(with: pm.range(at: 1))
                    let rawVal = innerNS.substring(with: pm.range(at: 2))
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if (rawVal.hasPrefix("{") && rawVal.hasSuffix("}"))
                        || (rawVal.hasPrefix("[") && rawVal.hasSuffix("]"))
                    {
                        if let d = rawVal.data(using: .utf8),
                            let obj = try? JSONSerialization.jsonObject(
                                with: d, options: [.allowFragments])
                        {
                            args[key] = obj
                            continue
                        }
                    }
                    if rawVal == "true" { args[key] = true }
                    else if rawVal == "false" { args[key] = false }
                    else if rawVal == "null" { args[key] = NSNull() }
                    else { args[key] = Self.coerceParam(rawVal, declared: paramTypes[name]?[key]) }
                }
            }
            if let data = try? JSONSerialization.data(
                withJSONObject: args, options: [.sortedKeys]),
                let s = String(data: data, encoding: .utf8)
            { return s }
            return "{}"
        }

        let logDeltas = outputPath != nil
        var deltaSeq = 0
        let callback: ((Int, String) -> Bool)? = (stream || logDeltas) ? { tok, delta in
            if logDeltas {
                self.logDelta(id: logId, seq: deltaSeq, tok: tok, text: delta)
                deltaSeq += 1
            }
            guard stream else { return true }
            guard alive, !delta.isEmpty else { return alive }
            streamBuf += delta

            // 1. Emit any newly opened tool_call headers as soon as we see
            //    `<tool_call><function=NAME>` in the buffer.
            if let regex = toolOpenRegex {
                let ns = streamBuf as NSString
                let opens = regex.matches(
                    in: streamBuf, range: NSRange(location: 0, length: ns.length))
                for idx in toolHeadersEmitted ..< opens.count {
                    let m = opens[idx]
                    let name = ns.substring(with: m.range(at: 1))
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if toolState[idx] == nil {
                        toolState[idx] = (name: name, args: [:], closed: false)
                    }
                    let obj: [String: Any] = [
                        "id": rid, "object": "chat.completion.chunk",
                        "created": now(), "model": self.engine.modelName,
                        "choices": [[
                            "index": 0,
                            "delta": ["tool_calls": [[
                                "index": idx,
                                "id": "call_\(idx)",
                                "type": "function",
                                "function": ["name": name, "arguments": ""],
                            ]]],
                            "finish_reason": NSNull(),
                        ]],
                    ]
                    alive = sse(obj)
                    if !alive { return false }
                    toolHeadersEmitted = idx + 1
                }
            }

            // 2. Stream reasoning/content deltas. Anything between the start of
            //    streamBuf and the first </think> or <tool_call> boundary is
            //    reasoning when thinking is enabled, content when it isn't.
            //    Anything after </think> is content. A tool-only reply (no
            //    </think>) treats the prefix as reasoning/content per the
            //    same flag.
            if let r = streamBuf.range(of: "</think>") {
                let reasoning = String(streamBuf[..<r.lowerBound])
                if reasoning.count > reasoningSent {
                    let toSend = String(reasoning.dropFirst(reasoningSent))
                    if !toSend.isEmpty {
                        let obj: [String: Any] = [
                            "id": rid, "object": "chat.completion.chunk",
                            "created": now(), "model": self.engine.modelName,
                            "choices": [["index": 0,
                                "delta": ["reasoning_content": toSend],
                                "finish_reason": NSNull()]],
                        ]
                        alive = sse(obj)
                        reasoningSent = reasoning.count
                    }
                }
                // Anything after </think> is real content — but we must skip
                // any <tool_call>...</tool_call> blocks in that tail so the
                // raw XML isn't emitted as a content delta in addition to the
                // parsed tool_calls deltas (which is what OpenCode users see
                // as "the tool call appearing twice, in the raw text").
                let afterIdx = streamBuf.index(after: r.upperBound)
                let tail = cleanContentTail(streamBuf, after: afterIdx, toolCloseRegex: toolCloseRegex)
                if tail.count > contentSent {
                    let toSend = String(tail.dropFirst(contentSent))
                    if !toSend.isEmpty {
                        let obj: [String: Any] = [
                            "id": rid, "object": "chat.completion.chunk",
                            "created": now(), "model": self.engine.modelName,
                            "choices": [["index": 0,
                                "delta": ["content": toSend],
                                "finish_reason": NSNull()]],
                        ]
                        alive = sse(obj)
                        contentSent = tail.count
                    }
                }
            } else if let r = streamBuf.range(of: "<tool_call>") {
                let prefix = String(streamBuf[..<r.lowerBound])
                if prefix.count > reasoningSent
                    && !prefix.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                {
                    let toSend = String(prefix.dropFirst(reasoningSent))
                    if !toSend.isEmpty {
                        let key = streamAsReasoning ? "reasoning_content" : "content"
                        let obj: [String: Any] = [
                            "id": rid, "object": "chat.completion.chunk",
                            "created": now(), "model": self.engine.modelName,
                            "choices": [["index": 0,
                                "delta": [key: toSend],
                                "finish_reason": NSNull()]],
                        ]
                        alive = sse(obj)
                        reasoningSent = prefix.count
                    }
                }
            } else {
                let key = streamAsReasoning ? "reasoning_content" : "content"
                let obj: [String: Any] = [
                    "id": rid, "object": "chat.completion.chunk",
                    "created": now(), "model": self.engine.modelName,
                    "choices": [["index": 0,
                        "delta": [key: delta],
                        "finish_reason": NSNull()]],
                ]
                alive = sse(obj)
                reasoningSent += delta.count
                return alive
            }

            // 3. For each newly completed <tool_call>...</tool_call> block,
            //    emit the arguments JSON. The header was already streamed in
            //    step 1, so this delta only contains `function.arguments`.
            guard let regex = toolCloseRegex else { return alive }
            let ns = streamBuf as NSString
            let matches = regex.matches(
                in: streamBuf, range: NSRange(location: 0, length: ns.length))
            for idx in 0 ..< matches.count {
                let m = matches[idx]
                guard var st = toolState[idx], !st.closed else { continue }
                let name = ns.substring(with: m.range(at: 1))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let inner = ns.substring(with: m.range(at: 2))
                let argsJSON = parseArgs(inner: inner, name: name)
                let obj: [String: Any] = [
                    "id": rid, "object": "chat.completion.chunk",
                    "created": now(), "model": self.engine.modelName,
                    "choices": [[
                        "index": 0,
                        "delta": ["tool_calls": [[
                            "index": idx,
                            "function": ["arguments": argsJSON],
                        ]]],
                        "finish_reason": NSNull(),
                    ]],
                ]
                alive = sse(obj)
                if !alive { return false }
                st.closed = true
                toolState[idx] = st
            }
            return alive
        } : nil
        let (rawText, _, stats) = engine.generate(
            promptIds: ids, params: params, visionEmbeds: visionEmbeds,
            shouldContinue: { self.peerAlive(fd) }, onToken: callback)
        if logDeltas {
            logResponse(
                id: logId, endpoint: "/v1/chat/completions", response: rawText,
                finish: stats.finishReason, stopTokenId: stats.stopTokenId,
                stopTokenHex: stats.stopTokenHex)
        }
        // Split reasoning from the visible response first, then parse tool
        // calls out of the content part only — so reasoning text never leaks
        // into parsed.content and vice versa.
        let split = Self.splitReasoning(rawText, thinking: thinking)
        let parsed = Self.parseToolCalls(from: split.content, paramTypes: paramTypes)
        let text: String = parsed.content
        let toolCalls = parsed.calls
        let finishReason: String = toolCalls != nil ? "tool_calls" : stats.finishReason
        if stream, alive {
            // Emit any tool_calls not already streamed incrementally (fallback for
            // a block whose `</function></tool_call>` closed after the last callback).
            if let calls = toolCalls {
                for idx in 0 ..< calls.count {
                    guard toolState[idx] == nil else { continue }
                    let call = calls[idx]
                    // Header first (id + name), then arguments, so the delta
                    // shape stays the same as the streaming path.
                    if let fn = call["function"] as? [String: Any],
                        let name = fn["name"] as? String
                    {
                        let head: [String: Any] = [
                            "id": rid, "object": "chat.completion.chunk",
                            "created": now(), "model": engine.modelName,
                            "choices": [[
                                "index": 0,
                                "delta": ["tool_calls": [[
                                    "index": idx,
                                    "id": call["id"] as? String ?? "call_\(idx)",
                                    "type": "function",
                                    "function": ["name": name, "arguments": ""],
                                ]]],
                                "finish_reason": NSNull(),
                            ]],
                        ]
                        alive = self.chunk(
                            fd, Data("data: ".utf8)
                                + (try! JSONSerialization.data(withJSONObject: head))
                                + Data("\n\n".utf8))
                        if !alive { break }
                        let args = fn["arguments"] as? String ?? "{}"
                        let argsDelta: [String: Any] = [
                            "id": rid, "object": "chat.completion.chunk",
                            "created": now(), "model": engine.modelName,
                            "choices": [[
                                "index": 0,
                                "delta": ["tool_calls": [[
                                    "index": idx,
                                    "function": ["arguments": args],
                                ]]],
                                "finish_reason": NSNull(),
                            ]],
                        ]
                        alive = self.chunk(
                            fd, Data("data: ".utf8)
                                + (try! JSONSerialization.data(withJSONObject: argsDelta))
                                + Data("\n\n".utf8))
                        if !alive { break }
                    }
                }
            }
            var fin: [String: Any] = [
                "id": rid, "object": "chat.completion.chunk",
                "created": Int(Date().timeIntervalSince1970), "model": engine.modelName,
                "choices": [["index": 0, "delta": [:], "finish_reason": finishReason]],
            ]
            if wantUsage {
                fin["usage"] = [
                    "prompt_tokens": stats.promptTokens,
                    "completion_tokens": stats.decodeTokens,
                    "total_tokens": stats.promptTokens + stats.decodeTokens,
                ]
            }
            chunk(fd, Data("data: ".utf8) + (try! JSONSerialization.data(withJSONObject: fin)) + Data("\n\n".utf8))
            chunk(fd, Data("data: [DONE]\n\n".utf8))
            endChunked(fd)
        } else {
            var message: [String: Any] = ["role": "assistant"]
            if !split.reasoning.isEmpty { message["reasoning_content"] = split.reasoning }
            // OpenAI expects content to be string or null; null when tool_calls only
            if let calls = toolCalls {
                message["content"] = text.isEmpty ? NSNull() : text as Any
                message["tool_calls"] = calls
            } else {
                message["content"] = text
            }
            respondJSON(
                fd,
                [
                    "id": rid, "object": "chat.completion",
                    "created": Int(Date().timeIntervalSince1970), "model": engine.modelName,
                    "choices": [
                        [
                            "index": 0, "finish_reason": finishReason,
                            "message": message,
                        ]
                    ],
                    "usage": [
                        "prompt_tokens": stats.promptTokens,
                        "completion_tokens": stats.decodeTokens,
                        "total_tokens": stats.promptTokens + stats.decodeTokens,
                    ],
                ], cors: cors)
        }
    }
}
