// DeepSeek-V4-Flash tokenizer + chat template export: converts the GGUF
// `tokenizer.ggml.*` metadata into a Hugging Face tokenizer directory that
// swift-transformers loads through the same `AutoTokenizer.from(modelFolder:)`
// path Engine.swift uses for the Qwen checkpoint. Consumes `GGUFFile` metadata
// only — no tensor data.

import Foundation
import Tokenizers

// MARK: - joyai-llm pre-tokenization

/// The `tokenizer.ggml.pre == "joyai-llm"` split, ported from the reference C
/// implementation (ds4.c `bpe_tokenize_text`, lines 41210-41298, with its
/// helpers at 40883-40969). The C code is a hand-rolled scanner, not a regex,
/// so this constant is the faithful translation of its if/else chain into an
/// ICU-regex alternation — alternatives appear in exactly the order of the C
/// branches, because both engines are backtracking leftmost-first. Each
/// alternative below names the C branch it mirrors:
///
///   1. `[0-9]{1,3}`                            — digit run, at most 3 (ds4.c:41240-41245)
///   2. `[\x{3040}-\x{309F}...]+`               — CJK/Hiragana/Katakana run (joyai_cjk_at,
///                                                ds4.c:40913-40917, 41246-41249)
///   3. `[!-/:-@\[-\x60{-~][A-Za-z]+`           — ASCII punct/symbol followed by ASCII
///                                                letters (joyai_ascii_punct_symbol,
///                                                ds4.c:40906-40911, 41250-41254)
///   4. `[^\x00-\x40\x5B-\x60\x7B-\x7F]+`       — letter-like run: ASCII letters plus
///                                                every non-ASCII character (CJK
///                                                included; the CJK alternative above
///                                                wins when a run *starts* with CJK,
///                                                exactly as the if-chain does)
///                                                (joyai_letter_like_at, ds4.c:40941-40955)
///   5. `LEAD?L+`                               — one leading non-newline non-punct char
///                                                (space, tab, \v, \f, control) joining
///                                                a letter run (" int" is one piece)
///                                                (ds4.c:41257-41262)
///   6. ` [PUNCT]+[\x0A\x0D]*`                  — a single leading space before a punct
///                                                run, keeping trailing newlines in the
///                                                same BPE word (">;\n") (ds4.c:41263-41268)
///   7. `[PUNCT]+[\x0A\x0D]*`                   — punct run + trailing newlines (ds4.c:41269-41271)
///   8. `[\x09-\x0D\x20]*[\x0A\x0D]+`           — whitespace run ending at its LAST
///                                                newline, inclusive (last_newline_end,
///                                                ds4.c:41272-41280); the explicit ASCII
///                                                class matters: non-ASCII whitespace is
///                                                letter-like in the C code
///   9. `[\x09-\x0D\x20]+(?=WS(?:L|PUNCT))`     — multi-space run before a letter/punct
///                                                leaves exactly one space to join the
///                                                next piece: "    int" -> "   " + " int"
///                                                (the join sub-rule, ds4.c:41281-41289)
///   10. `[\x09-\x0D\x20]+`                     — remaining whitespace runs (ds4.c:41290-41291)
///   11. `[\x00-\x08\x0E-\x1F\x7F]`             — any other ASCII control byte is a
///                                                single-character piece (final else,
///                                                ds4.c:41293-41294)
///
/// Where: PUNCT = `[!-/:-@\[-\x60{-~]` (ds4.c:40906-40911), L =
/// `[^\x00-\x40\x5B-\x60\x7B-\x7F]` (ASCII letters + all non-ASCII,
/// ds4.c:40941-40954), WS = `[\x09-\x0D\x20]` (ascii_space, ds4.c:40897-40900).
/// The pattern is total (every character is covered by some alternative) and
/// never matches empty, so splitting by it reproduces the scanner's pieces
/// exactly. Equivalence was checked against a transcription of the C scanner
/// on hand-written and randomized inputs (code, CJK, escapes, whitespace
/// torture cases) with zero boundary mismatches.
private let joyaiPreTokenizeRegex =
    #"[0-9]{1,3}|[\x{3040}-\x{309F}\x{30A0}-\x{30FF}\x{4E00}-\x{9FA5}]+"#
    + #"|[!-/:-@\[-\x60{-~][A-Za-z]+"#
    + #"|[^\x00-\x40\x5B-\x60\x7B-\x7F]+"#
    + #"|[\x00-\x09\x0B\x0C\x0E-\x1F\x20\x7F]?[^\x00-\x40\x5B-\x60\x7B-\x7F]+"#
    + #"| [!-/:-@\[-\x60{-~]+[\x0A\x0D]*"#
    + #"|[!-/:-@\[-\x60{-~]+[\x0A\x0D]*"#
    + #"|[\x09-\x0D\x20]*[\x0A\x0D]+"#
    + #"|[\x09-\x0D\x20]+(?=[\x09-\x0D\x20](?:[^\x00-\x40\x5B-\x60\x7B-\x7F]|[!-/:-@\[-\x60{-~]))"#
    + #"|[\x09-\x0D\x20]+"#
    + #"|[\x00-\x08\x0E-\x1F\x7F]"#

// MARK: - Export

/// Exports the tokenizer from a DeepSeek-V4 GGUF into `dir`, producing:
///
/// - `tokenizer.json` — byte-level BPE (GPT-2 alphabet) in HF format. The GGUF
///   token strings are already byte-level-unicode-mapped ("Ġthe", "ĊĊ"), so
///   they are copied verbatim into the vocab; merges are the length-prefixed
///   "a b" pair strings copied verbatim. The pre-tokenizer is a Sequence of
///   the joyai-llm Split (see above) and a regex-free ByteLevel step that only
///   byte-encodes each piece; the decoder is ByteLevel. Non-normal tokens
///   (`tokenizer.ggml.token_type` != 1) become `added_tokens` entries with
///   `special: true`, which is also what lets the ByteLevel decoder pass
///   special tokens like "<｜ref｜>" through unmapped.
/// - `tokenizer_config.json` — minimal config (tokenizer_class, bos/eos/pad,
///   the add_bos/add_eos flags, clean_up_tokenization_spaces false so decode
///   is byte-faithful).
/// - `chat_template.jinja` — the `tokenizer.chat_template` kv verbatim
///   (~5 KB); Hub loads it and merges it into the tokenizer config.
public func exportTokenizer(from gguf: GGUFFile, to dir: String) throws {
    func require(_ key: String) throws -> GGUFValue {
        guard let v = gguf.kv(key) else {
            throw ModelError("GGUF is missing \(key) — cannot export the tokenizer")
        }
        return v
    }
    func stringArray(_ key: String) throws -> [String] {
        guard let arr = try require(key).arrayValue else {
            throw ModelError("\(key) must be a string array")
        }
        guard let strings = arr.map(\.stringValue) as? [String] else {
            throw ModelError("\(key) must contain only strings")
        }
        return strings
    }

    let pre = try require("tokenizer.ggml.pre").stringValue
    guard pre == "joyai-llm" else {
        throw ModelError("unsupported tokenizer.ggml.pre '\(pre ?? "nil")' — this exporter ports only 'joyai-llm'")
    }
    let tokens = try stringArray("tokenizer.ggml.tokens")
    let merges = try stringArray("tokenizer.ggml.merges")
    guard !tokens.isEmpty else { throw ModelError("tokenizer.ggml.tokens is empty") }

    let tokenTypes: [Int]?
    if gguf.kv("tokenizer.ggml.token_type") != nil {
        guard let arr = gguf.kv("tokenizer.ggml.token_type")!.arrayValue,
            let types = arr.map(\.intValue) as? [Int], types.count == tokens.count
        else {
            throw ModelError("tokenizer.ggml.token_type must be an integer array the length of the token array")
        }
        tokenTypes = types
    } else {
        tokenTypes = nil
    }

    func tokenString(_ key: String) throws -> String {
        let id = try require(key).intValue
        guard let id, id >= 0, id < tokens.count else {
            throw ModelError("\(key) (\(id ?? -1)) is outside the vocabulary")
        }
        return tokens[id]
    }
    let bos = try tokenString("tokenizer.ggml.bos_token_id")
    let eos = try tokenString("tokenizer.ggml.eos_token_id")
    let pad = gguf.kv("tokenizer.ggml.padding_token_id")?.intValue.flatMap {
        $0 >= 0 && $0 < tokens.count ? tokens[$0] : nil
    }

    // GGML token types: 1 NORMAL, 2 UNKNOWN, 3 CONTROL, 4 USER_DEFINED,
    // 5 UNUSED, 6 BYTE. Everything non-normal is exported as an added (and
    // special) token: this file's 1283 are the <｜...｜> controls and the
    // six user-defined tags (<think>, </think>, DSML markers, ...).
    var addedTokens: [[String: Any]] = []
    if let tokenTypes {
        for (id, type) in tokenTypes.enumerated() where type != 1 {
            addedTokens.append([
                "id": id,
                "content": tokens[id],
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            ])
        }
    }

    let vocab = Dictionary(uniqueKeysWithValues: tokens.enumerated().map { ($1, $0) })
    let tokenizerJSON: [String: Any] = [
        "version": "1.0",
        "truncation": NSNull(),
        "padding": NSNull(),
        "added_tokens": addedTokens,
        "normalizer": NSNull(),
        "pre_tokenizer": [
            "type": "Sequence",
            "pretokenizers": [
                [
                    "type": "Split",
                    "pattern": ["Regex": joyaiPreTokenizeRegex],
                    "behavior": "Isolated",
                    "invert": false,
                ],
                [
                    // Byte-encode each pre-split piece without re-splitting:
                    // use_regex false is what keeps the joyai-llm boundaries.
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": false,
                    "use_regex": false,
                ],
            ],
        ],
        "post_processor": NSNull(),
        "decoder": ["type": "ByteLevel"],
        "model": [
            "type": "BPE",
            "dropout": NSNull(),
            "unk_token": NSNull(),
            "continuing_subword_prefix": NSNull(),
            "end_of_word_suffix": NSNull(),
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": vocab,
            "merges": merges,
        ],
    ]

    var tokenizerConfig: [String: Any] = [
        "tokenizer_class": "GPT2Tokenizer",
        "bos_token": bos,
        "eos_token": eos,
        "add_bos_token": gguf.kv("tokenizer.ggml.add_bos_token")?.boolValue ?? false,
        "add_eos_token": gguf.kv("tokenizer.ggml.add_eos_token")?.boolValue ?? false,
        "clean_up_tokenization_spaces": false,
    ]
    if let pad { tokenizerConfig["pad_token"] = pad }

    try FileManager.default.createDirectory(
        atPath: dir, withIntermediateDirectories: true)
    try writeJSON(tokenizerJSON, to: dir + "/tokenizer.json")
    try writeJSON(tokenizerConfig, to: dir + "/tokenizer_config.json")
    if let template = gguf.kv("tokenizer.chat_template")?.stringValue {
        try template.data(using: .utf8)?.write(to: URL(fileURLWithPath: dir + "/chat_template.jinja"))
    }
}

private func writeJSON(_ object: [String: Any], to path: String) throws {
    let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    try data.write(to: URL(fileURLWithPath: path))
}

// MARK: - Smoke path

/// Callable smoke check for the exported tokenizer (no CLI wiring — call this
/// from an integration harness). Parses the GGUF header, exports the tokenizer
/// into `modelDir`, loads it back with swift-transformers' `AutoTokenizer`,
/// and round-trips a battery of strings, returning a human-readable report.
public enum DS4TokenizerSmoke {
    public static func run(ggufPath: String, modelDir: String) async throws -> String {
        let gguf = try GGUFFile(path: ggufPath)
        _ = try DS4Config(gguf: gguf)
        try exportTokenizer(from: gguf, to: modelDir)

        let tokenizer = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: modelDir))

        var report: [String] = []
        func check(_ what: String, _ condition: @autoclosure () -> Bool) throws {
            guard condition() else { throw ModelError("tokenizer smoke failed: \(what)") }
            report.append("ok: \(what)")
        }

        // Special tokens come straight from the GGUF ids.
        try check("bos token id is 0", tokenizer.bosTokenId == 0)
        try check("eos token id is 1", tokenizer.eosTokenId == 1)
        try check("'!' maps to id 3 (byte-level single-char block)", tokenizer.convertTokenToId("!") == 3)
        try check("vocab contains 'Ġthe' (GPT-2 byte mapping)", tokenizer.convertTokenToId("Ġthe") != nil)
        try check("control token <｜ref｜> resolves", tokenizer.convertTokenToId("<｜ref｜>") != nil)

        // The pre-tokenizer split shape: one leading space joins the word
        // ("    int" splits "   " + " int", not "    " + "int"), and the
        // pieces byte-encode into the GPT-2 alphabet.
        try check("tokenize(' the') == ['Ġthe']", tokenizer.tokenize(text: " the") == ["Ġthe"])
        try check(
            "tokenize('    int') == ['ĠĠĠ', 'Ġint']",
            tokenizer.tokenize(text: "    int") == ["ĠĠĠ", "Ġint"])

        // Byte-faithful round trips (no clean-up, no prefix space).
        for text in [
            "Hello, world!",
            "  int main(void) { return 0; }",
            ">;\n",  // the punct rule keeps trailing newlines in the same BPE word
            "1,000,000 tokens",
            "mixed 中文 with English and 日本語テキスト",
            "café naïve ±5",
            "tabs\there\t\tand\t\t\tthere\n\n\nnewlines",
        ] {
            let ids = tokenizer.encode(text: text, addSpecialTokens: false)
            let back = tokenizer.decode(tokens: ids, skipSpecialTokens: false)
            try check("round trip \(String(reflecting: text))", back == text)
        }

        // Special tokens survive encode/decode verbatim.
        let special = "<｜ref｜>"
        let specialIds = tokenizer.encode(text: special, addSpecialTokens: false)
        try check("special token encodes to itself", specialIds == [129_278])
        try check(
            "special token decodes verbatim",
            tokenizer.decode(tokens: specialIds, skipSpecialTokens: false) == special)

        report.append("DS4 tokenizer smoke passed: \(report.count) checks")
        return report.joined(separator: "\n")
    }
}
