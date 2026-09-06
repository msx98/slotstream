// GGUF v3 reader: header-only parsing of the metadata key/value block and the
// tensor directory, pread-based (open + pread, matching Checkpoint.swift).
// Tensor data is never read here — the next layer preads it from `path` using
// `dataSectionOffset` + `GGUFTensorInfo.dataOffset`.

import Foundation

// MARK: - Tensor types

/// The GGUF tensor type IDs this model family uses. Unknown IDs parse fine
/// (with a nil `byteSize`) so metadata of newer files stays readable; only
/// tensor math on them is refused.
public enum GGUFTensorType: Equatable, Sendable {
    case f32      // 0
    case f16      // 1
    case q8_0     // 8: 34 B per 32 elems (F16 scale + 32 x i8)
    case i32      // 26
    case bf16     // 30
    case mxfp4    // 39: this family's MXFP4, 17 B per 32 elems (1 E8M0 scale byte + 16 nibble bytes)
    case unknown(UInt32)

    init(raw: UInt32) {
        switch raw {
        case 0: self = .f32
        case 1: self = .f16
        case 8: self = .q8_0
        case 26: self = .i32
        case 30: self = .bf16
        case 39: self = .mxfp4
        default: self = .unknown(raw)
        }
    }
}

// MARK: - Quant block math

/// Byte math for the quantized formats in this family. All sizes are derived
/// from these two numbers; nothing else may hardcode them.
public enum GGUFQuant {
    /// Elements per quantization block (32 for block quants, 1 for dense
    /// formats). Nil for unknown tensor types.
    public static func blockSize(_ type: GGUFTensorType) -> Int? {
        switch type {
        case .q8_0, .mxfp4: return 32
        case .f32, .f16, .bf16, .i32: return 1
        case .unknown: return nil
        }
    }

    /// Bytes per block: Q8_0 is 34 (F16 scale + 32 x i8), MXFP4 is 17 (one
    /// E8M0 scale byte + 16 nibble bytes per 32 elements).
    public static func bytesPerBlock(_ type: GGUFTensorType) -> Int? {
        switch type {
        case .f32: return 4
        case .f16, .bf16: return 2
        case .i32: return 4
        case .q8_0: return 34
        case .mxfp4: return 17
        case .unknown: return nil
        }
    }

    /// Byte size of `nElems` elements of `type`. Nil for unknown types, or
    /// when `nElems` is not a whole number of blocks (which the format
    /// forbids — the tensor data would be corrupt).
    public static func rowSize(_ type: GGUFTensorType, nElems: Int) -> Int? {
        guard let elems = blockSize(type), let bytes = bytesPerBlock(type), nElems >= 0,
            nElems % elems == 0
        else { return nil }
        return nElems / elems * bytes
    }
}

// MARK: - Values

/// A parsed GGUF metadata value. Arrays keep their elements as values; the
/// accessors below are the ergonomic way in.
public enum GGUFValue: Sendable {
    case u8(UInt8)
    case i8(Int8)
    case u16(UInt16)
    case i16(Int16)
    case u32(UInt32)
    case i32(Int32)
    case f32(Float)
    case bool(Bool)
    case string(String)
    case u64(UInt64)
    case i64(Int64)
    case f64(Double)
    case array([GGUFValue])

    public var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    /// Widening integer read: any integer or bool case as Int (false = 0).
    public var intValue: Int? {
        switch self {
        case .u8(let v): return Int(v)
        case .i8(let v): return Int(v)
        case .u16(let v): return Int(v)
        case .i16(let v): return Int(v)
        case .u32(let v): return Int(v)
        case .i32(let v): return Int(v)
        case .u64(let v): return Int(exactly: v)
        case .i64(let v): return Int(exactly: v)
        case .bool(let v): return v ? 1 : 0
        default: return nil
        }
    }

    /// Widening float read: f32/f64 (and integers) as Double.
    public var doubleValue: Double? {
        switch self {
        case .f32(let v): return Double(v)
        case .f64(let v): return v
        case .u8(let v): return Double(v)
        case .i8(let v): return Double(v)
        case .u16(let v): return Double(v)
        case .i16(let v): return Double(v)
        case .u32(let v): return Double(v)
        case .i32(let v): return Double(v)
        case .u64(let v): return Double(exactly: v)
        case .i64(let v): return Double(exactly: v)
        default: return nil
        }
    }

    public var boolValue: Bool? {
        if case .bool(let v) = self { return v }
        return nil
    }

    public var arrayValue: [GGUFValue]? {
        if case .array(let v) = self { return v }
        return nil
    }
}

// MARK: - Tensor directory

public struct GGUFTensorInfo: Sendable {
    public let name: String
    /// Dimensions in file order: `dims[0]` is the fastest-varying (innermost)
    /// axis, i.e. the reverse of the numpy shape llama.cpp tooling prints.
    public let dims: [Int]
    public let type: GGUFTensorType
    /// Byte offset relative to the start of the data section (see
    /// `GGUFFile.dataSectionOffset`).
    public let dataOffset: UInt64
    /// Type-derived byte size. Nil only for tensor types this reader does not
    /// know (`.unknown`); every known type has an exact size.
    public let byteSize: Int?
}

// MARK: - File

/// Header-only GGUF v3 reader. `init` parses the fixed header, the metadata
/// key/value block and the tensor directory through a sliding pread window,
/// then closes the descriptor: for a 156 GB file only the first few MB are
/// ever touched. Tensor bytes are read by later layers, which pread at
/// `dataSectionOffset + GGUFTensorInfo.dataOffset`.
public struct GGUFFile: Sendable {
    public let path: String
    public let metadata: [String: GGUFValue]
    public let tensors: [GGUFTensorInfo]
    /// File offset where tensor data begins: the first offset after the
    /// tensor directory, aligned up to `general.alignment` (default 32).
    public let dataSectionOffset: UInt64

    public init(path: String) throws {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw ModelError("cannot open GGUF \(path): \(String(cString: strerror(errno)))")
        }
        defer { close(fd) }

        var cursor = PreadCursor(fd: fd, path: path)

        let magic = try cursor.readBytes(4)
        guard magic == Array("GGUF".utf8) else {
            throw ModelError("\(path) is not a GGUF file (bad magic)")
        }
        let version = try cursor.readU32()
        guard version == 3 else {
            throw ModelError("\(path) is GGUF version \(version); this reader requires v3")
        }
        let tensorCount = try cursor.readU64()
        let kvCount = try cursor.readU64()
        guard tensorCount <= Self.maxCount, kvCount <= Self.maxCount else {
            throw ModelError("\(path) has an implausible tensor/kv count (\(tensorCount)/\(kvCount)) — file is corrupt")
        }

        var metadata: [String: GGUFValue] = [:]
        metadata.reserveCapacity(Int(kvCount))
        for _ in 0 ..< kvCount {
            let key = try cursor.readString()
            let type = try cursor.readU32()
            let value = try Self.readValue(type, from: &cursor)
            guard metadata[key] == nil else {
                throw ModelError("\(path) has duplicate metadata key \(key)")
            }
            metadata[key] = value
        }

        var tensors: [GGUFTensorInfo] = []
        tensors.reserveCapacity(Int(tensorCount))
        for _ in 0 ..< tensorCount {
            let name = try cursor.readString()
            let nDims = try cursor.readU32()
            guard nDims >= 1, nDims <= 8 else {
                throw ModelError("\(path): tensor \(name) has \(nDims) dimensions — outside 1...8")
            }
            var dims: [Int] = []
            dims.reserveCapacity(Int(nDims))
            var nElems = 1
            for _ in 0 ..< nDims {
                let d = try cursor.readU64()
                guard d <= UInt64(Int.max) else {
                    throw ModelError("\(path): tensor \(name) dimension \(d) overflows addressable size")
                }
                dims.append(Int(d))
                let (next, overflow) = nElems.multipliedReportingOverflow(by: Int(d))
                guard !overflow else {
                    throw ModelError("\(path): tensor \(name) element count overflows")
                }
                nElems = next
            }
            let type = GGUFTensorType(raw: try cursor.readU32())
            let offset = try cursor.readU64()
            if let elems = GGUFQuant.blockSize(type), nElems % elems != 0 {
                throw ModelError("\(path): tensor \(name) has \(nElems) elements, not a multiple of the \(type) block size \(elems)")
            }
            tensors.append(GGUFTensorInfo(
                name: name, dims: dims, type: type, dataOffset: offset,
                byteSize: GGUFQuant.rowSize(type, nElems: nElems)))
        }

        // The data section starts after the tensor directory, aligned up to
        // general.alignment (u32, power of two, default 32).
        var alignment = 32
        if let v = metadata["general.alignment"] {
            guard let a = v.intValue, a >= 1, a <= 1024, a.nonzeroBitCount == 1 else {
                throw ModelError("\(path): general.alignment must be a power of two in 1...1024, got \(v)")
            }
            alignment = a
        }
        let unaligned = cursor.position
        let dataOffset = (unaligned + UInt64(alignment - 1)) & ~UInt64(alignment - 1)

        self.path = path
        self.metadata = metadata
        self.tensors = tensors
        self.dataSectionOffset = dataOffset
    }

    public func kv(_ key: String) -> GGUFValue? {
        metadata[key]
    }

    public func tensor(named name: String) -> GGUFTensorInfo? {
        tensors.first { $0.name == name }
    }

    static let maxCount: UInt64 = 1 << 24

    // MARK: value parsing

    private static func readValue(_ type: UInt32, from cursor: inout PreadCursor) throws -> GGUFValue {
        switch type {
        case 0: return .u8(try cursor.readU8())
        case 1: return .i8(Int8(bitPattern: try cursor.readU8()))
        case 2: return .u16(try cursor.readU16())
        case 3: return .i16(Int16(bitPattern: try cursor.readU16()))
        case 4: return .u32(try cursor.readU32())
        case 5: return .i32(Int32(bitPattern: try cursor.readU32()))
        case 6: return .f32(Float(bitPattern: try cursor.readU32()))
        case 7: return .bool(try cursor.readU8() != 0)
        case 8: return .string(try cursor.readString())
        case 9:
            let elemType = try cursor.readU32()
            let count = try cursor.readU64()
            guard count <= maxCount else {
                throw ModelError("\(cursor.path): array metadata with \(count) elements — implausible")
            }
            guard elemType != 9 else {
                throw ModelError("\(cursor.path): nested GGUF arrays are not supported")
            }
            var values: [GGUFValue] = []
            values.reserveCapacity(Int(count))
            for _ in 0 ..< count {
                values.append(try readValue(elemType, from: &cursor))
            }
            return .array(values)
        case 10: return .u64(try cursor.readU64())
        case 11: return .i64(Int64(bitPattern: try cursor.readU64()))
        case 12: return .f64(Double(bitPattern: try cursor.readU64()))
        default:
            throw ModelError("\(cursor.path): unknown GGUF metadata value type \(type)")
        }
    }
}

// MARK: - Pread cursor

/// A sliding window over the file, filled with pread in 64 KB chunks so the
/// header parse never holds more than the window in memory even if a single
/// metadata string is large. Field reads beyond the window refill it.
private struct PreadCursor {
    let fd: Int32
    let path: String

    init(fd: Int32, path: String) {
        self.fd = fd
        self.path = path
    }

    private var buffer: [UInt8] = []
    private var start = 0  // parse cursor within buffer
    private var end = 0  // one past the last valid byte in buffer
    private var bufferFileOffset: UInt64 = 0  // absolute file offset of buffer[0]

    /// Absolute file position of the parse cursor.
    var position: UInt64 { bufferFileOffset + UInt64(start) }

    /// Guarantees `count` readable bytes starting at the cursor.
    private mutating func ensure(_ count: Int) throws {
        guard count > end - start else { return }
        guard count <= 1 << 26 else {
            throw ModelError("\(path): GGUF header field of \(count) bytes exceeds the 64 MB sanity limit")
        }
        if start > 0 {
            buffer.removeFirst(start)
            end -= start
            bufferFileOffset += UInt64(start)
            start = 0
        }
        if buffer.count < count {
            buffer.reserveCapacity(max(count, buffer.count * 2, 1 << 16))
            buffer.append(contentsOf: repeatElement(0, count: count - buffer.count))
        }
        while end - start < count {
            let want = count - (end - start)
            let at = end
            let got = buffer.withUnsafeMutableBytes { raw -> Int in
                Foundation.pread(fd, raw.baseAddress!.advanced(by: at), want, off_t(bufferFileOffset) + off_t(at))
            }
            guard got > 0 else {
                throw ModelError("\(path): unexpected end of GGUF header at byte \(bufferFileOffset + UInt64(end))")
            }
            end += got
        }
    }

    mutating func readU8() throws -> UInt8 {
        try ensure(1)
        defer { start += 1 }
        return buffer[start]
    }

    mutating func readU16() throws -> UInt16 {
        try ensure(2)
        defer { start += 2 }
        return UInt16(buffer[start]) | UInt16(buffer[start + 1]) << 8
    }

    mutating func readU32() throws -> UInt32 {
        try ensure(4)
        defer { start += 4 }
        return UInt32(buffer[start]) | UInt32(buffer[start + 1]) << 8
            | UInt32(buffer[start + 2]) << 16 | UInt32(buffer[start + 3]) << 24
    }

    mutating func readU64() throws -> UInt64 {
        try ensure(8)
        defer { start += 8 }
        var v: UInt64 = 0
        for i in 0 ..< 8 { v |= UInt64(buffer[start + i]) << (8 * UInt64(i)) }
        return v
    }

    /// u64 length + UTF-8 bytes (GGUF strings are length-prefixed, never
    /// NUL-terminated).
    mutating func readString() throws -> String {
        let len = try readU64()
        guard len <= 1 << 26 else {
            throw ModelError("\(path): GGUF string of \(len) bytes exceeds the 64 MB sanity limit")
        }
        let bytes = try readBytes(Int(len))
        guard let s = String(bytes: bytes, encoding: .utf8) else {
            throw ModelError("\(path): GGUF string at byte \(position) is not valid UTF-8")
        }
        return s
    }

    mutating func readBytes(_ count: Int) throws -> [UInt8] {
        try ensure(count)
        defer { start += count }
        return Array(buffer[start ..< start + count])
    }
}
