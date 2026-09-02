// Chunk index for the disk-persisted prefix KV cache.
//
// One chunk = one prefix of the prompt at a `prefillChunk` boundary (default
// 4096 tokens). Chunks are chained by parent: chunk N's key is
// `sha256(parent_sha || sha256(chunk_embeddings))`, so two conversations
// that share the first 4096 tokens but diverge at position 4097 get two
// distinct keys at depth 1 and never alias.
//
// On disk, each chunk lives at `kvcache_dir/<key>/` with three files:
//   - `data.kv` — the saved state arrays (KVCache/Indexer/LinearCache/Multi/MTP)
//   - `parent_sha.bin` — 32 bytes, the parent chunk's key (or empty if depth 0)
//   - `emb_sha.bin` — 32 bytes, the sha256 of the chunk's embeddings
//
// The metadata DB (SQLite, journal_mode=WAL) holds:
//   - last_used: timestamp of most recent save or load
//   - size_bytes: total bytes occupied by this chunk's directory
//   - parent_sha: parent's key (NULL for depth 0)
//
// Eviction: when total size > SLOTSTREAM_KVCACHE_MAX_GB (default 20),
// the saver deletes a forest of leaves (and orphans whose parent is gone)
// whose combined size clears the deficit, choosing the lowest-value
// chunks first (where a chunk's value is the max of its own and all its
// descendants' last_used, so leaves with live children are protected).
//
// Crashes mid-save leave a `.kv.partial` file in the chunk directory; the
// next save for that key detects the partial and rewrites. The metadata DB
// is updated only after the `.kv.partial` is renamed atomically to `data.kv`,
// so a crash before rename leaves no DB row pointing at partial data.

import Foundation
import CryptoKit
import SQLite3

/// Process-wide SQLite handle. Single connection per process; serialized
/// internally. WAL mode tolerates concurrent readers (future split: stats
/// tool) but writers serialize, which is what we want.
public final class ChunkIndex {
    static let shared = ChunkIndex()

    private var db: OpaquePointer?
    private let serial = DispatchQueue(label: "slotstream.chunkindex")
    private let dbPath: URL

    private init() {
        dbPath = DiskCache.dir.appendingPathComponent("metadata.db")
        try? FileManager.default.createDirectory(
            at: DiskCache.dir, withIntermediateDirectories: true)
        if sqlite3_open(dbPath.path, &db) != SQLITE_OK {
            FileHandle.standardError.write(
                Data("[kvcache] metadata.db open failed: \(String(cString: sqlite3_errmsg(db))) — disk cache disabled\n".utf8))
            db = nil
            return
        }
        // WAL: readers don't block writers, one writer at a time.
        sqlite3_exec(db, "PRAGMA journal_mode=WAL;", nil, nil, nil)
        sqlite3_exec(db, "PRAGMA synchronous=NORMAL;", nil, nil, nil)
        sqlite3_exec(db, "PRAGMA foreign_keys=ON;", nil, nil, nil)
        let schema = """
        CREATE TABLE IF NOT EXISTS chunks (
          key TEXT PRIMARY KEY,
          parent_sha TEXT,
          depth INTEGER NOT NULL,
          size_bytes INTEGER NOT NULL,
          last_used REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS chunks_parent ON chunks(parent_sha);
        CREATE INDEX IF NOT EXISTS chunks_last_used ON chunks(last_used);
        """
        // No FK on parent_sha: a child chunk can be inserted before its
        // parent (both are queued together in the save path), and SQLite's
        // foreign_keys check would reject the child until the parent's row
        // is visible. The chain walk stops at the first missing key anyway,
        // so the orphan is harmless — sweepOrphans() drops it on the next
        // eviction pass if nothing reconciles it.
        sqlite3_exec(db, "PRAGMA foreign_keys=OFF;", nil, nil, nil)
        if sqlite3_exec(db, schema, nil, nil, nil) != SQLITE_OK {
            FileHandle.standardError.write(
                Data("[kvcache] schema init failed: \(String(cString: sqlite3_errmsg(db)))\n".utf8))
        }
    }

    deinit { if db != nil { sqlite3_close(db) } }

    /// Stable key derivation. `parentSha` is nil for depth 0.
    public static func makeKey(parentSha: String?, embeddings: [Float]) -> String {
        var hasher = SHA256()
        if let p = parentSha { hasher.update(data: Data(p.utf8)) }
        // Hash the embeddings in deterministic little-endian order. The caller
        // passes a flat [Float] slice of the chunk's dequantized embedding
        // matrix; any byte-identical prefix produces the same hash, so the
        // chain matches across processes that re-derive from the same weights.
        let n = embeddings.count
        var buf = Data(capacity: n * 4)
        for v in embeddings {
            var f = v.bitPattern.littleEndian
            withUnsafeBytes(of: &f) { buf.append(contentsOf: $0) }
        }
        hasher.update(data: buf)
        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

/// Walk the chain depth-by-depth. For each depth i, the caller supplies
/// the embeddings of tokens `[i*chunk, (i+1)*chunk)` via `embedForDepth`.
/// We compute the per-chunk key, look it up, and stop at the first miss.
/// The expensive part (embedding extraction) is bounded to the actual chain
/// depth we end up consuming.
public func chainLength(
    chunk: Int, embedForDepth: (Int) -> [Float]?
) -> Int? {
    guard chunk > 0 else { return nil }
    return serial.sync {
        var prevKey: String? = nil
        var depth = 0
        while true {
            guard let emb = embedForDepth(depth) else { return depth == 0 ? nil : depth }
            let key = Self.makeKey(parentSha: prevKey, embeddings: emb)
            let stmt = prepare("SELECT 1 FROM chunks WHERE key = ? LIMIT 1;")
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, key, -1, SQLITE_TRANSIENT)
            guard sqlite3_step(stmt) == SQLITE_ROW else { return depth == 0 ? nil : depth }
            touchInternal(key: key)
            prevKey = key
            depth += 1
        }
    }
}

    /// Record a freshly saved chunk. Returns the resolved key (caller needs
    /// it to write the parent_sha.bin and emb_sha.bin files).
    public func register(
        key: String, parentSha: String?, depth: Int, sizeBytes: Int
    ) {
        serial.sync {
            let stmt = prepare("""
            INSERT INTO chunks(key, parent_sha, depth, size_bytes, last_used)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
              parent_sha=excluded.parent_sha,
              depth=excluded.depth,
              size_bytes=excluded.size_bytes,
              last_used=excluded.last_used;
            """)
            defer { sqlite3_finalize(stmt) }
            let now = Date().timeIntervalSince1970
            sqlite3_bind_text(stmt, 1, key, -1, SQLITE_TRANSIENT)
            if let p = parentSha {
                sqlite3_bind_text(stmt, 2, p, -1, SQLITE_TRANSIENT)
            } else {
                sqlite3_bind_null(stmt, 2)
            }
            sqlite3_bind_int(stmt, 3, Int32(depth))
            sqlite3_bind_int64(stmt, 4, Int64(sizeBytes))
            sqlite3_bind_double(stmt, 5, now)
            if sqlite3_step(stmt) != SQLITE_DONE {
                FileHandle.standardError.write(
                    Data("[kvcache] register failed for \(key.prefix(12)): \(String(cString: sqlite3_errmsg(db)))\n".utf8))
            }
        }
    }

    /// Update last_used on a hit so LRU sorts correctly.
    public func touch(key: String) {
        serial.sync { touchInternal(key: key) }
    }

    private func touchInternal(key: String) {
        let stmt = prepare("UPDATE chunks SET last_used = ? WHERE key = ?;")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_double(stmt, 1, Date().timeIntervalSince1970)
        sqlite3_bind_text(stmt, 2, key, -1, SQLITE_TRANSIENT)
        sqlite3_step(stmt)
    }

    /// Total disk usage summed across rows. Includes the directory itself,
    /// not the kvcache parent dir.
    public func totalBytes() -> Int {
        serial.sync {
            let stmt = prepare("SELECT COALESCE(SUM(size_bytes), 0) FROM chunks;")
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
            return Int(sqlite3_column_int64(stmt, 0))
        }
    }

    /// Evict leaves-first to free at least `bytesNeeded` bytes. Returns bytes
    /// actually freed. The "value" of a chunk = max(its own and all its
    /// descendants' last_used), recomputed before selection so a parent with
    /// a live child is implicitly protected by inheriting the child's time.
    public func evictLeaves(bytesNeeded: Int, maxBytes: Int) -> Int {
        serial.sync {
            // Lift parent.last_used to max(children.last_used) for non-leaves.
            // SQLite has no recursive CTE update that's cheap; instead, iterate
            // depths from deepest to 0, recomputing each non-leaf as
            // max(self.last_used, max(child.last_used)). Children whose parent
            // has been evicted are reclassified as roots (parent_sha NULL).
            let maxDepth = intScalar("SELECT COALESCE(MAX(depth), -1) FROM chunks;")
            if maxDepth < 0 { return 0 }
            for d in stride(from: maxDepth, through: 1, by: -1) {
                let upd = prepare("""
                UPDATE chunks
                SET last_used = (
                  SELECT MAX(c2.last_used)
                  FROM chunks c2
                  WHERE c2.parent_sha = chunks.key
                )
                WHERE depth = ?
                  AND EXISTS (SELECT 1 FROM chunks c2 WHERE c2.parent_sha = chunks.key);
                """)
                sqlite3_bind_int(upd, 1, Int32(d))
                sqlite3_step(upd)
                sqlite3_finalize(upd)
            }

            // Now pick chunks in last_used ASC order until bytesNeeded is met.
            // Skip any chunk that still has children — by construction, a chunk
            // with children has last_used >= its oldest descendant, but we
            // only want to evict leaves here so we don't dangle trees.
            var freed = 0
            let stmt = prepare("""
            SELECT key, size_bytes FROM chunks
            WHERE NOT EXISTS (SELECT 1 FROM chunks c2 WHERE c2.parent_sha = chunks.key)
            ORDER BY last_used ASC;
            """)
            defer { sqlite3_finalize(stmt) }
            var toDelete: [(String, Int)] = []
            while sqlite3_step(stmt) == SQLITE_ROW {
                let keyC = sqlite3_column_text(stmt, 0)
                let size = Int(sqlite3_column_int64(stmt, 1))
                let key = keyC.map { String(cString: $0) } ?? ""
                toDelete.append((key, size))
            }
            for (key, size) in toDelete {
                if freed >= bytesNeeded && totalBytesScalar() <= maxBytes { break }
                deleteChunk(key: key)
                freed += size
                FileHandle.standardError.write(
                    Data("[kvcache] evicted \(key.prefix(12)) (\(size) bytes)\n".utf8))
            }
            return freed
        }
    }

    private func deleteChunk(key: String) {
        // Drop the DB row first (CASCADE orphans to parent_sha NULL). The
        // directory is removed separately by the caller — we don't have
        // access to the file system here, and the caller already needs to
        // know the chunk path anyway.
        let stmt = prepare("DELETE FROM chunks WHERE key = ?;")
        sqlite3_bind_text(stmt, 1, key, -1, SQLITE_TRANSIENT)
        sqlite3_step(stmt)
        sqlite3_finalize(stmt)
    }

    public func remove(key: String) {
        serial.sync { deleteChunk(key: key) }
    }

    /// Read parent_sha back (or nil for depth 0).
    public func parentSha(key: String) -> String? {
        serial.sync {
            let stmt = prepare("SELECT parent_sha FROM chunks WHERE key = ? LIMIT 1;")
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, key, -1, SQLITE_TRANSIENT)
            guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
            guard let p = sqlite3_column_text(stmt, 0) else { return nil }
            return String(cString: p)
        }
    }

    // MARK: - low-level

    private func prepare(_ sql: String) -> OpaquePointer? {
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) != SQLITE_OK {
            FileHandle.standardError.write(
                Data("[kvcache] prepare failed: \(String(cString: sqlite3_errmsg(db))) sql=\(sql)\n".utf8))
            return nil
        }
        return stmt
    }

    private func intScalar(_ sql: String) -> Int {
        let stmt = prepare(sql)
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
        return Int(sqlite3_column_int64(stmt, 0))
    }

    private func totalBytesScalar() -> Int { intScalar("SELECT COALESCE(SUM(size_bytes),0) FROM chunks;") }
}

// SQLite needs the SQLITE_TRANSIENT macro; Swift's SQLite3 module exposes
// it as `unsafeBitCast(-1, to: sqlite3_destructor_type.self)` historically,
// but the Swift module also defines `SQLITE_TRANSIENT` as a global.
internal let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)