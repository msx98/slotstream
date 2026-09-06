// Boot and generation milestones: one greppable stderr line per event, so a
// log polled during a slow, swap-throttled boot shows which action is live
// instead of minutes of silence. Unconditional by design — a trace you have
// to remember to enable is worthless at 3am. Plain Date() + stderr write:
// no locks, no MLX sync, nothing a hot path would notice.

import Foundation

public enum Milestone {
    /// `START: <name>` — returns the timestamp to hand back to `end`.
    @discardableResult
    public static func start(_ name: String) -> Date {
        let started = Date()
        FileHandle.standardError.write("START: \(name)\n".data(using: .utf8)!)
        return started
    }

    /// `END: <name> (took N.Ns)` — pass the exact name `start` printed, so a
    /// grep pairs the lines.
    public static func end(_ name: String, _ started: Date) {
        FileHandle.standardError.write(
            String(format: "END: %@ (took %.1fs)\n", name, -started.timeIntervalSinceNow)
                .data(using: .utf8)!)
    }
}
