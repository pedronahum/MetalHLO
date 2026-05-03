// CalibrateTest — verify HardwareProfile.calibrate() derives sensible constants
import Foundation
import Metal
import QuartzCore
import HeterogeneousFusion

guard let device = MTLCreateSystemDefaultDevice() else {
    print("No Metal device")
    exit(1)
}

let ramGB = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
print("Device: \(device.name), RAM: \(String(format: "%.0f", ramGB))GB")
print()

// Force fresh calibration via load() (which saves to disk)
HardwareProfile.invalidateCache()
let start = CACurrentMediaTime()
let profile = HardwareProfile.load(device: device)
let elapsed = CACurrentMediaTime() - start

print()
print("=== RESULTS (calibrated in \(String(format: "%.1f", elapsed))s) ===")
print("  minOutputElements: \(profile.minOutputElements / 1_000_000)M")
print("  minOutputColumns:  \(profile.minOutputColumns)")
print("  maxOutputBytes:    \(profile.maxOutputBytes / (1024 * 1024))MB")
print("  contentionCeiling: \(String(format: "%.2f", profile.contentionCeiling))x")
print("  concurrencyBudget: \(profile.concurrencyBudget)")
print()

// Verify cache round-trip
let profile2 = HardwareProfile.load(device: device)
let cacheOK = profile2.deviceName == profile.deviceName
    && profile2.minOutputElements == profile.minOutputElements
    && profile2.minOutputColumns == profile.minOutputColumns
    && profile2.maxOutputBytes == profile.maxOutputBytes
print("Cache round-trip: \(cacheOK ? "PASS" : "FAIL")")

// Verify constants are physically plausible. We don't anchor on M1 values
// because faster chips legitimately produce smaller minOutputElements/
// minOutputColumns (more compute means smaller workloads become profitable
// to partition) and may produce contentionCeiling < 1.0 (current fusion
// strategy not profitable on that chip — a real result, not a calibration
// error). M1 references shown alongside for context.
//
// maxOutputBytes is the one constant that scales linearly with RAM (calibrator
// targets ~25%), so it keeps a tight ±40% RAM-derived bound.
let expectedMaxBytes = Int(Double(ProcessInfo.processInfo.physicalMemory) * 0.25)
let m1Ref = (elem: 10_000_000, N: 32_768, ceil: 1.7, budget: 1)

func within(_ val: Int, _ ref: Int, _ tol: Double) -> Bool {
    let ratio = Double(val) / Double(ref)
    return ratio >= (1.0 - tol) && ratio <= (1.0 + tol)
}

let checks: [(String, Bool)] = [
    ("minOutputElements (plausible 100K–50M; M1 ref: \(m1Ref.elem / 1_000_000)M)",
     profile.minOutputElements >= 100_000 && profile.minOutputElements <= 50_000_000),
    ("minOutputColumns (plausible 256–100K; M1 ref: \(m1Ref.N))",
     profile.minOutputColumns >= 256 && profile.minOutputColumns <= 100_000),
    ("maxOutputBytes (expect ~\(expectedMaxBytes / (1024*1024))MB ±40%, ~25% of \(String(format: "%.0f", ramGB))GB)",
     within(profile.maxOutputBytes, expectedMaxBytes, 0.40)),
    ("contentionCeiling (plausible 0.5x–3.0x; M1 ref: \(String(format: "%.1f", m1Ref.ceil))x; <1.0 means fusion gated off)",
     profile.contentionCeiling >= 0.5 && profile.contentionCeiling <= 3.0),
    ("concurrencyBudget (plausible 1–4; M1 ref: \(m1Ref.budget))",
     profile.concurrencyBudget >= 1 && profile.concurrencyBudget <= 4),
]

print()
print("=== VALIDATION (plausible-range bounds) ===")
var allPass = true
for (name, ok) in checks {
    print("  \(ok ? "PASS" : "FAIL"): \(name)")
    if !ok { allPass = false }
}
print()
print(allPass ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED")

// Clean up
HardwareProfile.invalidateCache()
