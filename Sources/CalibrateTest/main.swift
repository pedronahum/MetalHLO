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

// Verify constants are in reasonable ranges
let manual = (elem: 10_000_000, N: 32_768, mem: 2048 * 1024 * 1024, ceil: 1.7, budget: 1)
let tolerance = 0.40  // 40% tolerance vs manual M1 constants

func within(_ val: Int, _ ref: Int, _ tol: Double) -> Bool {
    let ratio = Double(val) / Double(ref)
    return ratio >= (1.0 - tol) && ratio <= (1.0 + tol)
}

let checks: [(String, Bool)] = [
    ("minOutputElements (expect ~\(manual.elem / 1_000_000)M ±40%)",
     within(profile.minOutputElements, manual.elem, tolerance)),
    ("minOutputColumns (expect ~\(manual.N) ±40%)",
     within(profile.minOutputColumns, manual.N, tolerance)),
    ("maxOutputBytes (expect ~\(manual.mem / (1024*1024))MB ±40%)",
     within(profile.maxOutputBytes, manual.mem, tolerance)),
    ("contentionCeiling (expect ~\(manual.ceil)x)",
     profile.contentionCeiling >= 1.0 && profile.contentionCeiling <= 3.0),
    ("concurrencyBudget (expect \(manual.budget))",
     profile.concurrencyBudget >= 1 && profile.concurrencyBudget <= 3),
]

print()
print("=== VALIDATION vs MANUAL M1 CONSTANTS ===")
var allPass = true
for (name, ok) in checks {
    print("  \(ok ? "PASS" : "FAIL"): \(name)")
    if !ok { allPass = false }
}
print()
print(allPass ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED")

// Clean up
HardwareProfile.invalidateCache()
