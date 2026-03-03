// ANEAvailabilityTests.swift
// ANERuntimeTests

import Testing
@testable import ANERuntime

@Suite("ANE Availability", .serialized)
struct ANEAvailabilityTests {

    @Test("Probe returns a valid result on any machine")
    func probeReturnsResult() {
        let avail = ANEAvailability()
        let info = avail.probe()
        // On any machine, we get a definitive answer
        if info.isAvailable {
            #expect(info.reason == nil)
        } else {
            #expect(info.reason != nil)
        }
    }

    @Test("Probe result is cached across calls")
    func probeCaching() {
        let avail = ANEAvailability()
        let info1 = avail.probe()
        let info2 = avail.probe()
        #expect(info1.isAvailable == info2.isAvailable)
        #expect(info1.reason == info2.reason)
        #expect(info1.chipName == info2.chipName)
    }

    @Test("ANEAvailability is thread-safe")
    func threadSafety() async {
        let avail = ANEAvailability()
        await withTaskGroup(of: ANEAvailabilityInfo.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    avail.probe()
                }
            }
            var results: [ANEAvailabilityInfo] = []
            for await info in group {
                results.append(info)
            }
            // All concurrent probes return the same result
            let first = results[0]
            for result in results {
                #expect(result.isAvailable == first.isAvailable)
            }
        }
    }

    #if arch(arm64)
    @Test("On ARM64 Mac, chip name is detected")
    func chipNameDetected() {
        let avail = ANEAvailability()
        let info = avail.probe()
        #expect(info.chipName != nil)
        #expect(info.chipName!.contains("Apple"))
    }

    @Test("On ARM64 Mac with ANE, core count is reported")
    func coreCountDetected() {
        let avail = ANEAvailability()
        let info = avail.probe()
        if info.isAvailable {
            #expect(info.coreCount > 0)
        }
    }
    #endif
}
