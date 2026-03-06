// ANEWatchdogTests.swift
// ANERuntimeTests
//
// Tests for ANEWatchdog timeout functionality.

import Testing
import Foundation
@testable import ANERuntime

@Suite("ANE Watchdog")
struct ANEWatchdogTests {

    @Test("Fast operation completes within timeout")
    func fastOperationSucceeds() throws {
        let result = try ANEWatchdog.withTimeout(5_000) {
            return 42
        }
        #expect(result == 42)
    }

    @Test("Operation result is correctly returned")
    func resultPassthrough() throws {
        let data: [Float] = [1.0, 2.0, 3.0]
        let result = try ANEWatchdog.withTimeout(5_000) {
            return data.map { $0 * 2 }
        }
        #expect(result == [2.0, 4.0, 6.0])
    }

    @Test("Operation error propagates through watchdog")
    func errorPropagation() {
        struct TestError: Error {}

        #expect(throws: TestError.self) {
            try ANEWatchdog.withTimeout(5_000) {
                throw TestError()
            } as Int
        }
    }

    @Test("Slow operation triggers timeout")
    func timeoutTriggered() {
        #expect(throws: ANEError.self) {
            try ANEWatchdog.withTimeout(50) {
                Thread.sleep(forTimeInterval: 1.0)
                return 0
            }
        }
    }

    @Test("Timeout error message contains duration")
    func timeoutErrorMessage() {
        do {
            try ANEWatchdog.withTimeout(100) {
                Thread.sleep(forTimeInterval: 2.0)
                return 0
            }
            #expect(Bool(false), "Should have thrown")
        } catch let error as ANEError {
            let description = "\(error)"
            #expect(description.contains("100ms"))
        } catch {
            #expect(Bool(false), "Wrong error type: \(error)")
        }
    }
}
