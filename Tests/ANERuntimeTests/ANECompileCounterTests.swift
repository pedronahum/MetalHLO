// ANECompileCounterTests.swift
// ANERuntimeTests

import Testing
@testable import ANERuntime

@Suite("ANE Compile Counter", .serialized)
struct ANECompileCounterTests {

    @Test("Counter starts at zero")
    func initialCount() {
        let counter = CompileCounter()
        #expect(counter.currentCount == 0)
        #expect(counter.canCompile)
        #expect(!counter.isWarning)
    }

    @Test("Counter increments correctly")
    func increment() throws {
        let counter = CompileCounter(warningLimit: 5, maximumLimit: 10)
        for i in 1...4 {
            let count = try counter.increment()
            #expect(count == i)
        }
        #expect(counter.canCompile)
        #expect(!counter.isWarning)
    }

    @Test("Counter warns at warning limit")
    func warningLimit() throws {
        let counter = CompileCounter(warningLimit: 3, maximumLimit: 10)
        for _ in 1...3 {
            _ = try counter.increment()
        }
        #expect(counter.isWarning)
        #expect(counter.canCompile)
    }

    @Test("Counter reports canCompile=false at hard limit")
    func hardLimitCanCompile() throws {
        let counter = CompileCounter(warningLimit: 2, maximumLimit: 3)
        _ = try counter.increment()  // 1
        _ = try counter.increment()  // 2
        _ = try counter.increment()  // 3 — at limit
        #expect(!counter.canCompile)
    }

    @Test("Counter throws at hard limit")
    func hardLimitThrows() throws {
        let counter = CompileCounter(warningLimit: 2, maximumLimit: 3)
        _ = try counter.increment()  // 1
        _ = try counter.increment()  // 2
        _ = try counter.increment()  // 3
        #expect(throws: ANEError.self) {
            _ = try counter.increment()  // 4 — exceeds limit
        }
    }

    @Test("Counter reset works")
    func reset() throws {
        let counter = CompileCounter(warningLimit: 2, maximumLimit: 5)
        _ = try counter.increment()
        _ = try counter.increment()
        #expect(counter.isWarning)
        counter.reset()
        #expect(counter.currentCount == 0)
        #expect(!counter.isWarning)
        #expect(counter.canCompile)
    }

    @Test("Counter is thread-safe")
    func concurrency() async throws {
        let counter = CompileCounter(warningLimit: 50, maximumLimit: 100)
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<50 {
                group.addTask {
                    _ = try? counter.increment()
                }
            }
        }
        #expect(counter.currentCount == 50)
    }

    @Test("Default limits match documented values")
    func defaultLimits() {
        #expect(CompileCounter.defaultWarningLimit == 100)
        #expect(CompileCounter.defaultHardLimit == 115)
    }
}
