// OfficialTestManager.swift
// MetalHLOTests
//
// Manager for official StableHLO tests from:
// https://github.com/openxla/stablehlo/tree/main/stablehlo/tests
//
// Mirrors the official directory structure:
// - tests/interpret/   - Interpreter conformance tests (98 tests)
// - tests/math/        - Math operation tests (47 tests)
// - tests/transforms/  - Transformation tests (19 tests)
// - tests/chlo/        - CHLO operation tests (2 tests)
// - tests/vhlo/        - VHLO versioning tests (54 tests)
// - tests/ops/         - Standalone operation tests (28 tests)

import Foundation

/// Test categories matching StableHLO test directories
public enum StableHLOTestCategory: String, CaseIterable, Sendable {
    case interpret = "interpret"
    case math = "math"
    case transforms = "transforms"
    case chlo = "chlo"
    case vhlo = "vhlo"
    case ops = "ops"

    /// GitHub raw URL base for this category
    var baseURL: String {
        switch self {
        case .ops:
            // ops files are in the tests root directory
            return "https://raw.githubusercontent.com/openxla/stablehlo/main/stablehlo/tests/"
        default:
            return "https://raw.githubusercontent.com/openxla/stablehlo/main/stablehlo/tests/\(rawValue)/"
        }
    }

    /// GitHub API URL for listing files
    var apiURL: String {
        switch self {
        case .ops:
            return "https://api.github.com/repos/openxla/stablehlo/contents/stablehlo/tests"
        default:
            return "https://api.github.com/repos/openxla/stablehlo/contents/stablehlo/tests/\(rawValue)"
        }
    }
}

/// Manager for fetching and caching official StableHLO tests
public final class OfficialTestManager: @unchecked Sendable {

    public static let shared = OfficialTestManager()

    /// Directory containing bundled test files in the project
    private let bundledTestsDir: URL?

    /// Fallback cache directory for downloaded tests
    private let cacheDir: URL

    /// In-memory file list cache (protected by cacheQueue)
    private var fileListCache: [StableHLOTestCategory: [String]] = [:]

    /// Serial queue for thread-safe cache access
    private let cacheQueue = DispatchQueue(label: "com.metalhlo.testmanager.cache")

    private init() {
        // First, try to find bundled tests in the project directory
        // This file is at: Tests/MetalHLOTests/Conformance/StableHLO/OfficialTestManager.swift
        // Test files are at: Tests/MetalHLOTests/Conformance/StableHLO/{category}/
        let thisFile = URL(fileURLWithPath: #file)
        self.bundledTestsDir = thisFile.deletingLastPathComponent()

        // Fallback cache in ~/.cache/metalhlo/stablehlo-tests/
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        self.cacheDir = homeDir
            .appendingPathComponent(".cache")
            .appendingPathComponent("metalhlo")
            .appendingPathComponent("stablehlo-tests")
    }

    /// Get bundled test directory for a category (if available)
    public func bundledDirectory(for category: StableHLOTestCategory) -> URL? {
        guard let bundledDir = bundledTestsDir else { return nil }
        let categoryDir = bundledDir.appendingPathComponent(category.rawValue)
        return FileManager.default.fileExists(atPath: categoryDir.path) ? categoryDir : nil
    }

    /// Get cache directory for a category
    public func cacheDirectory(for category: StableHLOTestCategory) -> URL {
        cacheDir.appendingPathComponent(category.rawValue)
    }

    /// Ensure a test file is available locally, downloading if needed
    public func ensureTestFile(_ filename: String, category: StableHLOTestCategory) async throws -> URL {
        // First, check bundled tests in project directory
        if let bundledDir = bundledDirectory(for: category) {
            let bundledURL = bundledDir.appendingPathComponent(filename)
            if FileManager.default.fileExists(atPath: bundledURL.path) {
                return bundledURL
            }
        }

        // Fall back to cache directory
        let categoryDir = cacheDirectory(for: category)
        let localURL = categoryDir.appendingPathComponent(filename)

        // Create directory if needed
        try FileManager.default.createDirectory(at: categoryDir, withIntermediateDirectories: true)

        // Check if file exists in cache
        if FileManager.default.fileExists(atPath: localURL.path) {
            return localURL
        }

        // Download from GitHub
        let remoteURL = URL(string: category.baseURL + filename)!
        let (data, response) = try await URLSession.shared.data(from: remoteURL)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw OfficialTestError.downloadFailed(filename)
        }

        // Save to local cache
        try data.write(to: localURL)

        return localURL
    }

    /// List all available test files in a category
    public func listTestFiles(in category: StableHLOTestCategory) async throws -> [String] {
        // Check memory cache (thread-safe read)
        let cached: [String]? = cacheQueue.sync { fileListCache[category] }
        if let cached = cached {
            return cached
        }

        // First, try to list bundled tests
        if let bundledDir = bundledDirectory(for: category) {
            if let files = try? FileManager.default.contentsOfDirectory(atPath: bundledDir.path) {
                let mlirFiles = files.filter { $0.hasSuffix(".mlir") }.sorted()
                if !mlirFiles.isEmpty {
                    cacheQueue.sync { fileListCache[category] = mlirFiles }
                    return mlirFiles
                }
            }
        }

        // Fall back to GitHub API
        let apiURL = URL(string: category.apiURL)!
        var request = URLRequest(url: apiURL)
        request.setValue("application/vnd.github.v3+json", forHTTPHeaderField: "Accept")

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw OfficialTestError.listFailed(category.rawValue)
        }

        // Parse JSON response
        guard let items = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw OfficialTestError.invalidResponse
        }

        // Extract .mlir filenames
        let files = items.compactMap { item -> String? in
            guard let name = item["name"] as? String,
                  name.hasSuffix(".mlir") else {
                return nil
            }
            return name
        }.sorted()

        // Cache result (thread-safe write)
        cacheQueue.sync { fileListCache[category] = files }

        return files
    }

    /// Download all test files in a category
    public func downloadAllTests(in category: StableHLOTestCategory, progress: ((Int, Int) -> Void)? = nil) async throws {
        let files = try await listTestFiles(in: category)
        let total = files.count

        for (index, filename) in files.enumerated() {
            _ = try await ensureTestFile(filename, category: category)
            progress?(index + 1, total)
        }
    }

    /// Get all locally cached test files in a category
    public func cachedTestFiles(in category: StableHLOTestCategory) -> [URL] {
        let categoryDir = cacheDirectory(for: category)

        guard let files = try? FileManager.default.contentsOfDirectory(
            at: categoryDir,
            includingPropertiesForKeys: nil
        ) else {
            return []
        }

        return files.filter { $0.pathExtension == "mlir" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// Clear cache for a category
    public func clearCache(for category: StableHLOTestCategory) throws {
        let categoryDir = cacheDirectory(for: category)
        try FileManager.default.removeItem(at: categoryDir)
    }

    /// Clear all caches
    public func clearAllCaches() throws {
        try FileManager.default.removeItem(at: cacheDir)
    }
}

/// Errors for official test management
public enum OfficialTestError: Error, CustomStringConvertible {
    case downloadFailed(String)
    case listFailed(String)
    case invalidResponse
    case testNotFound(String)

    public var description: String {
        switch self {
        case .downloadFailed(let file):
            return "Failed to download test file: \(file)"
        case .listFailed(let category):
            return "Failed to list tests in category: \(category)"
        case .invalidResponse:
            return "Invalid response from GitHub API"
        case .testNotFound(let name):
            return "Test not found: \(name)"
        }
    }
}
