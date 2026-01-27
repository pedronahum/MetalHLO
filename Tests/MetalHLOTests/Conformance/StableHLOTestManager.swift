// StableHLOTestManager.swift
// MetalHLOTests
//
// Manages downloading and caching of StableHLO test files from the
// openxla/stablehlo repository.

import Foundation

/// Manages StableHLO test files - downloading, caching, and organizing them
public final class StableHLOTestManager: @unchecked Sendable {

    /// Base URL for raw StableHLO test files
    private static let baseURL = "https://raw.githubusercontent.com/openxla/stablehlo/main/stablehlo/testdata/"

    /// GitHub API URL for listing test files
    private static let apiURL = "https://api.github.com/repos/openxla/stablehlo/contents/stablehlo/testdata"

    /// Local cache directory for downloaded test files
    private let cacheDirectory: URL

    /// File storing the list of all available tests
    private var testListCacheFile: URL {
        cacheDirectory.appendingPathComponent(".test-list.json")
    }

    /// Shared instance using default cache location
    public static let shared = StableHLOTestManager()

    /// Initialize with custom cache directory
    public init(cacheDirectory: URL? = nil) {
        if let dir = cacheDirectory {
            self.cacheDirectory = dir
        } else {
            // Default to ~/.cache/metalhlo/stablehlo-testdata/
            let home = FileManager.default.homeDirectoryForCurrentUser
            self.cacheDirectory = home
                .appendingPathComponent(".cache")
                .appendingPathComponent("metalhlo")
                .appendingPathComponent("stablehlo-testdata")
        }
    }

    /// Ensure cache directory exists
    private func ensureCacheDirectory() throws {
        try FileManager.default.createDirectory(
            at: cacheDirectory,
            withIntermediateDirectories: true
        )
    }

    /// Get the local path for a test file
    public func localPath(for testName: String) -> URL {
        let filename = testName.hasSuffix(".mlir") ? testName : "\(testName).mlir"
        return cacheDirectory.appendingPathComponent(filename)
    }

    /// Check if a test file is cached locally
    public func isCached(_ testName: String) -> Bool {
        FileManager.default.fileExists(atPath: localPath(for: testName).path)
    }

    /// Download a test file if not cached
    public func ensureTestFile(_ testName: String) async throws -> URL {
        let localURL = localPath(for: testName)

        if isCached(testName) {
            return localURL
        }

        try ensureCacheDirectory()

        let filename = testName.hasSuffix(".mlir") ? testName : "\(testName).mlir"
        let remoteURL = URL(string: Self.baseURL + filename)!

        let (data, response) = try await URLSession.shared.data(from: remoteURL)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw StableHLOTestError.downloadFailed(testName)
        }

        try data.write(to: localURL)
        return localURL
    }

    /// Download multiple test files concurrently
    public func ensureTestFiles(_ testNames: [String]) async throws -> [URL] {
        try await withThrowingTaskGroup(of: URL.self) { group in
            for name in testNames {
                group.addTask {
                    try await self.ensureTestFile(name)
                }
            }

            var urls: [URL] = []
            for try await url in group {
                urls.append(url)
            }
            return urls
        }
    }

    /// List all cached test files
    public func listCachedTests() -> [String] {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: cacheDirectory,
            includingPropertiesForKeys: nil
        ) else {
            return []
        }

        return files
            .filter { $0.pathExtension == "mlir" }
            .map { $0.deletingPathExtension().lastPathComponent }
            .sorted()
    }

    /// Clear the test cache
    public func clearCache() throws {
        if FileManager.default.fileExists(atPath: cacheDirectory.path) {
            try FileManager.default.removeItem(at: cacheDirectory)
        }
    }

    /// Get cache size in bytes
    public func cacheSize() -> Int64 {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: cacheDirectory,
            includingPropertiesForKeys: [.fileSizeKey]
        ) else {
            return 0
        }

        return files.reduce(0) { total, url in
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            return total + Int64(size)
        }
    }

    // MARK: - Bulk Test Discovery

    /// Fetch the list of all available test files from GitHub API
    public func fetchAllAvailableTests(forceRefresh: Bool = false) async throws -> [String] {
        // Check cached list first
        if !forceRefresh, let cached = try? loadCachedTestList() {
            return cached
        }

        try ensureCacheDirectory()

        var allTests: [String] = []
        var page = 1
        let perPage = 100

        while true {
            let url = URL(string: "\(Self.apiURL)?per_page=\(perPage)&page=\(page)")!
            var request = URLRequest(url: url)
            request.setValue("application/vnd.github.v3+json", forHTTPHeaderField: "Accept")
            request.setValue("MetalHLO-ConformanceTests", forHTTPHeaderField: "User-Agent")

            let (data, response) = try await URLSession.shared.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                throw StableHLOTestError.downloadFailed("Invalid response")
            }

            if httpResponse.statusCode == 403 {
                // Rate limited - use cached list or partial result
                print("GitHub API rate limited. Using partial/cached results.")
                break
            }

            guard httpResponse.statusCode == 200 else {
                throw StableHLOTestError.downloadFailed("HTTP \(httpResponse.statusCode)")
            }

            // Parse JSON response
            guard let items = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
                break
            }

            if items.isEmpty {
                break
            }

            let testNames = items.compactMap { item -> String? in
                guard let name = item["name"] as? String,
                      name.hasSuffix(".mlir") else {
                    return nil
                }
                return String(name.dropLast(5)) // Remove .mlir extension
            }

            allTests.append(contentsOf: testNames)
            page += 1

            // GitHub API returns max 1000 items even with pagination
            // and the testdata directory has ~2000+ files
            // We need to use the Tree API instead for large directories
            if items.count < perPage {
                break
            }
        }

        // If we got rate limited or hit pagination limits, try Tree API
        if allTests.count < 500 {
            allTests = try await fetchAllTestsViaTreeAPI()
        }

        // Cache the list
        try? saveCachedTestList(allTests)

        return allTests.sorted()
    }

    /// Fetch all tests using GitHub Tree API (handles large directories better)
    private func fetchAllTestsViaTreeAPI() async throws -> [String] {
        // First get the default branch SHA
        let repoURL = URL(string: "https://api.github.com/repos/openxla/stablehlo")!
        var repoRequest = URLRequest(url: repoURL)
        repoRequest.setValue("application/vnd.github.v3+json", forHTTPHeaderField: "Accept")
        repoRequest.setValue("MetalHLO-ConformanceTests", forHTTPHeaderField: "User-Agent")

        let (repoData, _) = try await URLSession.shared.data(for: repoRequest)
        guard let repoInfo = try? JSONSerialization.jsonObject(with: repoData) as? [String: Any],
              let defaultBranch = repoInfo["default_branch"] as? String else {
            return []
        }

        // Get the tree for the testdata directory
        let treeURL = URL(string: "https://api.github.com/repos/openxla/stablehlo/git/trees/\(defaultBranch)?recursive=1")!
        var treeRequest = URLRequest(url: treeURL)
        treeRequest.setValue("application/vnd.github.v3+json", forHTTPHeaderField: "Accept")
        treeRequest.setValue("MetalHLO-ConformanceTests", forHTTPHeaderField: "User-Agent")

        let (treeData, _) = try await URLSession.shared.data(for: treeRequest)
        guard let treeInfo = try? JSONSerialization.jsonObject(with: treeData) as? [String: Any],
              let tree = treeInfo["tree"] as? [[String: Any]] else {
            return []
        }

        // Filter for testdata/*.mlir files
        let testNames = tree.compactMap { item -> String? in
            guard let path = item["path"] as? String,
                  path.hasPrefix("stablehlo/testdata/"),
                  path.hasSuffix(".mlir") else {
                return nil
            }
            let filename = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
            return filename
        }

        return testNames
    }

    /// Load cached test list
    private func loadCachedTestList() throws -> [String]? {
        guard FileManager.default.fileExists(atPath: testListCacheFile.path) else {
            return nil
        }

        // Check if cache is older than 24 hours
        let attrs = try FileManager.default.attributesOfItem(atPath: testListCacheFile.path)
        if let modDate = attrs[.modificationDate] as? Date,
           Date().timeIntervalSince(modDate) > 86400 {
            return nil
        }

        let data = try Data(contentsOf: testListCacheFile)
        return try JSONDecoder().decode([String].self, from: data)
    }

    /// Save test list to cache
    private func saveCachedTestList(_ tests: [String]) throws {
        let data = try JSONEncoder().encode(tests)
        try data.write(to: testListCacheFile)
    }

    /// Download all available test files
    /// Returns tuple of (downloaded, failed, skipped)
    public func downloadAllTests(
        concurrency: Int = 10,
        progressHandler: ((Int, Int) -> Void)? = nil
    ) async throws -> (downloaded: Int, failed: Int, skipped: Int) {
        let allTests = try await fetchAllAvailableTests()
        let total = allTests.count

        var downloaded = 0
        var failed = 0
        var skipped = 0

        // Process in batches to control concurrency
        for batchStart in stride(from: 0, to: total, by: concurrency) {
            let batchEnd = min(batchStart + concurrency, total)
            let batch = Array(allTests[batchStart..<batchEnd])

            await withTaskGroup(of: (String, Bool, Bool).self) { group in
                for testName in batch {
                    group.addTask {
                        if self.isCached(testName) {
                            return (testName, true, true) // skipped
                        }
                        do {
                            _ = try await self.ensureTestFile(testName)
                            return (testName, true, false) // downloaded
                        } catch {
                            return (testName, false, false) // failed
                        }
                    }
                }

                for await (_, success, wasSkipped) in group {
                    if wasSkipped {
                        skipped += 1
                    } else if success {
                        downloaded += 1
                    } else {
                        failed += 1
                    }
                }
            }

            progressHandler?(batchEnd, total)
        }

        return (downloaded, failed, skipped)
    }

    /// Get statistics about available vs cached tests
    public func getCacheStatistics() async throws -> CacheStatistics {
        let allTests = try await fetchAllAvailableTests()
        let cachedTests = listCachedTests()
        let cachedSet = Set(cachedTests)

        let cachedCount = allTests.filter { cachedSet.contains($0) }.count

        return CacheStatistics(
            totalAvailable: allTests.count,
            cached: cachedCount,
            cacheSize: cacheSize()
        )
    }
}

/// Statistics about the test cache
public struct CacheStatistics: Sendable {
    public let totalAvailable: Int
    public let cached: Int
    public let cacheSize: Int64

    public var coveragePercent: Double {
        totalAvailable > 0 ? Double(cached) / Double(totalAvailable) * 100 : 0
    }

    public var cacheSizeMB: Double {
        Double(cacheSize) / (1024 * 1024)
    }

    public var summary: String {
        """
        StableHLO Test Cache Statistics
        ===============================
        Available tests: \(totalAvailable)
        Cached tests:    \(cached) (\(String(format: "%.1f", coveragePercent))%)
        Cache size:      \(String(format: "%.1f", cacheSizeMB)) MB
        """
    }
}

/// Predefined test suites organized by operation category
public enum StableHLOTestSuite {

    /// Core arithmetic operations
    public static let arithmetic: [String] = [
        "add_float32_20_20_float32_20_20",
        "mul_float32_20_20_float32_20_20",
        "div_float32_2_float32_2",  // StableHLO uses smaller shapes for div
        "max_float32_20_20_float32_20_20",
        "min_float32_20_20_float32_20_20",
        "pow_float32_20_30_float32_20_30",  // StableHLO uses 20x30 for pow
        "atan2_float32_20_20_float32_20_20",
    ]

    /// Unary math operations
    public static let unaryMath: [String] = [
        "abs_float32_20_20",
        "neg_float32_20_20",
        "exp_float32_20_20",
        "log_float32_20_20",
        "log1p_float32_20_20",
        "expm1_float32_20_20",
        "sqrt_float32_20_20",
        "rsqrt_float32_20_20",
        "cbrt_float32_20_20",
        "ceil_float32_20_20",
        "floor_float32_20_20",
        "sin_float32_20_20",
        "cos_float32_20_20",
        "tanh_float32_20_20",
    ]

    /// Trigonometric operations
    public static let trigonometric: [String] = [
        "sin_float32_20_20",
        "cos_float32_20_20",
        "tan_float32_20_20",
        "asin_float32_20_20",
        "acos_float32_20_20",
        "atan_float32_20_20",
        "sinh_float32_20_20",
        "cosh_float32_20_20",
        "tanh_float32_20_20",
    ]

    /// Comparison operations
    public static let comparison: [String] = [
        "eq_float32_20_20_float32_20_20",
        "ne_float32_20_20_float32_20_20",
        "lt_float32_20_20_float32_20_20",
        "le_float32_20_20_float32_20_20",
        "gt_float32_20_20_float32_20_20",
        "ge_float32_20_20_float32_20_20",
    ]

    /// Reduction operations (using actual StableHLO testdata file names)
    public static let reduction: [String] = [
        "reduce_sum_float32_2_3",
        "reduce_max_float32_2_3",
        "reduce_min_float32_2_3",
        "reduce_prod_float32_2_3",
        // argmax/argmin not in standard StableHLO testdata
    ]

    /// Matrix operations
    public static let matrix: [String] = [
        "dot_general_float32_4_3_float32_3_2",
        "transpose_float32_20_20",
    ]

    /// Core operations for LLM inference (high priority)
    public static let llmCore: [String] = [
        // Element-wise (using actual StableHLO testdata file names)
        "add_float32_20_20_float32_20_20",
        "mul_float32_20_20_float32_20_20",
        "div_float32_2_float32_2",  // actual file name

        // Activations
        "exp_float32_20_20",
        "tanh_float32_20_20",
        "rsqrt_float32_20_20",

        // Reductions (using actual StableHLO testdata file names)
        "reduce_sum_float32_2_3",  // actual file name
        "reduce_max_float32_2_3",  // actual file name
    ]

    /// Float16 variants (important for inference)
    public static let float16Core: [String] = [
        "add_float16_20_20_float16_20_20",
        "mul_float16_20_20_float16_20_20",
        "exp_float16_20_20",
        "tanh_float16_20_20",
        "abs_float16_20_20",
    ]

    /// BFloat16 variants
    public static let bfloat16Core: [String] = [
        "add_bfloat16_20_20_bfloat16_20_20",
        "mul_bfloat16_20_20_bfloat16_20_20",
        "exp_bfloat16_20_20",
        "tanh_bfloat16_20_20",
        "abs_bfloat16_20_20",
    ]

    /// All priority tests for initial conformance
    public static var priorityTests: [String] {
        return llmCore + unaryMath
    }

    /// Comprehensive test suite
    public static var comprehensive: [String] {
        return arithmetic + unaryMath + trigonometric + comparison + reduction
    }
}

/// Errors related to StableHLO test management
public enum StableHLOTestError: Error, CustomStringConvertible {
    case downloadFailed(String)
    case testNotFound(String)
    case cacheError(String)

    public var description: String {
        switch self {
        case .downloadFailed(let name):
            return "Failed to download test file: \(name)"
        case .testNotFound(let name):
            return "Test file not found: \(name)"
        case .cacheError(let reason):
            return "Cache error: \(reason)"
        }
    }
}
