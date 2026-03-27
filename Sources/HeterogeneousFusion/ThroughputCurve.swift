// ThroughputCurve.swift
// HeterogeneousFusion
//
// Phase 3: Models a compute unit's execution time as a function of slice size.
// time(rows) = a + b * rows + c * rows²
//   a = fixed overhead (dispatch, encode, sync)
//   b = linear term (per-row compute)
//   c = quadratic term (memory pressure, cache effects)
//
// Fitted via least-squares from Phase 1 measurements.

import Foundation

/// Models execution time vs. slice size for one (unit, op) pair.
///
/// Uses piecewise linear interpolation between calibration measurements
/// for accuracy within the measured range, with quadratic extrapolation outside.
///
/// `estimate(rows)` returns predicted wall-clock time in milliseconds.
public struct ThroughputCurve: Sendable, Codable {
    /// Fixed overhead in ms (dispatch, encode, fence).
    public let a: Double
    /// Linear coefficient: ms per row.
    public let b: Double
    /// Quadratic coefficient: ms per row².
    public let c: Double
    /// Which unit this curve models.
    public let unit: ComputeUnit
    /// Which op this curve models.
    public let op: HLOOpCode
    /// Raw calibration data points, sorted by rows (for piecewise interpolation).
    public let dataPoints: [(rows: Int, timeMs: Double)]

    enum CodingKeys: String, CodingKey {
        case a, b, c, unit, op, dpRows, dpTimes
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        a = try container.decode(Double.self, forKey: .a)
        b = try container.decode(Double.self, forKey: .b)
        c = try container.decode(Double.self, forKey: .c)
        unit = try container.decode(ComputeUnit.self, forKey: .unit)
        op = try container.decode(HLOOpCode.self, forKey: .op)
        let rows = try container.decodeIfPresent([Int].self, forKey: .dpRows) ?? []
        let times = try container.decodeIfPresent([Double].self, forKey: .dpTimes) ?? []
        dataPoints = zip(rows, times).map { (rows: $0.0, timeMs: $0.1) }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(a, forKey: .a)
        try container.encode(b, forKey: .b)
        try container.encode(c, forKey: .c)
        try container.encode(unit, forKey: .unit)
        try container.encode(op, forKey: .op)
        try container.encode(dataPoints.map(\.rows), forKey: .dpRows)
        try container.encode(dataPoints.map(\.timeMs), forKey: .dpTimes)
    }

    public init(a: Double, b: Double, c: Double, unit: ComputeUnit, op: HLOOpCode,
                dataPoints: [(rows: Int, timeMs: Double)] = []) {
        self.a = a
        self.b = b
        self.c = c
        self.unit = unit
        self.op = op
        self.dataPoints = dataPoints.sorted { $0.rows < $1.rows }
    }

    /// Predict execution time in ms for a given number of rows.
    ///
    /// Uses piecewise linear interpolation when data points are available
    /// and the query falls within the measured range. Falls back to
    /// quadratic extrapolation outside the range.
    public func estimate(rows: Int) -> Double {
        if let interpolated = interpolate(rows: rows) {
            return interpolated
        }
        let r = Double(rows)
        return max(0, a + b * r + c * r * r)
    }

    /// Predict execution time for a scaled shape.
    /// For matmul [M, K] @ [K, N], rows = M (split dimension).
    public func estimate(rows: Int, K: Int, N: Int) -> Double {
        // The curve is fitted against total rows; K and N are implicit
        // in the coefficients (fitted per-shape-class or per-op).
        estimate(rows: rows)
    }

    /// Throughput in rows/ms at a given slice size.
    public func throughput(rows: Int) -> Double {
        let t = estimate(rows: rows)
        guard t > 0 else { return 0 }
        return Double(rows) / t
    }

    /// Piecewise linear interpolation between data points.
    /// Returns nil if no data points or query is outside the range.
    private func interpolate(rows: Int) -> Double? {
        guard dataPoints.count >= 2 else { return nil }
        let r = Double(rows)

        // Outside measured range: fall back to quadratic
        guard r >= Double(dataPoints.first!.rows),
              r <= Double(dataPoints.last!.rows) else {
            return nil
        }

        // Find bracketing interval
        for i in 0..<(dataPoints.count - 1) {
            let r0 = Double(dataPoints[i].rows)
            let r1 = Double(dataPoints[i + 1].rows)
            if r >= r0 && r <= r1 {
                let t0 = dataPoints[i].timeMs
                let t1 = dataPoints[i + 1].timeMs
                let frac = (r1 > r0) ? (r - r0) / (r1 - r0) : 0
                return max(0, t0 + frac * (t1 - t0))
            }
        }
        return nil
    }

    // MARK: - Fitting

    /// Fit a ThroughputCurve from (rows, timeMs) measurement pairs.
    ///
    /// Uses ordinary least-squares to find a, b, c that minimize
    /// Σ(time_i - a - b*rows_i - c*rows_i²)².
    ///
    /// Requires at least 3 data points for a quadratic fit.
    /// Falls back to linear (c=0) with 2 points, constant (b=c=0) with 1.
    public static func fit(
        measurements: [(rows: Int, timeMs: Double)],
        unit: ComputeUnit,
        op: HLOOpCode
    ) -> ThroughputCurve {
        let n = measurements.count

        let sorted = measurements.sorted { $0.rows < $1.rows }

        guard n >= 1 else {
            return ThroughputCurve(a: 0, b: 0, c: 0, unit: unit, op: op)
        }

        if n == 1 {
            let (rows, time) = measurements[0]
            return ThroughputCurve(
                a: 0,
                b: rows > 0 ? time / Double(rows) : time,
                c: 0,
                unit: unit, op: op, dataPoints: sorted
            )
        }

        if n == 2 {
            let r0 = Double(measurements[0].rows)
            let r1 = Double(measurements[1].rows)
            let t0 = measurements[0].timeMs
            let t1 = measurements[1].timeMs

            guard r0 != r1 else {
                return ThroughputCurve(a: t0, b: 0, c: 0, unit: unit, op: op, dataPoints: sorted)
            }

            let b = (t1 - t0) / (r1 - r0)
            let a = t0 - b * r0
            return ThroughputCurve(a: a, b: b, c: 0, unit: unit, op: op, dataPoints: sorted)
        }

        // 3+ points: full quadratic least-squares
        // Normal equations for y = a + b*x + c*x²
        // [n,     Σx,    Σx²  ] [a]   [Σy    ]
        // [Σx,    Σx²,   Σx³  ] [b] = [Σxy   ]
        // [Σx²,   Σx³,   Σx⁴  ] [c]   [Σx²y  ]
        var sx: Double = 0, sx2: Double = 0, sx3: Double = 0, sx4: Double = 0
        var sy: Double = 0, sxy: Double = 0, sx2y: Double = 0

        for (rows, time) in measurements {
            let x = Double(rows)
            let x2 = x * x
            sx += x
            sx2 += x2
            sx3 += x * x2
            sx4 += x2 * x2
            sy += time
            sxy += x * time
            sx2y += x2 * time
        }

        let nd = Double(n)
        // Solve 3x3 system using Cramer's rule
        let m = [
            [nd, sx, sx2],
            [sx, sx2, sx3],
            [sx2, sx3, sx4],
        ]
        let rhs = [sy, sxy, sx2y]

        let det = det3(m)
        guard abs(det) > 1e-15 else {
            // Degenerate: fall back to linear
            let b = n > 1 ? sxy / sx2 : 0
            return ThroughputCurve(a: 0, b: b, c: 0, unit: unit, op: op, dataPoints: sorted)
        }

        let detA = det3([
            [rhs[0], m[0][1], m[0][2]],
            [rhs[1], m[1][1], m[1][2]],
            [rhs[2], m[2][1], m[2][2]],
        ])
        let detB = det3([
            [m[0][0], rhs[0], m[0][2]],
            [m[1][0], rhs[1], m[1][2]],
            [m[2][0], rhs[2], m[2][2]],
        ])
        let detC = det3([
            [m[0][0], m[0][1], rhs[0]],
            [m[1][0], m[1][1], rhs[1]],
            [m[2][0], m[2][1], rhs[2]],
        ])

        return ThroughputCurve(
            a: detA / det,
            b: detB / det,
            c: detC / det,
            unit: unit, op: op,
            dataPoints: sorted
        )
    }

    /// 3×3 determinant.
    private static func det3(_ m: [[Double]]) -> Double {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
}
