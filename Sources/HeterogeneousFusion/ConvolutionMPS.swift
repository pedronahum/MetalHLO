// ConvolutionMPS.swift
// HeterogeneousFusion
//
// Phase 5: Output-channel-sliceable 2D convolution via MPSCNNConvolution.
// This is the ANE's home territory — MPSCNNConvolution routes through Apple's
// Neural Engine for eligible shapes on Apple Silicon.
//
// NCHW buffer layout (matches ConvolutionGPU). The MPS path handles
// buffer ↔ MPSImage conversion internally using writeBytes/readBytes.

import Metal
import MetalPerformanceShaders
import QuartzCore

/// Data source for MPSCNNConvolution weights.
/// Holds a pointer to the weight data and provides it to MPS on demand.
private final class ConvWeightDataSource: NSObject, MPSCNNConvolutionDataSource {

    private let weightData: UnsafeMutableRawPointer
    private let weightByteCount: Int
    private let convDesc: MPSCNNConvolutionDescriptor
    private let ownedCopy: Bool

    init(weights: UnsafeRawPointer, byteCount: Int, descriptor: MPSCNNConvolutionDescriptor, copy: Bool = true) {
        if copy {
            self.weightData = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: 16)
            self.weightData.copyMemory(from: weights, byteCount: byteCount)
            self.ownedCopy = true
        } else {
            self.weightData = UnsafeMutableRawPointer(mutating: weights)
            self.ownedCopy = false
        }
        self.weightByteCount = byteCount
        self.convDesc = descriptor
        super.init()
    }

    deinit {
        if ownedCopy { weightData.deallocate() }
    }

    func dataType() -> MPSDataType { .float32 }
    func descriptor() -> MPSCNNConvolutionDescriptor { convDesc }
    func weights() -> UnsafeMutableRawPointer { weightData }
    func biasTerms() -> UnsafeMutablePointer<Float>? { nil }
    func load() -> Bool { true }
    func purge() {}
    func label() -> String? { "conv_weights" }
    func copy(with zone: NSZone? = nil) -> Any {
        ConvWeightDataSource(weights: weightData, byteCount: weightByteCount, descriptor: convDesc, copy: true)
    }
}

/// Performs 2D convolution via MPSCNNConvolution (routes to ANE on Apple Silicon).
/// Supports output-channel slicing for partitioned execution.
public final class ConvolutionMPS: @unchecked Sendable {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    public init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw HeterogeneousError.metalSetupFailed("Failed to create MPS convolution command queue")
        }
        self.commandQueue = queue
    }

    /// Expose command queue for shared use.
    public var queue: MTLCommandQueue { commandQueue }

    /// Execute a 2D convolution (or output-channel slice) synchronously.
    ///
    /// MPS CNN operates on MPSImage (texture-backed), so this method:
    /// 1. Copies NCHW input buffer → MPSImage
    /// 2. Runs MPSCNNConvolution (may route to ANE)
    /// 3. Copies output MPSImage → NCHW output buffer
    ///
    /// For output-channel slicing: creates a conv with sliceOutC output channels,
    /// using the weight slice starting at weightsOffset.
    ///
    /// - Returns: Wall-clock time in seconds (includes buffer↔texture overhead).
    public func execute(
        input: MTLBuffer, weights: MTLBuffer, output: MTLBuffer,
        weightsOffset: Int, outputOffset: Int,
        shape: ConvShape, sliceOutC: Int
    ) -> Double {
        let start = CACurrentMediaTime()

        // MPS CNN weight layout: [outC, kH, kW, inC] (OIHW→OHWI internally)
        // Our NCHW weights are [outC, inC, kH, kW].
        // MPSCNNConvolution expects OHWI layout, so we need to transpose.
        let filterCount = sliceOutC * shape.inChannels * shape.kernelH * shape.kernelW
        let transposedWeights = UnsafeMutablePointer<Float>.allocate(capacity: filterCount)
        defer { transposedWeights.deallocate() }

        let srcWeights = (weights.contents() + weightsOffset).assumingMemoryBound(to: Float.self)
        transposeWeightsNCHWtoOHWI(
            src: srcWeights, dst: transposedWeights,
            outC: sliceOutC, inC: shape.inChannels,
            kH: shape.kernelH, kW: shape.kernelW
        )

        // Create convolution descriptor
        let desc = MPSCNNConvolutionDescriptor(
            kernelWidth: shape.kernelW,
            kernelHeight: shape.kernelH,
            inputFeatureChannels: shape.inChannels,
            outputFeatureChannels: sliceOutC
        )
        desc.strideInPixelsX = shape.strideW
        desc.strideInPixelsY = shape.strideH

        // Create data source with transposed weights
        let weightByteCount = filterCount * MemoryLayout<Float>.size
        let dataSource = ConvWeightDataSource(
            weights: transposedWeights, byteCount: weightByteCount,
            descriptor: desc, copy: true
        )

        let conv = MPSCNNConvolution(device: device, weights: dataSource)
        conv.offset = MPSOffset(x: shape.kernelW / 2 - shape.padW, y: shape.kernelH / 2 - shape.padH, z: 0)
        conv.edgeMode = .zero

        // Create MPSImages
        let inputDesc = MPSImageDescriptor(
            channelFormat: .float32,
            width: shape.width, height: shape.height,
            featureChannels: shape.inChannels
        )
        inputDesc.numberOfImages = shape.batch

        let outputDesc = MPSImageDescriptor(
            channelFormat: .float32,
            width: shape.outWidth, height: shape.outHeight,
            featureChannels: sliceOutC
        )
        outputDesc.numberOfImages = shape.batch

        let inputImage = MPSImage(device: device, imageDescriptor: inputDesc)
        let outputImage = MPSImage(device: device, imageDescriptor: outputDesc)

        // Copy NCHW buffer → MPSImage
        writeNCHWToMPSImage(
            buffer: input, image: inputImage,
            batch: shape.batch, channels: shape.inChannels,
            height: shape.height, width: shape.width
        )

        // Encode and execute
        guard let cb = commandQueue.makeCommandBuffer() else { return 0 }
        conv.encode(commandBuffer: cb, sourceImage: inputImage, destinationImage: outputImage)
        cb.commit()
        cb.waitUntilCompleted()

        // Copy output MPSImage → NCHW buffer
        readMPSImageToNCHW(
            image: outputImage, buffer: output, bufferOffset: outputOffset,
            batch: shape.batch, channels: sliceOutC,
            height: shape.outHeight, width: shape.outWidth
        )

        return CACurrentMediaTime() - start
    }

    /// Encode convolution into a command buffer. Returns the output MPSImage
    /// which must be read back after the command buffer completes.
    ///
    /// This two-phase encode+readback pattern is used by FusedExecutor
    /// to overlap GPU/MPS/CPU work via commit-before-wait.
    public func encodeReturningImage(
        input: MTLBuffer, weights: MTLBuffer,
        weightsOffset: Int,
        shape: ConvShape, sliceOutC: Int,
        commandBuffer: MTLCommandBuffer
    ) -> MPSImage? {
        // Transpose weights
        let filterCount = sliceOutC * shape.inChannels * shape.kernelH * shape.kernelW
        let transposedWeights = UnsafeMutablePointer<Float>.allocate(capacity: filterCount)
        defer { transposedWeights.deallocate() }

        let srcWeights = (weights.contents() + weightsOffset).assumingMemoryBound(to: Float.self)
        transposeWeightsNCHWtoOHWI(
            src: srcWeights, dst: transposedWeights,
            outC: sliceOutC, inC: shape.inChannels,
            kH: shape.kernelH, kW: shape.kernelW
        )

        let desc = MPSCNNConvolutionDescriptor(
            kernelWidth: shape.kernelW,
            kernelHeight: shape.kernelH,
            inputFeatureChannels: shape.inChannels,
            outputFeatureChannels: sliceOutC
        )
        desc.strideInPixelsX = shape.strideW
        desc.strideInPixelsY = shape.strideH

        let weightByteCount = filterCount * MemoryLayout<Float>.size
        let dataSource = ConvWeightDataSource(
            weights: transposedWeights, byteCount: weightByteCount,
            descriptor: desc, copy: true
        )

        let conv = MPSCNNConvolution(device: device, weights: dataSource)
        conv.offset = MPSOffset(x: shape.kernelW / 2 - shape.padW, y: shape.kernelH / 2 - shape.padH, z: 0)
        conv.edgeMode = .zero

        let inputDesc = MPSImageDescriptor(
            channelFormat: .float32,
            width: shape.width, height: shape.height,
            featureChannels: shape.inChannels
        )
        inputDesc.numberOfImages = shape.batch

        let outputDesc = MPSImageDescriptor(
            channelFormat: .float32,
            width: shape.outWidth, height: shape.outHeight,
            featureChannels: sliceOutC
        )
        outputDesc.numberOfImages = shape.batch

        let inputImage = MPSImage(device: device, imageDescriptor: inputDesc)
        let outputImage = MPSImage(device: device, imageDescriptor: outputDesc)

        writeNCHWToMPSImage(
            buffer: input, image: inputImage,
            batch: shape.batch, channels: shape.inChannels,
            height: shape.height, width: shape.width
        )

        conv.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: outputImage)
        return outputImage
    }

    /// Read an MPSImage back to NCHW buffer. Call after command buffer completes.
    public func readBackImage(
        _ image: MPSImage, to buffer: MTLBuffer, bufferOffset: Int,
        batch: Int, channels: Int, height: Int, width: Int
    ) {
        readMPSImageToNCHW(
            image: image, buffer: buffer, bufferOffset: bufferOffset,
            batch: batch, channels: channels,
            height: height, width: width
        )
    }

    // MARK: - Weight Layout Conversion

    /// Transpose weights from NCHW [Cout, Cin, Kh, Kw] to OHWI [Cout, Kh, Kw, Cin]
    /// which is what MPSCNNConvolution expects.
    private func transposeWeightsNCHWtoOHWI(
        src: UnsafePointer<Float>, dst: UnsafeMutablePointer<Float>,
        outC: Int, inC: Int, kH: Int, kW: Int
    ) {
        for oc in 0..<outC {
            for kh in 0..<kH {
                for kw in 0..<kW {
                    for ic in 0..<inC {
                        // src: [oc][ic][kh][kw]
                        let srcIdx = oc * (inC * kH * kW) + ic * (kH * kW) + kh * kW + kw
                        // dst: [oc][kh][kw][ic]
                        let dstIdx = oc * (kH * kW * inC) + kh * (kW * inC) + kw * inC + ic
                        dst[dstIdx] = src[srcIdx]
                    }
                }
            }
        }
    }

    // MARK: - Buffer ↔ MPSImage Conversion

    /// Write NCHW buffer data to an MPSImage.
    /// MPSImage.writeBytes expects CHW layout per image index (batch element).
    private func writeNCHWToMPSImage(
        buffer: MTLBuffer, image: MPSImage,
        batch: Int, channels: Int, height: Int, width: Int
    ) {
        let spatial = height * width
        let batchStride = channels * spatial
        let bytesPerImage = batchStride * MemoryLayout<Float>.size
        let ptr = buffer.contents().assumingMemoryBound(to: Float.self)

        for b in 0..<batch {
            let batchPtr = ptr + b * batchStride
            image.writeBytes(
                batchPtr,
                dataLayout: .featureChannelsxHeightxWidth,
                bytesPerRow: width * MemoryLayout<Float>.size,
                bytesPerImage: bytesPerImage,
                region: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                  size: MTLSize(width: width, height: height, depth: 1)),
                featureChannelInfo: MPSImageReadWriteParams(
                    featureChannelOffset: 0,
                    numberOfFeatureChannelsToReadWrite: channels
                ),
                imageIndex: b
            )
        }
    }

    /// Read MPSImage data to NCHW buffer.
    private func readMPSImageToNCHW(
        image: MPSImage, buffer: MTLBuffer, bufferOffset: Int,
        batch: Int, channels: Int, height: Int, width: Int
    ) {
        let spatial = height * width
        let batchStride = channels * spatial
        let bytesPerImage = batchStride * MemoryLayout<Float>.size
        let ptr = (buffer.contents() + bufferOffset).assumingMemoryBound(to: Float.self)

        for b in 0..<batch {
            let batchPtr = UnsafeMutableRawPointer(ptr + b * batchStride)
            image.readBytes(
                batchPtr,
                dataLayout: .featureChannelsxHeightxWidth,
                bytesPerRow: width * MemoryLayout<Float>.size,
                bytesPerImage: bytesPerImage,
                region: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                  size: MTLSize(width: width, height: height, depth: 1)),
                featureChannelInfo: MPSImageReadWriteParams(
                    featureChannelOffset: 0,
                    numberOfFeatureChannelsToReadWrite: channels
                ),
                imageIndex: b
            )
        }
    }
}
