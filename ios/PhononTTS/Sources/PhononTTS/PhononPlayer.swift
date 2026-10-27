import AVFoundation

/// Plays Phonon's audio as it arrives, so sound starts with the first chunk rather than after
/// the last.
///
/// ```swift
/// let player = try PhononPlayer()
/// try await player.play(phonon.stream("Hello there."))
/// ```
///
/// On iOS it sets the shared audio session to `.playback` unless told not to; an app that
/// manages its own session should pass `configureSession: false`.
public final class PhononPlayer: @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format = AVAudioFormat(standardFormatWithSampleRate: Phonon.sampleRate, channels: 1)!

    public init(configureSession: Bool = true) throws {
        #if os(iOS)
        if configureSession {
            try AVAudioSession.sharedInstance().setCategory(.playback)
            try AVAudioSession.sharedInstance().setActive(true)
        }
        #endif
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        try engine.start()
        node.play()
    }

    deinit {
        node.stop()
        engine.stop()
    }

    /// Queue samples to play after whatever is already queued. Safe from any thread.
    public func enqueue(_ samples: [Float]) {
        guard !samples.isEmpty,
              let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))
        else { return }
        buf.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { buf.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        node.scheduleBuffer(buf)
    }

    /// Play a stream as it is generated. Returns when the stream ends; the last chunks may
    /// still be playing. Keep this player alive until playback finishes.
    /// Cancel the calling task to stop the stream, and call `stop()` to clear scheduled audio.
    public func play(_ stream: AsyncThrowingStream<[Float], Error>) async throws {
        for try await chunk in stream {
            enqueue(chunk)
        }
    }

    /// Drop everything queued and go quiet. Also cancel the playback task to stop generation;
    /// otherwise it can enqueue more audio. Playing again after this works as before.
    public func stop() {
        node.stop()
        node.play()
    }
}
