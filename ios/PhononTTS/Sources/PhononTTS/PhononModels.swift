import CryptoKit
import Foundation

/// Getting the models onto the device, into a writable directory `Phonon.load` can use.
///
/// A model bundle is a directory written by the `export_coreml` example in `ptts`: three Core ML
/// packages, the host tensors, the tokenizer, the voices and a `bundle.json` listing every
/// file with its size and SHA-256.
///
/// Two ways to deliver it:
/// - **Download it** on first run from any static file host with `download(from:)`. This keeps
///   the app small, which matters: the App Store will not download an app over 200 MB on a
///   cellular connection without asking.
/// - **Ship it in the app** as a folder reference and copy it out with `install(bundled:)`.
///   Core ML compiles a model next to itself, so it cannot be used from the read-only app
///   bundle directly.
///
/// Either way the installed copy is excluded from iCloud backup, and replaced whole when the
/// source's `bundle.json` changes (the exporter stamps each export).
public enum PhononModels {
    struct Manifest: Decodable {
        struct File: Decodable {
            let path: String
            let size: Int
            let sha256: String
        }
        let built: Int
        let files: [File]
    }

    /// `Application Support/PhononTTS/models`.
    public static var defaultDirectory: URL {
        URL.applicationSupportDirectory.appending(path: "PhononTTS/models", directoryHint: .isDirectory)
    }

    static func manifest(at dir: URL) -> Manifest? {
        (try? Data(contentsOf: dir.appending(path: "bundle.json")))
            .flatMap { try? JSONDecoder().decode(Manifest.self, from: $0) }
    }

    /// Whether `dir` holds a complete bundle.
    public static func isInstalled(at dir: URL = defaultDirectory) -> Bool {
        guard let m = manifest(at: dir) else { return false }
        return m.files.allSatisfy { FileManager.default.fileExists(atPath: dir.appending(path: $0.path).path) }
    }

    /// Copy a bundle shipped inside the app (for example `Bundle.main.url(forResource: "Models",
    /// withExtension: nil)`) to `dir`, unless the same export is already there.
    @discardableResult
    public static func install(bundled source: URL, to dir: URL = defaultDirectory) throws -> URL {
        let fm = FileManager.default
        guard let want = manifest(at: source) else {
            throw PhononError(description: "\(source.path) has no readable bundle.json")
        }
        if isInstalled(at: dir), manifest(at: dir)?.built == want.built { return dir }
        try? fm.removeItem(at: dir)
        try fm.createDirectory(at: dir.deletingLastPathComponent(), withIntermediateDirectories: true)
        try fm.copyItem(at: source, to: dir)
        try excludeFromBackup(dir)
        return dir
    }

    /// Download a bundle from `base`, which serves `bundle.json` and every file it lists at the
    /// same relative paths, into `dir`. Does nothing if that export is already installed.
    ///
    /// Each file is checked against its size and SHA-256 before it is kept. An interrupted
    /// download resumes: files already fetched and verified are not fetched again. The bundle
    /// only replaces an installed one once every file is in, so a failure leaves the old one
    /// usable. `progress` receives the fraction of bytes done, on an arbitrary thread.
    @discardableResult
    public static func download(
        from base: URL,
        to dir: URL = defaultDirectory,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> URL {
        let fm = FileManager.default
        let (data, response) = try await URLSession.shared.data(from: base.appending(path: "bundle.json"))
        try check(response, "bundle.json")
        let want = try JSONDecoder().decode(Manifest.self, from: data)
        if isInstalled(at: dir), manifest(at: dir)?.built == want.built {
            progress?(1)
            return dir
        }
        // Staged beside the destination, so the final move is a rename on one volume.
        let staging = dir.deletingLastPathComponent().appending(path: "\(dir.lastPathComponent).partial-\(want.built)")
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        let total = Double(want.files.reduce(0) { $0 + $1.size })
        var done = 0.0
        for file in want.files {
            let dest = staging.appending(path: file.path)
            if !(try matches(dest, file)) {
                let before = done
                let (tmp, response) = try await URLSession.shared.download(
                    from: base.appending(path: file.path),
                    delegate: progress.map { p in Progress { p((before + Double($0)) / total) } }
                )
                try check(response, file.path)
                guard try matches(tmp, file) else {
                    try? fm.removeItem(at: tmp)
                    throw PhononError(description: "\(file.path) does not match its size or SHA-256")
                }
                try fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? fm.removeItem(at: dest)
                try fm.moveItem(at: tmp, to: dest)
            }
            done += Double(file.size)
            progress?(done / total)
        }
        try data.write(to: staging.appending(path: "bundle.json"))
        try? fm.removeItem(at: dir)
        try fm.moveItem(at: staging, to: dir)
        try excludeFromBackup(dir)
        return dir
    }

    private static func check(_ response: URLResponse, _ what: String) throws {
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw PhononError(description: "\(what): HTTP \(http.statusCode)")
        }
    }

    private static func matches(_ url: URL, _ file: Manifest.File) throws -> Bool {
        guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize, size == file.size
        else { return false }
        let h = try FileHandle(forReadingFrom: url)
        defer { try? h.close() }
        var hasher = SHA256()
        while let chunk = try h.read(upToCount: 8 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined() == file.sha256
    }

    private static func excludeFromBackup(_ url: URL) throws {
        var url = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try url.setResourceValues(values)
    }

    /// Byte progress of one download.
    private final class Progress: NSObject, URLSessionDownloadDelegate, Sendable {
        let report: @Sendable (Int64) -> Void
        init(_ report: @escaping @Sendable (Int64) -> Void) { self.report = report }

        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                        didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                        totalBytesExpectedToWrite: Int64) {
            report(totalBytesWritten)
        }

        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                        didFinishDownloadingTo location: URL) {}
    }
}
