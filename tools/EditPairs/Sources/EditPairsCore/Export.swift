import Foundation
import ImageIO
import UniformTypeIdentifiers

/// One edit pair: 1-4 source (control) images, the target, and an optional caption of its own.
public struct Pair: Identifiable, Equatable {
    public let id = UUID()
    public var sources: [URL]
    public var target: URL?
    public var caption: String

    public init(sources: [URL] = [], target: URL? = nil, caption: String = "") {
        self.sources = sources
        self.target = target
        self.caption = caption
    }

    public static let maxSources = 4

    public var isComplete: Bool { !sources.isEmpty && target != nil }

    /// The pair's own caption, or the universal one when it has none.
    public func instruction(universal: String) -> String {
        let own = caption.trimmingCharacters(in: .whitespacesAndNewlines)
        return own.isEmpty ? universal.trimmingCharacters(in: .whitespacesAndNewlines) : own
    }
}

public enum ExportError: LocalizedError {
    case nothingToExport
    case missingCaptions([Int])
    case unreadableImage(URL)
    case zipFailed(String)

    public var errorDescription: String? {
        switch self {
        case .nothingToExport:
            return "No complete pairs yet: each pair needs a source and a target."
        case .missingCaptions(let rows):
            let list = rows.prefix(8).map(String.init).joined(separator: ", ")
            return "Pair \(list)\(rows.count > 8 ? "…" : "") has no caption. Type one, or set a universal caption."
        case .unreadableImage(let url):
            return "Couldn't read \(url.lastPathComponent)."
        case .zipFailed(let message):
            return "Couldn't make the zip: \(message)"
        }
    }
}

public enum Exporter {
    /// File types the trainer reads as they are; anything else (HEIC, TIFF, GIF…) is converted to PNG.
    public static let trainerExtensions: Set<String> = ["jpg", "jpeg", "png", "webp", "bmp"]
    /// What can be dropped in.
    public static let imageExtensions: Set<String> = trainerExtensions.union(["heic", "heif", "tif", "tiff", "gif"])

    public static func isImage(_ url: URL) -> Bool {
        imageExtensions.contains(url.pathExtension.lowercased())
    }

    /// Image files among dropped items (folders are searched all the way down), sorted by name the way Finder does.
    public static func images(in urls: [URL]) -> [URL] {
        var found: [URL] = []
        let fm = FileManager.default
        for url in urls {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: url.path, isDirectory: &isDir) else { continue }
            if isDir.boolValue {
                let items = fm.enumerator(at: url, includingPropertiesForKeys: nil,
                                          options: [.skipsHiddenFiles, .skipsPackageDescendants])
                while let item = items?.nextObject() as? URL {
                    if isImage(item) { found.append(item) }
                }
            } else if isImage(url) {
                found.append(url)
            }
        }
        return found.sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
    }

    /// Lays out the dataset the trainer's Edit pairs upload expects:
    ///   edits.jsonl, src/0001.jpg (src/0001_2.jpg for a second source…), tgt/0001.png + tgt/0001.txt
    /// Incomplete pairs are skipped. Returns how many pairs were written.
    @discardableResult
    public static func writeFolder(pairs: [Pair], universal: String, to dir: URL) throws -> Int {
        let ready = pairs.filter(\.isComplete)
        guard !ready.isEmpty else { throw ExportError.nothingToExport }
        let missing = pairs.enumerated()
            .filter { $0.element.isComplete && $0.element.instruction(universal: universal).isEmpty }
            .map { $0.offset + 1 }
        guard missing.isEmpty else { throw ExportError.missingCaptions(missing) }

        let fm = FileManager.default
        let src = dir.appendingPathComponent("src"), tgt = dir.appendingPathComponent("tgt")
        try fm.createDirectory(at: src, withIntermediateDirectories: true)
        try fm.createDirectory(at: tgt, withIntermediateDirectories: true)
        let digits = max(4, String(ready.count).count)
        var lines: [String] = []
        for (n, pair) in ready.enumerated() {
            let stem = String(format: "%0\(digits)d", n + 1)
            var sourcePaths: [String] = []
            for (i, url) in pair.sources.prefix(Pair.maxSources).enumerated() {
                let name = try copyImage(url, to: src, stem: i == 0 ? stem : "\(stem)_\(i + 1)")
                sourcePaths.append("src/\(name)")
            }
            let target = try copyImage(pair.target!, to: tgt, stem: stem)
            let instruction = pair.instruction(universal: universal)
            try instruction.write(to: tgt.appendingPathComponent("\(stem).txt"), atomically: true, encoding: .utf8)
            let row: [String: Any] = ["sources": sourcePaths, "target": "tgt/\(target)", "instruction": instruction]
            let data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys, .withoutEscapingSlashes])
            lines.append(String(decoding: data, as: UTF8.self))
        }
        try (lines.joined(separator: "\n") + "\n")
            .write(to: dir.appendingPathComponent("edits.jsonl"), atomically: true, encoding: .utf8)
        return ready.count
    }

    /// The same layout as `writeFolder`, zipped with edits.jsonl at the top: drop it on an Edit pairs dataset.
    @discardableResult
    public static func writeZip(pairs: [Pair], universal: String, to zip: URL) throws -> Int {
        let fm = FileManager.default
        let staging = fm.temporaryDirectory.appendingPathComponent("editpairs-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: staging) }
        let count = try writeFolder(pairs: pairs, universal: universal, to: staging)
        if fm.fileExists(atPath: zip.path) { try fm.removeItem(at: zip) }
        let ditto = Process()
        ditto.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        ditto.arguments = ["-c", "-k", "--norsrc", "--noextattr", staging.path, zip.path]
        let err = Pipe()
        ditto.standardError = err
        try ditto.run()
        ditto.waitUntilExit()
        guard ditto.terminationStatus == 0 else {
            let message = String(decoding: err.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            throw ExportError.zipFailed(message.isEmpty ? "ditto exited with \(ditto.terminationStatus)" : message)
        }
        return count
    }

    /// Copies an image the trainer can read as is, or converts it to PNG. Returns the new file name.
    static func copyImage(_ url: URL, to dir: URL, stem: String) throws -> String {
        let ext = url.pathExtension.lowercased()
        if trainerExtensions.contains(ext) {
            let name = "\(stem).\(ext)"
            let dest = dir.appendingPathComponent(name)
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.copyItem(at: url, to: dest)
            return name
        }
        let name = "\(stem).png"
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
              let dest = CGImageDestinationCreateWithURL(dir.appendingPathComponent(name) as CFURL,
                                                         UTType.png.identifier as CFString, 1, nil)
        else { throw ExportError.unreadableImage(url) }
        // keep the photo's orientation
        let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        var options: [CFString: Any] = [:]
        if let orientation = props?[kCGImagePropertyOrientation] { options[kCGImagePropertyOrientation] = orientation }
        CGImageDestinationAddImage(dest, image, options as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { throw ExportError.unreadableImage(url) }
        return name
    }

    /// Puts dropped images into the pairs: an image whose name matches a pair's other side goes there
    /// (cat.jpg in sources pairs with cat.png in targets), the rest fill the first open slots in order,
    /// then new pairs are added.
    public static func fill(_ pairs: inout [Pair], with urls: [URL], asTargets: Bool) {
        for url in urls {
            let stem = url.deletingPathExtension().lastPathComponent.lowercased()
            func otherStem(_ p: Pair) -> String? {
                (asTargets ? p.sources.first : p.target)?.deletingPathExtension().lastPathComponent.lowercased()
            }
            func isOpen(_ p: Pair) -> Bool { asTargets ? p.target == nil : p.sources.isEmpty }
            let index = pairs.firstIndex { isOpen($0) && otherStem($0) == stem } ?? pairs.firstIndex(where: isOpen)
            if let i = index {
                if asTargets { pairs[i].target = url } else { pairs[i].sources = [url] }
            } else {
                pairs.append(asTargets ? Pair(target: url) : Pair(sources: [url]))
            }
        }
    }
}
