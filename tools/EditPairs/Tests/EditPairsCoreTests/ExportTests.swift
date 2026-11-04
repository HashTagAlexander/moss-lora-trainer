import EditPairsCore
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest

final class ExportTests: XCTestCase {
    var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("editpairs-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    /// A tiny solid image of the given type.
    func image(_ name: String, _ type: UTType) throws -> URL {
        let url = dir.appendingPathComponent(name)
        let ctx = CGContext(data: nil, width: 8, height: 6, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(red: 1, green: 0, blue: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: 8, height: 6))
        let dest = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        return url
    }

    func testFillPairsByNameThenOrder() throws {
        let a = try image("a.png", .png), b = try image("b.png", .png)
        let tb = try image("b.jpg", .jpeg), tx = try image("x.jpg", .jpeg)
        var pairs: [Pair] = []
        Exporter.fill(&pairs, with: [a, b], asTargets: false)
        Exporter.fill(&pairs, with: [tb, tx], asTargets: true)
        XCTAssertEqual(pairs.count, 2)
        XCTAssertEqual(pairs[1].target, tb)  // b ↔ b by name
        XCTAssertEqual(pairs[0].target, tx)  // the rest in order
        Exporter.fill(&pairs, with: [tx], asTargets: true)
        XCTAssertEqual(pairs.count, 3)  // no open slot: a new pair
    }

    func testFolderLayoutCaptionsAndConversion() throws {
        let s1 = try image("one.png", .png), s2 = try image("two.tiff", .tiff), t1 = try image("t1.jpg", .jpeg)
        let s3 = try image("three.jpg", .jpeg), t2 = try image("t2.heic", .heic)
        let pairs = [
            Pair(sources: [s1, s2], target: t1, caption: "  put image 2 on the table "),
            Pair(sources: [s3], target: t2),
            Pair(sources: [s3]),  // incomplete: skipped
        ]
        let out = dir.appendingPathComponent("out")
        XCTAssertEqual(try Exporter.writeFolder(pairs: pairs, universal: "make it black and white", to: out), 2)
        let rows = try String(contentsOf: out.appendingPathComponent("edits.jsonl"), encoding: .utf8)
            .split(separator: "\n").map { try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] }
        XCTAssertEqual(rows[0]["sources"] as! [String], ["src/0001.png", "src/0001_2.png"])  // TIFF -> PNG
        XCTAssertEqual(rows[0]["target"] as! String, "tgt/0001.jpg")
        XCTAssertEqual(rows[0]["instruction"] as! String, "put image 2 on the table")
        XCTAssertEqual(rows[1]["target"] as! String, "tgt/0002.png")  // HEIC -> PNG
        XCTAssertEqual(try String(contentsOf: out.appendingPathComponent("tgt/0002.txt"), encoding: .utf8),
                       "make it black and white")
        for path in ["src/0001.png", "src/0001_2.png", "src/0002.jpg", "tgt/0001.jpg", "tgt/0002.png", "tgt/0001.txt"] {
            XCTAssertTrue(FileManager.default.fileExists(atPath: out.appendingPathComponent(path).path), path)
        }
    }

    func testMissingCaptionIsAnError() throws {
        let s = try image("s.png", .png), t = try image("t.png", .png)
        XCTAssertThrowsError(try Exporter.writeFolder(pairs: [Pair(sources: [s], target: t)], universal: " ",
                                                      to: dir.appendingPathComponent("out")))
    }

    func testZip() throws {
        let s = try image("s.png", .png), t = try image("t.png", .png)
        let s2 = try image("s2.tiff", .tiff), t2 = try image("t2.jpg", .jpeg)
        let zip = dir.appendingPathComponent("pairs.zip")
        let pairs = [Pair(sources: [s], target: t), Pair(sources: [s2, s], target: t2, caption: "put image 2 on the left")]
        XCTAssertEqual(try Exporter.writeZip(pairs: pairs, universal: "invert", to: zip), 2)
        if let keep = ProcessInfo.processInfo.environment["EDITPAIRS_KEEP_ZIP"] {  // for checking with the trainer
            try? FileManager.default.removeItem(atPath: keep)
            try FileManager.default.copyItem(at: zip, to: URL(fileURLWithPath: keep))
        }
        let list = Process()
        list.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        list.arguments = ["-Z1", zip.path]
        let pipe = Pipe()
        list.standardOutput = pipe
        try list.run()
        list.waitUntilExit()
        let names = Set(String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .split(separator: "\n").map(String.init))
        XCTAssertTrue(names.isSuperset(of: ["edits.jsonl", "src/0001.png", "tgt/0001.png", "tgt/0001.txt"]), "\(names)")
    }
}
