import AppKit
import EditPairsCore
import SwiftUI
import UniformTypeIdentifiers

@main
struct EditPairsApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate

    var body: some Scene {
        WindowGroup("Edit Pairs") {
            ContentView()
                .frame(minWidth: 760, minHeight: 480)
        }
        .windowResizability(.contentMinSize)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        #if DEBUG
        if let path = ProcessInfo.processInfo.environment["EDITPAIRS_SNAPSHOT"] { return snapshot(to: path) }
        #endif
        // run as a normal app with a Dock icon and keyboard focus, also when started with `swift run`
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    #if DEBUG
    /// Development aid: draws the window offscreen with the images in EDITPAIRS_DEMO (a folder) to a PNG, then quits.
    private func snapshot(to path: String) {
        var pairs: [Pair] = []
        let demo = ProcessInfo.processInfo.environment["EDITPAIRS_DEMO"].map { [URL(fileURLWithPath: $0)] } ?? []
        let images = Exporter.images(in: demo)
        Exporter.fill(&pairs, with: images.filter { $0.path.contains("/src/") }, asTargets: false)
        Exporter.fill(&pairs, with: images.filter { $0.path.contains("/tgt/") }, asTargets: true)
        if pairs.count > 1 { pairs[1].caption = "put the red square on the left" }
        if pairs.count > 2 { pairs[2].target = nil }
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 900, height: 640), styleMask: [.titled],
                              backing: .buffered, defer: false)
        window.contentView = NSHostingView(rootView: ContentView(pairs: pairs, universal: "turn this image to black and white"))
        window.contentView?.wantsLayer = true
        window.orderBack(nil)
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            let view = window.contentView!
            let scale = window.backingScaleFactor
            let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(view.bounds.width * scale),
                                       pixelsHigh: Int(view.bounds.height * scale), bitsPerSample: 8, samplesPerPixel: 4,
                                       hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0,
                                       bitsPerPixel: 0)!
            let ctx = NSGraphicsContext(bitmapImageRep: rep)!.cgContext
            ctx.translateBy(x: 0, y: view.bounds.height * scale)
            ctx.scaleBy(x: scale, y: -scale)  // layers draw top-down
            NSColor.windowBackgroundColor.setFill()
            NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: false)
            view.bounds.fill()
            view.layer?.render(in: ctx)
            try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
            NSApp.terminate(nil)
        }
    }
    #endif

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

// MARK: - Main window

struct ContentView: View {
    @State private var pairs: [Pair]
    @State private var universal: String
    @State private var status = ""
    @State private var problem = ""

    init(pairs: [Pair] = [], universal: String = "") {
        _pairs = State(initialValue: pairs)
        _universal = State(initialValue: universal)
    }

    private var ready: Int { pairs.filter(\.isComplete).count }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if pairs.isEmpty {
                emptyState
            } else {
                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(Array(pairs.enumerated()), id: \.element.id) { index, _ in
                            PairRow(number: index + 1, pair: $pairs[index], universal: universal) {
                                pairs.remove(at: index)
                            }
                        }
                    }
                    .padding(12)
                }
            }
            Divider()
            footer
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Universal caption").font(.headline)
                TextField("e.g. turn this image to black and white", text: $universal, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...4)
                Text("Used for every pair that has no caption of its own.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack(spacing: 10) {
                BulkDrop(title: "Drop sources", subtitle: "control images, many at once") { urls in
                    Exporter.fill(&pairs, with: urls, asTargets: false)
                }
                BulkDrop(title: "Drop targets", subtitle: "results, many at once") { urls in
                    Exporter.fill(&pairs, with: urls, asTargets: true)
                }
            }
            Text("Images pair up by file name when names match (cat.jpg ↔ cat.png), otherwise in name order.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(12)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "photo.on.rectangle.angled").font(.system(size: 40)).foregroundStyle(.tertiary)
            Text("Drop source and target images above, or add pairs one at a time.")
                .foregroundStyle(.secondary)
            Button("Add pair") { pairs.append(Pair()) }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var footer: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("\(ready) of \(pairs.count) pairs ready").font(.callout)
                if !problem.isEmpty {
                    Text(problem).font(.caption).foregroundStyle(.red)
                } else if !status.isEmpty {
                    Text(status).font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer()
            Button("Add pair") { pairs.append(Pair()) }
            Button("Clear all", role: .destructive) { pairs.removeAll(); status = ""; problem = "" }
                .disabled(pairs.isEmpty)
            Button("Export folder…") { export(zip: false) }.disabled(ready == 0)
            Button("Export .zip…") { export(zip: true) }
                .keyboardShortcut("e", modifiers: .command)
                .buttonStyle(.borderedProminent)
                .disabled(ready == 0)
        }
        .padding(12)
    }

    private func export(zip: Bool) {
        problem = ""
        let destination: URL
        if zip {
            let panel = NSSavePanel()
            panel.allowedContentTypes = [.zip]
            panel.nameFieldStringValue = "edit_pairs.zip"
            guard panel.runModal() == .OK, let url = panel.url else { return }
            destination = url
        } else {
            let panel = NSOpenPanel()
            panel.canChooseDirectories = true
            panel.canChooseFiles = false
            panel.canCreateDirectories = true
            panel.prompt = "Export here"
            panel.message = "Pick an empty folder: src/, tgt/ and edits.jsonl are written into it."
            guard panel.runModal() == .OK, let url = panel.url else { return }
            destination = url
        }
        do {
            let count = zip
                ? try Exporter.writeZip(pairs: pairs, universal: universal, to: destination)
                : try Exporter.writeFolder(pairs: pairs, universal: universal, to: destination)
            let skipped = pairs.count - count
            status = "Exported \(count) pairs to \(destination.lastPathComponent)"
                + (skipped > 0 ? " (\(skipped) incomplete skipped)" : "")
            NSWorkspace.shared.activateFileViewerSelecting([destination])
        } catch {
            problem = error.localizedDescription
        }
    }
}

// MARK: - Pieces

/// A wide drop zone for many images at once (files or folders). Click to pick files instead.
struct BulkDrop: View {
    let title: String
    let subtitle: String
    let onDrop: ([URL]) -> Void
    @State private var targeted = false

    var body: some View {
        VStack(spacing: 2) {
            Label(title, systemImage: "square.and.arrow.down.on.square").font(.headline)
            Text(subtitle).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, minHeight: 64)
        .background(RoundedRectangle(cornerRadius: 10).fill(targeted ? Color.accentColor.opacity(0.15) : Color.secondary.opacity(0.06)))
        .overlay(RoundedRectangle(cornerRadius: 10)
            .strokeBorder(targeted ? Color.accentColor : Color.secondary.opacity(0.4), style: StrokeStyle(lineWidth: 1.5, dash: [6])))
        .contentShape(Rectangle())
        .onTapGesture { if let urls = pickImages(multiple: true) { onDrop(urls) } }
        .dropDestination(for: URL.self) { urls, _ in
            let images = Exporter.images(in: urls)
            onDrop(images)
            return !images.isEmpty
        } isTargeted: { targeted = $0 }
    }
}

struct PairRow: View {
    let number: Int
    @Binding var pair: Pair
    let universal: String
    let onDelete: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Text("\(number)").font(.callout.monospacedDigit()).foregroundStyle(.secondary).frame(width: 28, alignment: .trailing)
                .padding(.top, 6)
            VStack(alignment: .leading, spacing: 4) {
                Text(pair.sources.count > 1 ? "Sources (\(pair.sources.count))" : "Source").font(.caption).foregroundStyle(.secondary)
                HStack(spacing: 6) {
                    ForEach(pair.sources, id: \.self) { url in
                        ImageSlot(url: url, onDrop: { replace(url, with: $0) }, onRemove: { pair.sources.removeAll { $0 == url } })
                    }
                    if pair.sources.count < Pair.maxSources {
                        ImageSlot(url: nil, hint: pair.sources.isEmpty ? "Source" : "+ image",
                                  onDrop: { urls in pair.sources.append(contentsOf: urls.prefix(Pair.maxSources - pair.sources.count)) },
                                  onRemove: {})
                    }
                }
            }
            Image(systemName: "arrow.right").foregroundStyle(.secondary).padding(.top, 50)
            VStack(alignment: .leading, spacing: 4) {
                Text("Target").font(.caption).foregroundStyle(.secondary)
                ImageSlot(url: pair.target, hint: "Target",
                          onDrop: { pair.target = $0.first }, onRemove: { pair.target = nil })
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("Caption").font(.caption).foregroundStyle(.secondary)
                TextField(universal.isEmpty ? "caption for this pair" : universal, text: $pair.caption, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(3...6)
                Text(pair.caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                     ? (universal.isEmpty ? "No caption yet" : "Uses the universal caption")
                     : "Own caption")
                    .font(.caption2)
                    .foregroundStyle(pair.instruction(universal: universal).isEmpty ? Color.red : Color.secondary)
            }
            .frame(minWidth: 200)
            Button(role: .destructive, action: onDelete) { Image(systemName: "trash") }
                .buttonStyle(.borderless)
                .help("Remove this pair")
                .padding(.top, 6)
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.secondary.opacity(0.05)))
    }

    private func replace(_ old: URL, with urls: [URL]) {
        guard let new = urls.first, let i = pair.sources.firstIndex(of: old) else { return }
        pair.sources[i] = new
    }
}

/// A thumbnail that takes a dropped image (or a click to pick one); the x removes it.
struct ImageSlot: View {
    let url: URL?
    var hint = ""
    let onDrop: ([URL]) -> Void
    let onRemove: () -> Void
    @State private var targeted = false
    @State private var image: NSImage?

    private let side: CGFloat = 112

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Group {
                if let image {
                    Image(nsImage: image).resizable().scaledToFit()
                } else if url == nil {
                    VStack(spacing: 4) {
                        Image(systemName: "plus.viewfinder").font(.title2)
                        Text(hint).font(.caption)
                    }
                    .foregroundStyle(.secondary)
                } else {
                    ProgressView().controlSize(.small)
                }
            }
            .frame(width: side, height: side)
            .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.08)))
            .overlay(RoundedRectangle(cornerRadius: 8)
                .strokeBorder(targeted ? Color.accentColor : Color.secondary.opacity(url == nil ? 0.4 : 0.15),
                              style: StrokeStyle(lineWidth: targeted ? 2 : 1, dash: url == nil ? [5] : [])))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            if url != nil {
                Button(action: onRemove) {
                    Image(systemName: "xmark.circle.fill").symbolRenderingMode(.palette)
                        .foregroundStyle(.white, .black.opacity(0.6))
                }
                .buttonStyle(.borderless)
                .padding(4)
            }
        }
        .help(url?.lastPathComponent ?? "Drop an image here, or click to choose one")
        .contentShape(Rectangle())
        .onTapGesture { if let urls = pickImages(multiple: url == nil) { onDrop(urls) } }
        .dropDestination(for: URL.self) { urls, _ in
            let images = Exporter.images(in: urls)
            onDrop(images)
            return !images.isEmpty
        } isTargeted: { targeted = $0 }
        .task(id: url) { image = await thumbnail(url, side: side * 2) }
    }
}

// MARK: - Helpers

@MainActor
func pickImages(multiple: Bool) -> [URL]? {
    let panel = NSOpenPanel()
    panel.allowsMultipleSelection = multiple
    panel.canChooseDirectories = multiple
    panel.allowedContentTypes = [.image]
    guard panel.runModal() == .OK else { return nil }
    let images = Exporter.images(in: panel.urls)
    return images.isEmpty ? nil : images
}

/// A small, orientation-corrected thumbnail, made off the main thread.
@MainActor
func thumbnail(_ url: URL?, side: CGFloat) async -> NSImage? {
    guard let url else { return nil }
    let cg = await Task.detached(priority: .userInitiated) { () -> CGImage? in
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: side,
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }.value
    guard let cg else { return nil }
    return NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
}
