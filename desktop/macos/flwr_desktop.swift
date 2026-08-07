// flwr — native macOS desktop shell.
//
// A real app window (NSWindow + WKWebView) over the flwr chat UI. It uses ONLY
// Apple's built-in frameworks (AppKit, WebKit, Foundation) — no Swift packages,
// no crates, no Homebrew. Compiled by the system `swiftc`. On launch it starts
// `flwr serve` as a child process, waits for the port, and loads the UI; on quit
// it stops the server.
//
// Config via environment:
//   FLWR_MODEL  model name/path to serve   (default: Qwen2.5-0.5B-Instruct-Q4_K_M.gguf)
//   FLWR_PORT   port                       (default: 11599)
//   FLWR_BIN    path to the flwr binary    (default: ~/.cargo/bin/flwr)

import AppKit
import WebKit

let env = ProcessInfo.processInfo.environment
let model = env["FLWR_MODEL"] ?? "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
let port = env["FLWR_PORT"] ?? "11599"
let flwrBin = env["FLWR_BIN"] ?? (NSHomeDirectory() + "/.cargo/bin/flwr")
let baseURL = URL(string: "http://127.0.0.1:\(port)/")!

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var web: WKWebView!
    var server: Process?

    func applicationDidFinishLaunching(_ note: Notification) {
        startServer()

        let frame = NSRect(x: 0, y: 0, width: 920, height: 700)
        window = NSWindow(
            contentRect: frame,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered, defer: false)
        window.title = "flwr"
        window.center()
        window.setFrameAutosaveName("flwr-window")

        web = WKWebView(frame: frame)
        web.autoresizingMask = [.width, .height]
        window.contentView = web

        loadWhenReady(attempts: 0)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    // Start `flwr serve` unless something is already answering on the port.
    func startServer() {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: flwrBin)
        p.arguments = ["serve", model, "--port", port]
        do { try p.run(); server = p } catch {
            NSLog("flwr-desktop: could not start \(flwrBin): \(error)")
        }
    }

    // Poll the server, then load the UI. Retries while it spins up.
    func loadWhenReady(attempts: Int) {
        var req = URLRequest(url: baseURL)
        req.timeoutInterval = 2
        URLSession.shared.dataTask(with: req) { _, resp, _ in
            DispatchQueue.main.async {
                if let h = resp as? HTTPURLResponse, h.statusCode == 200 {
                    self.web.load(URLRequest(url: baseURL))
                } else if attempts < 120 {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        self.loadWhenReady(attempts: attempts + 1)
                    }
                } else {
                    let html = "<body style='font:14px monospace;padding:2em'>"
                        + "flwr server did not start.<br>checked: \(flwrBin) serve \(model)"
                        + "</body>"
                    self.web.loadHTMLString(html, baseURL: nil)
                }
            }
        }.resume()
    }

    func applicationWillTerminate(_ note: Notification) {
        server?.terminate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool {
        true
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
