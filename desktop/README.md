# Desktop apps

One engine, one served UI, a thin native shell per platform. `flwr serve` hosts
the chat UI on `http://127.0.0.1:<port>`; each shell below opens that UI in a
chromeless window using the OS's own browser engine, so there is no bundled
runtime and nothing extra to install.

| Platform | Shell | Browser engine | Build / run |
|---|---|---|---|
| macOS | `macos/flwr_desktop.swift` | WKWebView (Apple WebKit) | `bash desktop/macos/build.sh` builds `~/Applications/Flwr.app`. |
| Windows | `windows/flwr.ps1` (+ `flwr.cmd`) | Edge / WebView2 (Chrome fallback) | Double-click `windows/flwr.cmd`, or `powershell -File windows/flwr.ps1`. |
| Any browser / ChromeOS | none | Chrome / Edge / Safari | `flwr serve <model> --gpu` then open `http://127.0.0.1:11434`. In Chrome or Edge: menu -> Install / Create shortcut for an app window. |

Prerequisite for all three: the `flwr` binary on PATH
(`cargo install --path . --bin flwr --bin hos`).

## Configuration (all shells)

| Env var | Default | Meaning |
|---|---|---|
| `FLWR_MODEL` | `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | Model name or path to serve. |
| `FLWR_PORT` | `11599` | Port the shell serves on (kept off 11434 so it does not clash with a separate `flwr serve`). |
| `FLWR_BIN` | `~/.cargo/bin/flwr[.exe]` | Path to the `flwr` binary. |

macOS: `open ~/Applications/Flwr.app`. Windows: `flwr.cmd MyModel.gguf` or set
`FLWR_MODEL` first. The shell starts the server, waits for the port, opens the
window, and stops the server when the window closes.

## Optimized per platform

- macOS gets the GPU: pass `--gpu` (the app defaults are fine for the small
  bundled model; edit the shell's `serve` arguments to add `--gpu` for larger
  models on Apple Silicon).
- Windows and Linux run the CPU path (AVX2-accelerated on x86).
- The browser path works identically on every OS, including ChromeOS, and can be
  installed as a standalone app window from Chrome or Edge.

## File associations

Double-clicking a `.flwr` / `.hos` model file to run it is set up separately;
see [../packaging/README.md](../packaging/README.md).
