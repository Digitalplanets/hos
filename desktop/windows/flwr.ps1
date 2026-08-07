# flwr - native Windows desktop launcher.
#
# A chromeless app window over the flwr chat UI, using only what ships with
# Windows: PowerShell plus the built-in Edge (WebView2) browser in --app mode.
# No packages, no crates, no install. It starts `flwr serve` as a child process,
# waits for the port, opens the app window, and stops the server when you close it.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File flwr.ps1 [model]
#   (or double-click flwr.cmd)
#
# Config via environment:
#   FLWR_MODEL  model name/path to serve   (default: Qwen2.5-0.5B-Instruct-Q4_K_M.gguf)
#   FLWR_PORT   port                        (default: 11599)
#   FLWR_BIN    path to flwr.exe            (default: %USERPROFILE%\.cargo\bin\flwr.exe)

$ErrorActionPreference = "Stop"

$model = if ($args.Count -ge 1) { $args[0] }
         elseif ($env:FLWR_MODEL) { $env:FLWR_MODEL }
         else { "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" }
$port  = if ($env:FLWR_PORT) { $env:FLWR_PORT } else { "11599" }
$bin   = if ($env:FLWR_BIN) { $env:FLWR_BIN }
         else { Join-Path $env:USERPROFILE ".cargo\bin\flwr.exe" }
if (-not (Test-Path $bin)) {
    $onPath = (Get-Command flwr.exe -ErrorAction SilentlyContinue)
    if ($onPath) { $bin = $onPath.Source }
}
$url = "http://127.0.0.1:$port/"

if (-not (Test-Path $bin) -and -not (Get-Command flwr.exe -ErrorAction SilentlyContinue)) {
    Write-Error "flwr.exe not found. Set FLWR_BIN, or run: cargo install --path . --bin flwr"
    exit 1
}

Write-Host "starting: $bin serve $model --port $port"
$server = Start-Process -FilePath $bin -ArgumentList @("serve", $model, "--port", $port) -PassThru -WindowStyle Hidden

# Wait for the server to answer on the port (up to ~60s).
$ready = $false
for ($i = 0; $i -lt 120; $i++) {
    try {
        $r = Invoke-WebRequest -Uri $url -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $ready) {
    Write-Warning "flwr server did not answer on $url"
    if ($server -and -not $server.HasExited) { $server.Kill() }
    exit 1
}

# Find a Chromium browser and open the UI as a chromeless app window.
# Edge ships with Windows; Chrome is the fallback.
$browsers = @(
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe"
)
$browser = $browsers | Where-Object { Test-Path $_ } | Select-Object -First 1

try {
    if ($browser) {
        $profileDir = Join-Path $env:LOCALAPPDATA "flwr\browser-profile"
        $ui = Start-Process -FilePath $browser -PassThru -ArgumentList @(
            "--app=$url",
            "--user-data-dir=$profileDir",
            "--window-size=920,700"
        )
        # Keep the server alive until the app window closes.
        $ui.WaitForExit()
    } else {
        # No Chromium browser found: open the default browser to the URL and
        # keep the server running until this window is closed.
        Start-Process $url
        Write-Host "flwr is serving at $url  -  press Ctrl+C or close this window to stop."
        while (-not $server.HasExited) { Start-Sleep -Seconds 1 }
    }
} finally {
    if ($server -and -not $server.HasExited) { $server.Kill() }
}
