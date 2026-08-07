# flwr installer for Windows. Installs the `hos` engine and the `flwr` app.
#
#   irm https://www.flwr.systems/install.ps1 | iex
#
# Downloads a prebuilt x86-64 binary (uses AVX2/FMA at runtime where present).
$ErrorActionPreference = "Stop"
$repo = "Digitalplanets/hos"
$dest = if ($env:FLWR_BIN_DIR) { $env:FLWR_BIN_DIR } else { "$env:LOCALAPPDATA\flwr\bin" }
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$url = "https://github.com/$repo/releases/latest/download/flwr-windows-x86_64.zip"
$zip = "$env:TEMP\flwr-windows.zip"
Write-Host "flwr  downloading prebuilt binaries ..."
Invoke-WebRequest -Uri $url -OutFile $zip
Expand-Archive -Path $zip -DestinationPath $dest -Force
Remove-Item $zip -Force

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$dest*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$dest", "User")
    Write-Host "flwr  added $dest to your PATH (restart the terminal to pick it up)"
}
Write-Host "flwr  installed hos + flwr to $dest"
Write-Host "flwr  try it:  flwr pull flwr-bloom  ;  flwr run flwr-bloom"
