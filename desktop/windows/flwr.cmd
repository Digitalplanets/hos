@echo off
REM Double-click launcher for the flwr desktop app on Windows.
REM Runs flwr.ps1 (which starts `flwr serve` and opens a chromeless app window).
REM Pass a model name/path as the first argument, or set FLWR_MODEL.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0flwr.ps1" %*
