# Share the webapp over the local network.
#
# Detects this machine's LAN IP, sets REFLEX_API_URL so the frontend
# Javascript points at the right backend host, and starts reflex bound
# to 0.0.0.0 so anyone on the same Wi-Fi can hit it.
#
# Usage:
#     pwsh.exe scripts\share-lan.ps1
#     powershell.exe -ExecutionPolicy Bypass -File scripts\share-lan.ps1
#
# Press Ctrl-C in the same window to stop sharing.

[CmdletBinding()]
param(
    # If you have multiple network adapters and the auto-detect picks
    # the wrong one, override here. e.g. -LanIp 192.168.1.42
    [string]$LanIp = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot

# Pick the first non-loopback, non-APIPA IPv4 address on a connected
# adapter. APIPA addresses (169.254.x.x) mean "no DHCP lease" — they'd
# advertise a URL nobody can reach.
if (-not $LanIp) {
    $candidate = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object {
            $_.IPAddress -ne "127.0.0.1" -and
            $_.IPAddress -notlike "169.254.*" -and
            $_.PrefixOrigin -ne "WellKnown"
        } |
        Sort-Object -Property InterfaceMetric |
        Select-Object -First 1

    if (-not $candidate) {
        Write-Error @"
Could not auto-detect a LAN IP. Find it manually with:
    ipconfig | findstr IPv4
then re-run as:
    scripts\share-lan.ps1 -LanIp 192.168.1.42
"@
        exit 1
    }
    $LanIp = $candidate.IPAddress
    Write-Host "Auto-detected LAN IP: $LanIp" -ForegroundColor Cyan
}

$apiUrl = "http://${LanIp}:8000"
$frontUrl = "http://${LanIp}:3000"

Write-Host ""
Write-Host "===============================================================" -ForegroundColor Yellow
Write-Host " Sharing the webapp on the local network" -ForegroundColor Yellow
Write-Host "===============================================================" -ForegroundColor Yellow
Write-Host ""
Write-Host " Friends on the same Wi-Fi can open:"
Write-Host "     $frontUrl" -ForegroundColor Green
Write-Host ""
Write-Host " Press Ctrl-C in this window to stop sharing."
Write-Host ""
Write-Host " First run on a network: Windows will prompt to allow inbound"
Write-Host " connections on ports 3000 / 8000. Click Allow on both."
Write-Host ""
Write-Host "===============================================================" -ForegroundColor Yellow
Write-Host ""

$env:REFLEX_API_URL = $apiUrl

Set-Location $repoRoot
py -3.14 -m reflex run --backend-host 0.0.0.0
