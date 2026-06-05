# Share the webapp over the internet via a Cloudflare quick tunnel.
#
# This requires `cloudflared` to be on PATH. Install once with:
#     winget install --id Cloudflare.cloudflared
#
# Reflex needs two tunnels — one for the frontend (3000) and one for
# the backend (8000) — because the browser-side JS talks to both. The
# frontend has to know the backend's PUBLIC tunnel URL at build time,
# so this script:
#   1. Spawns the backend tunnel and waits for its trycloudflare URL
#   2. Sets REFLEX_API_URL to that URL
#   3. Spawns the frontend tunnel
#   4. Starts reflex
#   5. Prints the shareable frontend URL when both tunnels are ready
#
# Press Ctrl-C in this window to tear everything down (cloudflared
# child processes get killed too).
#
# Usage:
#     pwsh.exe scripts\share-tunnel.ps1
#     powershell.exe -ExecutionPolicy Bypass -File scripts\share-tunnel.ps1

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot

# Reflex's frontend dev server is Vite under the hood, and Vite refuses
# requests whose Host header isn't an address it's bound to (a DNS-
# rebinding mitigation). The random *.trycloudflare.com hostname trips
# that check every time. Patch the generated vite.config to allow any
# host — we're already gated behind cloudflared, the extra check is
# noise for our use case.
function Enable-AllHosts-In-ViteConfig {
    param([string]$RepoRoot)
    $configPath = Join-Path $RepoRoot ".web\vite.config.js"
    if (-not (Test-Path $configPath)) {
        # .web/ gets regenerated on first `reflex run`. We don't bail —
        # the first reflex start below will create it, and the script
        # retries the patch after that initial build.
        return $false
    }
    $content = Get-Content $configPath -Raw
    if ($content -match "allowedHosts") {
        return $true   # already patched
    }
    $patched = $content -replace "(server\s*:\s*\{)", "`$1`r`n    allowedHosts: true,"
    if ($patched -eq $content) {
        Write-Warning "Could not locate `server: {` in vite.config.js to patch."
        return $false
    }
    Set-Content -Path $configPath -Value $patched -NoNewline -Encoding utf8
    Write-Host "  patched .web/vite.config.js (allowedHosts: true)" -ForegroundColor DarkGray
    return $true
}

# Confirm cloudflared is installed before we kick anything off.
$cloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
if (-not $cloudflared) {
    Write-Error @"
cloudflared is not on PATH. Install once with:
    winget install --id Cloudflare.cloudflared
"@
    exit 1
}

# Helper — start a cloudflared quick tunnel for a localhost port, then
# scrape stderr for the trycloudflare.com URL it prints on startup.
function Start-Tunnel {
    param([int]$LocalPort, [string]$Label)

    $logFile = Join-Path $env:TEMP "cloudflared-$Label-$PID.log"
    # cloudflared writes its URL to stderr, so merge streams into the log.
    $proc = Start-Process cloudflared `
        -ArgumentList @("tunnel", "--url", "http://localhost:$LocalPort") `
        -PassThru -NoNewWindow -RedirectStandardError $logFile `
        -RedirectStandardOutput "$logFile.out"

    Write-Host "  starting $Label tunnel (pid $($proc.Id))..." -ForegroundColor DarkGray

    # Poll the log file for the public URL. Cloudflare quick tunnels
    # typically come up in 3-10 seconds.
    $url = $null
    $deadline = (Get-Date).AddSeconds(45)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 500
        if (-not (Test-Path $logFile)) { continue }
        $content = Get-Content $logFile -Raw -ErrorAction SilentlyContinue
        if ($content -match "https://[a-zA-Z0-9-]+\.trycloudflare\.com") {
            $url = $matches[0]
            break
        }
    }

    if (-not $url) {
        Write-Error "Timed out waiting for $Label tunnel URL. See $logFile"
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        exit 1
    }
    return @{ Url = $url; Process = $proc; Log = $logFile }
}

Write-Host ""
Write-Host "===============================================================" -ForegroundColor Yellow
Write-Host " Spinning up Cloudflare tunnels..." -ForegroundColor Yellow
Write-Host "===============================================================" -ForegroundColor Yellow

# Backend tunnel first — frontend has to know its URL.
$backendTunnel  = Start-Tunnel -LocalPort 8000 -Label "backend"
Write-Host "  backend  -> $($backendTunnel.Url)" -ForegroundColor Green

$frontendTunnel = Start-Tunnel -LocalPort 3000 -Label "frontend"
Write-Host "  frontend -> $($frontendTunnel.Url)" -ForegroundColor Green

# Make sure the tunnels die when this script exits (Ctrl-C, error, ...).
$tunnelPids = @($backendTunnel.Process.Id, $frontendTunnel.Process.Id)
Register-EngineEvent PowerShell.Exiting -Action {
    foreach ($p in $tunnelPids) {
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
    }
} | Out-Null

Write-Host ""
Write-Host "===============================================================" -ForegroundColor Yellow
Write-Host " Webapp is live on the internet" -ForegroundColor Yellow
Write-Host "===============================================================" -ForegroundColor Yellow
Write-Host ""
Write-Host " Share this URL with friends:"
Write-Host "     $($frontendTunnel.Url)" -ForegroundColor Green
Write-Host ""
Write-Host " Press Ctrl-C in this window to tear down both tunnels."
Write-Host "===============================================================" -ForegroundColor Yellow
Write-Host ""

$env:REFLEX_API_URL = $backendTunnel.Url

# Try patching now in case .web/ already exists from a prior run.
$patched = Enable-AllHosts-In-ViteConfig -RepoRoot $repoRoot

# Background watcher — reflex regenerates .web/vite.config.js on first
# start (and after some config changes), which would clobber our patch.
# Poll for 30 seconds and re-apply if needed; that's long enough for
# reflex's initial Bun install + Vite scaffolding even on a cold machine.
$watcherJob = Start-Job -ScriptBlock {
    param($RepoRoot)
    $configPath = Join-Path $RepoRoot ".web\vite.config.js"
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 500
        if (Test-Path $configPath) {
            $content = Get-Content $configPath -Raw
            if ($content -notmatch "allowedHosts") {
                $patched = $content -replace "(server\s*:\s*\{)", "`$1`r`n    allowedHosts: true,"
                if ($patched -ne $content) {
                    Set-Content -Path $configPath -Value $patched -NoNewline -Encoding utf8
                }
            }
        }
    }
} -ArgumentList $repoRoot

try {
    Set-Location $repoRoot
    py -3.14 -m reflex run
}
finally {
    # Kill cloudflared child processes if reflex exits for any reason.
    foreach ($p in $tunnelPids) {
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
    }
    Stop-Job -Job $watcherJob -ErrorAction SilentlyContinue
    Remove-Job -Job $watcherJob -Force -ErrorAction SilentlyContinue
    Write-Host "Tunnels stopped." -ForegroundColor DarkGray
}
