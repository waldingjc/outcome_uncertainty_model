# Registers a Windows Task Scheduler task that runs the backfill wrapper
# 10 minutes after each user logon. Idempotent — re-running replaces the
# existing task.
#
# Usage (in a PowerShell terminal opened as your normal user, no admin needed):
#     scripts\install-backfill-task.ps1
#
# To run manually after install:           Start-ScheduledTask -TaskName OutcomeUncertaintyBackfill
# To inspect:                              Get-ScheduledTask -TaskName OutcomeUncertaintyBackfill
# To remove:                               scripts\uninstall-backfill-task.ps1

[CmdletBinding()]
param(
    [string]$TaskName = "OutcomeUncertaintyBackfill"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$wrapperPath = Join-Path $repoRoot "scripts\run-backfill.ps1"

if (-not (Test-Path $wrapperPath)) {
    throw "Wrapper script not found at $wrapperPath"
}

Write-Output "Repo root:   $repoRoot"
Write-Output "Wrapper:     $wrapperPath"
Write-Output "Task name:   $TaskName"
Write-Output "Trigger:     10 minutes after $env:USERNAME logs on"
Write-Output ""

# --- Action: invoke powershell.exe with the wrapper ---
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$wrapperPath`"" `
    -WorkingDirectory $repoRoot

# --- Trigger: at logon for the current user, with 10-minute delay ---
# `-AtLogOn -User <name>` produces a CIM trigger object whose Delay we set
# explicitly to "PT10M" (ISO 8601 duration: 10 minutes). The delay is what
# gives the OS time to settle before we start hitting the API.
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = "PT10M"

# --- Settings ---
# - 30-minute hard runtime cap (kills the task if it hangs)
# - Retry once if it fails, after a 1-hour delay
# - Start when next available if a fire was missed (e.g. PC was off)
# - Run on battery as well as AC (laptop friendly)
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Hours 1) `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

# --- Principal: run as the current user, only when logged on ---
# `-LogonType Interactive` means no password is stored; the task only runs
# when the user is logged on (matches the user's "always logged in" stance).
# `-RunLevel Limited` means standard (non-elevated) privileges.
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

# --- Register (replacing any existing task with the same name) ---
$task = Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Outcome Uncertainty Model — daily api-football fixture backfill. Fires 10 minutes after logon, runs the historical backfill, exits cleanly when daily quota is exhausted." `
    -Force

Write-Output "Registered task: $TaskName"
Write-Output ""
Write-Output "Useful follow-ups:"
Write-Output "  Run it now (smoke test):  Start-ScheduledTask -TaskName $TaskName"
Write-Output "  See last result:          Get-ScheduledTaskInfo -TaskName $TaskName"
Write-Output "  Logs:                     $repoRoot\data\logs\backfill-<date>.log"
Write-Output "  Last-run summary JSON:    $repoRoot\data\logs\last-run.json"
Write-Output "  Remove the task:          scripts\uninstall-backfill-task.ps1"
