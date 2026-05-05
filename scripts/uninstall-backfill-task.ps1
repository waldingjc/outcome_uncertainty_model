# Removes the OutcomeUncertaintyBackfill scheduled task. Safe to re-run.

[CmdletBinding()]
param(
    [string]$TaskName = "OutcomeUncertaintyBackfill"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $existing) {
    Write-Output "Task '$TaskName' is not registered — nothing to remove."
    exit 0
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Output "Removed task: $TaskName"
