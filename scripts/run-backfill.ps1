# Wrapper that the Windows Task Scheduler invokes once per logon (10-min
# delayed). Runs the historical backfill, then shows a Windows toast
# notification with a summary, and writes a daily log file to data/logs/.
#
# Designed to be safe to invoke manually too:
#     pwsh.exe scripts\run-backfill.ps1
#     powershell.exe -ExecutionPolicy Bypass -File scripts\run-backfill.ps1
#
# Pre-flight: the Python script itself probes /status (free against quota)
# at startup and exits cleanly with `skipped_pre_flight=true` in the summary
# JSON if there's no budget — so re-runs on the same day are zero-cost.

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Resolve repo root (this script lives in <repo>/scripts/).
$repoRoot = Split-Path -Parent $PSScriptRoot

# Ensure log dir exists and pick today's log file.
$logDir = Join-Path $repoRoot "data\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir ("backfill-{0}.log" -f (Get-Date -Format "yyyy-MM-dd"))
$summaryFile = Join-Path $logDir "last-run.json"

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    $line = "$ts $Message"
    Write-Output $line

    # Use a .NET StreamWriter opened with FileShare.ReadWrite so concurrent
    # readers (e.g. tail -f, less +F, log viewers) don't lock us out.
    # Add-Content / Out-File both open with exclusive write, which fails
    # the moment anything else has the file open even for read.
    try {
        $stream = [System.IO.FileStream]::new(
            $logFile,
            [System.IO.FileMode]::Append,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::ReadWrite
        )
        $writer = [System.IO.StreamWriter]::new($stream, [System.Text.Encoding]::UTF8)
        $writer.WriteLine($line)
        $writer.Flush()
        $writer.Dispose()
        $stream.Dispose()
    } catch {
        # Last-resort fallback: don't let a logging failure kill the run.
        # The line is still in the script's stdout via Write-Output above.
        Write-Output "WARN: log write failed: $_"
    }
}

function Write-PlainSummary {
    # Always-visible fallback: write the summary as plain text next to the
    # log files, so even if the toast machinery silently fails the user has
    # a file to glance at. Safe to call multiple times.
    param(
        [Parameter(Mandatory=$true)][string]$Title,
        [Parameter(Mandatory=$true)][string]$Body
    )
    $summaryTxt = Join-Path $logDir "last-run.txt"
    try {
        $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
        $content = @"
$now

$Title
---
$Body
"@
        # FileShare.ReadWrite so anything tailing the file doesn't lock us out.
        $stream = [System.IO.FileStream]::new(
            $summaryTxt,
            [System.IO.FileMode]::Create,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::ReadWrite
        )
        $writer = [System.IO.StreamWriter]::new($stream, [System.Text.Encoding]::UTF8)
        $writer.Write($content)
        $writer.Flush()
        $writer.Dispose()
        $stream.Dispose()
    } catch {
        Write-Log "Failed to write last-run.txt: $_"
    }
}

function Show-Toast {
    param(
        [Parameter(Mandatory=$true)][string]$Title,
        [Parameter(Mandatory=$true)][string]$Body
    )

    # Always write the plain-text summary first — that one Just Works.
    Write-PlainSummary -Title $Title -Body $Body

    try {
        # WinRT bindings come built-in on Windows 10+; no external module needed.
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null
        [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime] | Out-Null

        # Build the Toast XML. ToastGeneric supports two text fields: title and body.
        $escapedTitle = [System.Security.SecurityElement]::Escape($Title)
        $escapedBody  = [System.Security.SecurityElement]::Escape($Body)
        $xmlString = @"
<toast>
  <visual>
    <binding template="ToastGeneric">
      <text>$escapedTitle</text>
      <text>$escapedBody</text>
    </binding>
  </visual>
</toast>
"@

        $xmlDoc = [Windows.Data.Xml.Dom.XmlDocument]::new()
        $xmlDoc.LoadXml($xmlString)
        $toast = [Windows.UI.Notifications.ToastNotification]::new($xmlDoc)

        # Windows requires an Application User Model ID (AUMID) registered
        # somewhere on the system before it will display a toast. Custom
        # strings are silently dropped — that's why our earlier "OutcomeUncertaintyBackfill"
        # didn't show. PowerShell's own AUMID is always present, so we route
        # the toast through it. (BurntToast does the same thing under the hood.)
        $aumid = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($aumid).Show($toast)
        Write-Log "Toast displayed under AUMID: $aumid"
    } catch {
        Write-Log "Toast notification failed (summary still saved to last-run.txt): $_"
    }
}

function Format-Summary {
    param([Parameter(Mandatory=$true)]$Summary)

    $queue = $Summary.queue
    $pending   = if ($queue.PSObject.Properties['pending'])   { $queue.pending }   else { 0 }
    $completed = if ($queue.PSObject.Properties['completed']) { $queue.completed } else { 0 }
    $noAccess  = if ($queue.PSObject.Properties['no_access']) { $queue.no_access } else { 0 }
    $failed    = if ($queue.PSObject.Properties['failed'])    { $queue.failed }    else { 0 }
    $totalQueue = $pending + $completed + $noAccess + $failed
    $pctDone = if ($totalQueue -gt 0) { [int]([math]::Round(100 * ($completed + $noAccess) / $totalQueue)) } else { 0 }

    # `skipped_pre_flight` is only present in the summary JSON when the
    # pre-flight check actually triggered — on a normal successful run
    # it's absent. Strict mode treats a missing property as an error, so
    # check for existence before reading. Same defensive pattern as the
    # queue counts above.
    $wasSkipped = $false
    if ($Summary.PSObject.Properties['skipped_pre_flight']) {
        $wasSkipped = [bool]$Summary.skipped_pre_flight
    }
    if ($wasSkipped) {
        return "Skipped: budget already at $($Summary.daily_calls_remaining). Queue: $completed done, $pending pending ($pctDone%)."
    }

    $jobsToday = $Summary.jobs_completed
    $fixturesToday = $Summary.fixtures_total
    $budget = $Summary.daily_calls_remaining
    return @"
Today: $jobsToday jobs, $fixturesToday fixtures
Queue: $completed done / $pending pending ($pctDone%)
Daily calls remaining: $budget
"@
}

Write-Log "=== Backfill wrapper start ==="
Write-Log "Repo root: $repoRoot"

$exitCode = 0
try {
    Set-Location $repoRoot

    # Run the Python script. Pipe both stdout and stderr into the log via
    # 2>&1 so we capture everything. PowerShell preserves $LASTEXITCODE.
    $output = & python -m src.ingest.historical_backfill --no-raw 2>&1
    $exitCode = $LASTEXITCODE
    foreach ($line in $output) { Write-Log $line }
    Write-Log "Python exit code: $exitCode"

    if (Test-Path $summaryFile) {
        $summary = Get-Content $summaryFile -Raw | ConvertFrom-Json
        $body = Format-Summary -Summary $summary
        if ($exitCode -eq 0) {
            Show-Toast -Title "Outcome Uncertainty Backfill" -Body $body
        } else {
            Show-Toast -Title "Backfill exited with code $exitCode" -Body $body
        }
    } else {
        Show-Toast -Title "Backfill: no summary file" -Body "Check $logFile"
    }
} catch {
    $exitCode = 1
    Write-Log "ERROR: $($_.Exception.Message)"
    Write-Log $_.ScriptStackTrace
    Show-Toast -Title "Backfill FAILED" -Body "$($_.Exception.Message)`nSee $logFile"
}

Write-Log "=== Backfill wrapper done (exit=$exitCode) ==="
exit $exitCode
