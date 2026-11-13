<#
.SYNOPSIS
  ViralForge launcher (PowerShell).

.DESCRIPTION
  Checks the toolchain, installs dependencies on first run, builds the web
  bundle if it is missing, then starts the server. The server is detached by
  default so closing this window does not stop a running job; pass
  -Foreground to keep it attached and watch the log.

  The interface opens in its own full-screen window rather than a browser tab,
  because a tab keeps the browser's GPU process alive and that can be the
  difference between the transcription model fitting on the GPU and going to the
  CPU. Pass -Windowed to fill the screen but keep the title bar, or -NoBrowser to
  open no interface at all.

.EXAMPLE
  .\start.ps1
  .\start.ps1 -Port 9000 -NoBrowser
  .\start.ps1 -Foreground
  .\start.ps1 -Windowed
#>
[CmdletBinding()]
param(
    [int]$Port = 8821,
    [string]$Host_ = "127.0.0.1",
    [switch]$NoBrowser,
    [switch]$Windowed,
    [switch]$Dev,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Assert-Tool($name, $probe) {
    if (Get-Command $name -ErrorAction SilentlyContinue) { return $true }
    Write-Host "  [X] $name was not found on PATH." -ForegroundColor Red
    return $false
}

Write-Host ""
Write-Host "  ViralForge - long video to short clips" -ForegroundColor Cyan
Write-Host "  -------------------------------------------" -ForegroundColor DarkCyan
Write-Host ""

if (-not (Assert-Tool "python" $null)) { Write-Host "      Install Python 3.10+ from python.org"; exit 1 }

$needBuild = -not (Test-Path "web\dist\index.html") -or -not (Test-Path "web\node_modules")
$hasNode = [bool](Get-Command node -ErrorAction SilentlyContinue)

if ($Dev) {
    if (-not $hasNode) { Write-Host "  [X] Node.js is required for --dev"; exit 1 }
    Write-Host "  [.] Installing Python packages..."
    python -m pip install --quiet -r requirements.txt
    Write-Host "  [.] Starting Vite on :5173 and the API on :$Port" -ForegroundColor Yellow
    $api = Start-Process python -ArgumentList "-m", "clipforge", "--port", $Port -PassThru -NoNewWindow
    Start-Sleep -Seconds 3
    Push-Location web
    npm install --no-fund --no-audit | Out-Null
    npm run dev
    Pop-Location
    Stop-Process -Id $api.Id -ErrorAction SilentlyContinue
    exit 0
}

if ($needBuild) {
    if (-not $hasNode) {
        Write-Host "  [X] Node.js is required to build the web interface." -ForegroundColor Red
        Write-Host "      Install the LTS build from nodejs.org, or pre-build it with:" -ForegroundColor DarkGray
        Write-Host "        cd web; npm install; npm run build" -ForegroundColor DarkGray
        exit 1
    }
    Write-Host "  [.] Building the web interface (first run only)..."
    Push-Location web
    npm install --no-fund --no-audit | Out-Null
    npm run build
    Pop-Location
}

python -c "import fastapi" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [.] Installing Python packages (first run only)..."
    python -m pip install --quiet -r requirements.txt
}

# Detached by default. A pipeline dies with the process, so closing the window
# that started it used to leave a job half-finished and the UI looking hung. The
# server keeps running, logs to data\logs\clipforge.log, and prints its PID and
# the command to stop it. Use -Foreground to watch the log instead.
$args = @("-m", "clipforge", "--host", $Host_, "--port", $Port)
# Own window rather than a browser tab: a tab keeps Chrome's GPU process and
# every other tab alive, which on an 8 GB card can be the difference between
# the transcription model fitting on the GPU and going to the CPU. -NoBrowser
# still means "open no interface at all".
if (-not $NoBrowser) { $args += "--window" }
if ($Windowed) { $args += "--windowed" }
if (-not $Foreground) { $args += "--detach" }
python @args
