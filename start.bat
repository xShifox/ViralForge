@echo off
REM ---------------------------------------------------------------------------
REM  ViralForge launcher
REM    1. checks Python, Node and FFmpeg
REM    2. installs dependencies on first run
REM    3. builds the UI if the bundle is missing or stale
REM    4. starts the server and opens its own window
REM ---------------------------------------------------------------------------
setlocal

cd /d "%~dp0"

echo.
echo   ViralForge - long video to short clips
echo   -------------------------------------------
echo.

REM --- Python -----------------------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
  echo   [X] Python 3.10+ was not found on PATH. Install it from python.org
  echo       and tick "Add Python to PATH" during setup.
  pause
  exit /b 1
)

REM --- Node (only needed to build the UI) -------------------------------------
set NEED_BUILD=0
if not exist "web\dist\index.html" set NEED_BUILD=1
if not exist "web\node_modules" set NEED_BUILD=1

where node >nul 2>&1
if errorlevel 1 (
  if "%NEED_BUILD%"=="1" (
    echo   [X] Node.js is required to build the web interface.
    echo       Install the LTS build from nodejs.org and run this again.
    pause
    exit /b 1
  )
) else (
  if "%NEED_BUILD%"=="1" (
    echo   [.] Building the web interface ^(first run only^)...
    pushd web
    call npm install --no-fund --no-audit >nul 2>&1
    call npm run build
    if errorlevel 1 (
      echo   [X] The UI build failed.
      popd
      pause
      exit /b 1
    )
    popd
  )
)

REM --- Python dependencies ----------------------------------------------------
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
  echo   [.] Installing Python packages ^(first run only, this takes a minute^)...
  python -m pip install --quiet -r requirements.txt
  if errorlevel 1 (
    echo   [X] Dependency installation failed.
    pause
    exit /b 1
  )
)

REM --- Run --------------------------------------------------------------------
REM  Detached by default. A pipeline spends minutes in Whisper and again encoding
REM  every clip, and it dies with the process - so closing the window that
REM  started it used to leave a job half-finished and the UI looking hung. The
REM  server keeps running, logs to data\logs\clipforge.log, and prints its PID
REM  and the command to stop it. Pass --foreground to watch the log instead.
set RUN_ARGS=--detach %*
echo RUN_ARGS | findstr /i "--foreground" >nul
if not errorlevel 1 set RUN_ARGS=%*

REM  Its own window rather than a browser tab. A tab keeps Chrome's GPU process
REM  and every other tab alive, and on a card with 8 GB that can be the difference
REM  between the transcription model fitting on the GPU and being pushed onto the
REM  CPU - roughly a tenfold slowdown.
REM
REM  The window is maximised by default: it fills the screen but keeps the title
REM  bar and the taskbar, so the close, minimise and maximise buttons are still
REM  there. Pass --fullscreen to cover the taskbar too, or --browser for a tab.
set UI_ARGS=--window
echo RUN_ARGS | findstr /i "--browser" >nul
if not errorlevel 1 set UI_ARGS=--open
echo RUN_ARGS | findstr /i "--fullscreen" >nul
if not errorlevel 1 set UI_ARGS=--window --fullscreen

echo   [.] Starting ViralForge on http://127.0.0.1:8821
echo.
python -m clipforge %UI_ARGS% %RUN_ARGS%

endlocal
