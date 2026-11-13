@echo off
REM ===========================================================================
REM  ViralForge - one-time setup
REM
REM  For a person who would rather not follow the written instructions. Right
REM  click the folder, choose "Open in Terminal", then paste:
REM
REM      setup.cmd
REM
REM  It installs the three things ViralForge needs and checks they worked:
REM      1. Python 3.13
REM      2. FFmpeg 9
REM      3. the Python packages in requirements.txt   (about 2 GB)
REM
REM  Expect 10-30 minutes, most of it downloading. Nothing here uploads
REM  anything or sends data anywhere - it only downloads.
REM ===========================================================================

setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo ============================================================
echo   ViralForge setup
echo ============================================================
echo.

REM ---------------------------------------------------------------- winget
where winget >nul 2>&1
if errorlevel 1 (
    echo [X] App Installer ^(winget^) is not available on this machine.
    echo.
    echo     This script needs it. It ships with Windows 11 and Windows 10
    echo     at version 1809 and later. To get it, install "App Installer"
    echo     from the Microsoft Store, or follow START-HERE.txt instead.
    echo.
    pause
    exit /b 1
)
echo [1/4] winget found.

REM ---------------------------------------------------------------- python
echo.
echo [2/4] Python 3.13...
where python >nul 2>&1
if errorlevel 1 (
    echo       not found, installing it now.
    winget install --id Python.Python.3.13 --exact --source winget ^
        --accept-package-agreements --accept-source-agreements --silent
) else (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo       already installed: %%v
)

REM A fresh Python install is not on the PATH of the terminal that is already
REM open, so point at the one winget just put in the user's programs folder.
if errorlevel 1 (
    set "PYEXE="
    for /d %%d in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
        if exist "%%d\python.exe" set "PYEXE=%%d\python.exe"
    )
    if not defined PYEXE (
        for /d %%d in ("%ProgramFiles%\Python3*") do (
            if exist "%%d\python.exe" set "PYEXE=%%d\python.exe"
        )
    )
    if not defined PYEXE (
        echo [X] Python was installed but could not be found.
        echo     Close this window, open a new Command Prompt, and run setup.cmd again.
        pause
        exit /b 1
    )
    set "PYTHON=!PYEXE!"
) else (
    set "PYTHON=python"
)
for /f "tokens=2" %%v in ('"%PYTHON%" --version 2^>^&1') do echo       using %%v  ^(%PYTHON%^)

REM ---------------------------------------------------------------- ffmpeg
echo.
echo [3/4] FFmpeg...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo       not found, installing it now.
    winget install --id Gyan.FFmpeg --exact --source winget ^
        --accept-package-agreements --accept-source-agreements --silent
    echo.
    echo       FFmpeg installs to a fixed location but the PATH is not refreshed
    echo       in this window, so it is pointed at directly below.
    if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe" (
        set "FFBIN=%LOCALAPPDATA%\Microsoft\WinGet\Links"
    ) else (
        set "FFBIN="
    )
) else (
    for /f "tokens=3" %%v in ('ffmpeg -version 2^>^&1 ^| findstr /b "ffmpeg version"') do echo       already installed: %%v
    set "FFBIN="
)

REM ---------------------------------------------------------------- packages
echo.
echo [4/4] Python packages from requirements.txt - about 2 GB.
echo       This is the long part. It is supposed to print a lot.
echo.
"%PYTHON%" -m pip install --upgrade pip
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [X] The package install failed. The most common cause on Windows is
    echo     permissions. Open Command Prompt as Administrator and run
    echo     setup.cmd again.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------- checks
echo.
echo ============================================================
echo   checking
echo ============================================================
"%PYTHON%" -c "import fastapi, uvicorn, faster_whisper, PIL, numpy; print('  packages      ok')"
if errorlevel 1 (
    echo [X] the packages did not import. Re-run as Administrator.
    pause
    exit /b 1
)

if defined FFBIN (
    "%FFBIN%\ffmpeg.exe" -version >nul 2>&1
    if errorlevel 1 (
        echo [X] ffmpeg was installed but did not run.
        pause
        exit /b 1
    )
    echo  ffmpeg        ok  ^(!FFBIN!^)
) else (
    ffmpeg -version >nul 2>&1
    if errorlevel 1 (
        echo [X] ffmpeg is not on the PATH. Open a NEW Command Prompt and run
        echo     setup.cmd again, or add the folder holding ffmpeg.exe to the
        echo     Path environment variable.
        pause
        exit /b 1
    )
    echo  ffmpeg        ok
)

echo.
echo ============================================================
echo   Ready.
echo.
echo   Start ViralForge by double clicking:   ViralForge.vbs
echo   (not start.bat - that one flashes a black console window)
echo.
echo   It will ask where to save each job's clips, and put them there.
echo   Do that first run on wifi: the ~3 GB speech model downloads once.
echo.
echo   If anything misbehaves, run:   python -m clipforge --open
echo ============================================================
echo.
pause
endlocal
