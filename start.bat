@echo off
REM SplatCraft - double-click to run (Windows). Finds Python 3.10 - 3.12 and runs run.py.
REM Arguments are passed through, e.g.  start.bat --host 0.0.0.0   start.bat --cpu   start.bat --home D:\SplatCraftData
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PY="
for %%V in (3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo Python 3.10 - 3.12 was not found.
    where winget >nul 2>nul
    if !errorlevel! equ 0 (
        choice /C YN /M "Install Python 3.12 now with winget"
        if !errorlevel! equ 1 (
            winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
            py -3.12 -c "import sys" >nul 2>nul && set "PY=py -3.12"
        )
    )
)
if not defined PY (
    echo.
    echo Please install Python 3.12 from https://www.python.org/downloads/release/python-31210/
    echo ^(tick "Add python.exe to PATH"^), then run start.bat again.
    pause
    exit /b 1
)

%PY% run.py %*
set "RC=%errorlevel%"
if not "%RC%"=="0" pause
exit /b %RC%
