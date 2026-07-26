@echo off
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "PYTHON="
for /f "tokens=* delims=" %%i in ('where.exe python') do if not defined PYTHON set "PYTHON=%%i"
if not defined PYTHON (
    echo [ERROR] Python not found in PATH.
    pause
    exit /b 1
)
set "LAUNCHER_PATH="
for /f "delims=" %%F in ('dir /b /s /a-d "%ROOT%launcher.py" 2^>nul') do (
    set "LAUNCHER_PATH=%%F"
)
if not defined LAUNCHER_PATH (
    echo [ERROR] launcher.py not found in the workspace.
    pause
    exit /b 1
)
"%PYTHON%" "%LAUNCHER_PATH%"
pause
