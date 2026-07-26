REM LINT:IGNORE S007
@echo off
REM LINT:IGNORE W028, S019
set "CODEPAGE=936"
REM LINT:IGNORE W028, SEC013
chcp %CODEPAGE% > nul
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "PYTHON="
for /f "tokens=* delims=" %%i in ('where.exe python') do if not defined PYTHON set "PYTHON=%%i"
if not defined PYTHON (
    echo [ERROR] Python not found in PATH.
    pause
    exit /b 1
)
echo Starting Math Evaluator...
set "LAUNCHER_PATH="
for /f "delims=" %%F in ('dir /b /s /a-d "%ROOT%launcher.py" 2^>nul') do (
    set "LAUNCHER_PATH=%%F"
)
if not defined LAUNCHER_PATH (
    echo [ERROR] launcher.py not found in the workspace.
    pause
    exit /b 1
)
"%PYTHON%" "%LAUNCHER_PATH%" 2>"%TEMP%\math_eval_error.log"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Program crashed. See log below:
    type "%TEMP%\math_eval_error.log"
    REM LINT:IGNORE W001
    pause
)
