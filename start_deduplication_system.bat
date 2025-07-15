@echo off
echo ============================================================
echo               DeduplicationSystem Launcher
echo ============================================================
echo.

REM Check if Python is installed
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in your PATH.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo Running DeduplicationSystem...
echo This will automatically install all needed dependencies
echo and start the application.
echo.
echo (This might take a minute on first run)
echo.

REM Run the launcher script
python run_app.py

echo.
if %ERRORLEVEL% NEQ 0 (
    echo Application failed to start. Please check the error messages above.
    echo.
    pause
) else (
    echo Application closed. You can restart it by running this script again.
    timeout /t 5
)