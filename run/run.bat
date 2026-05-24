@echo off
title AI Image Classifier - Launcher
echo ==============================================================
echo     AI Image Classifier - Launcher
echo ==============================================================
echo.

:: Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found on your system!
    echo Please download and install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

:: Run the Python launcher script
python "%~dp0run.py"
pause
