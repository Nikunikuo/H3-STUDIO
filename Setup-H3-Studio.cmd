@echo off
chcp 65001 >nul
cd /d "%~dp0"
title H3 Studio - First-time setup
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1" %*
if errorlevel 1 (
  echo.
  echo H3 Studio setup did not complete. Read the message above, then rerun this file.
  pause
  exit /b 1
)
echo.
pause
