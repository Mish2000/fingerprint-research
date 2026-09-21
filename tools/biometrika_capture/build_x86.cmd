@echo off
setlocal
powershell.exe -NoProfile -File "%~dp0build_x86.ps1" %*
exit /b %errorlevel%
