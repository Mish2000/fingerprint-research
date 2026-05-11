@echo off
setlocal

set "PROBE_DIR=%~dp0"
set "REPO_ROOT=%PROBE_DIR%..\.."
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars32.bat"
set "TWAIN_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.19041.0\um\x86\twain_32.lib"

if not exist "%VCVARS%" (
  echo ERROR: vcvars32.bat not found: "%VCVARS%"
  exit /b 1
)

if not exist "%TWAIN_LIB%" (
  echo ERROR: twain_32.lib not found: "%TWAIN_LIB%"
  exit /b 1
)

if not exist "%PROBE_DIR%third_party\twain\twain.h" (
  echo ERROR: official TWAIN header not found: "%PROBE_DIR%third_party\twain\twain.h"
  exit /b 1
)

if not exist "%PROBE_DIR%bin\x86" mkdir "%PROBE_DIR%bin\x86"

call "%VCVARS%" >nul
if errorlevel 1 exit /b %errorlevel%

cl /nologo /EHsc /std:c++17 /W4 /DWIN32 /D_WINDOWS /D_CRT_SECURE_NO_WARNINGS ^
  /I "%PROBE_DIR%third_party\twain" ^
  /Fo:"%PROBE_DIR%bin\x86\\" ^
  /Fe:"%PROBE_DIR%bin\x86\twain_source_probe.exe" ^
  "%PROBE_DIR%src\twain_source_probe.cpp" ^
  "%TWAIN_LIB%" ^
  user32.lib ^
  /link /MACHINE:X86

exit /b %errorlevel%
