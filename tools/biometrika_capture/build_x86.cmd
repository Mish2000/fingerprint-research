@echo off
setlocal

set "CAPTURE_DIR=%~dp0"
set "REPO_ROOT=%CAPTURE_DIR%..\.."
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars32.bat"
set "VCVARS_FALLBACK=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars32.bat"
set "TWAIN_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.19041.0\um\x86\twain_32.lib"
set "OBJ_DIR=%TEMP%\biometrika_capture_x86_obj"

if not exist "%VCVARS%" (
  set "VCVARS=%VCVARS_FALLBACK%"
)

if not exist "%VCVARS%" (
  echo ERROR: vcvars32.bat not found.
  exit /b 1
)

if not exist "%TWAIN_LIB%" (
  echo ERROR: twain_32.lib not found: "%TWAIN_LIB%"
  exit /b 1
)

if not exist "%CAPTURE_DIR%third_party\twain\twain.h" (
  echo ERROR: TWAIN header not found: "%CAPTURE_DIR%third_party\twain\twain.h"
  exit /b 1
)

if not exist "%CAPTURE_DIR%bin\x86" mkdir "%CAPTURE_DIR%bin\x86"
if not exist "%OBJ_DIR%" mkdir "%OBJ_DIR%"

call "%VCVARS%" >nul
if errorlevel 1 exit /b %errorlevel%

cl /nologo /EHsc /std:c++17 /W4 /DWIN32 /D_WINDOWS /D_CRT_SECURE_NO_WARNINGS ^
  /I "%CAPTURE_DIR%third_party\twain" ^
  /Fo:"%OBJ_DIR%\\" ^
  /Fe:"%CAPTURE_DIR%bin\x86\biometrika_twain_capture.exe" ^
  "%CAPTURE_DIR%src\biometrika_twain_capture.cpp" ^
  "%TWAIN_LIB%" ^
  user32.lib ^
  /link /MACHINE:X86

exit /b %errorlevel%
