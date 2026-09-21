[CmdletBinding()]
param([string] $OutputDirectory = (Join-Path $PSScriptRoot '..\..\.local\scanner'))
$ErrorActionPreference = 'Stop'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Install the official Visual Studio C++ build tools to build the optional TWAIN helper.' }
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'The Visual Studio x86/x64 C++ tools are missing.' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars32.bat'
$sdkRoot = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\Lib'
$twain = Get-ChildItem -LiteralPath $sdkRoot -Directory | Sort-Object Name -Descending |
    ForEach-Object { Join-Path $_.FullName 'um\x86\twain_32.lib' } |
    Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $twain) { throw 'The Windows SDK x86 TWAIN import library is missing.' }
$buildDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $buildDirectory -Force | Out-Null
$environmentBefore = @{}
Get-ChildItem Env: | ForEach-Object { $environmentBefore[$_.Name] = $_.Value }
try {
    # Only this child shell and this script's process environment are changed.
    $variables = & cmd.exe /d /s /c "call `"$vcvars`" >nul && set"
    if ($LASTEXITCODE -ne 0) { throw 'Visual Studio environment initialization failed.' }
    foreach ($line in $variables) {
        if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
    }
    $source = Join-Path $PSScriptRoot 'src\biometrika_twain_capture.cpp'
    $include = Join-Path $PSScriptRoot 'third_party\twain'
    $output = Join-Path $buildDirectory 'biometrika_twain_capture.exe'
    & cl.exe /nologo /EHsc /std:c++17 /W4 /DWIN32 /D_WINDOWS /D_CRT_SECURE_NO_WARNINGS "/I$include" "/Fo$buildDirectory\" "/Fe$output" $source $twain user32.lib /link /MACHINE:X86
    if ($LASTEXITCODE -ne 0) { throw 'TWAIN helper compilation failed.' }
    Write-Output "Built x86 TWAIN helper: $output"
}
finally {
    Get-ChildItem Env: | Where-Object { -not $environmentBefore.ContainsKey($_.Name) } | ForEach-Object { [Environment]::SetEnvironmentVariable($_.Name, $null, 'Process') }
    foreach ($key in $environmentBefore.Keys) { [Environment]::SetEnvironmentVariable($key, $environmentBefore[$key], 'Process') }
}
