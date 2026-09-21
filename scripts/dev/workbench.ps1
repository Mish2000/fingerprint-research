[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('init-config', 'check', 'prepare-models', 'build-java', 'check-java', 'serve-java', 'start', 'stop', 'db-up', 'db-stop')]
    [string] $Command,
    [string] $EnvFile = '',
    [string] $EnvironmentName = 'fingerprint_research_cpu',
    [string] $Python = '',
    [switch] $TestProfile
)
$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (-not $EnvFile) { $EnvFile = Join-Path $repoRoot '.env' }
elseif (-not [System.IO.Path]::IsPathRooted($EnvFile)) { $EnvFile = Join-Path $repoRoot $EnvFile }
if (-not $Python) { $Python = $env:FPBENCH_PYTHON }
if (-not $Python) {
    $Python = Join-Path $env:USERPROFILE ".conda\envs\$EnvironmentName\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) {
        $candidates = @((Join-Path $env:ProgramData 'miniconda3\Scripts\conda.exe'), (Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe'))
        $condaPath = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
        if (-not $condaPath) { throw 'Provide -Python with the prepared environment executable.' }
        $info = (& $condaPath env list --json | ConvertFrom-Json)
        $prefix = $info.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvironmentName } | Select-Object -First 1
        if (-not $prefix) { throw 'Run bootstrap.ps1 or select an existing prepared -EnvironmentName.' }
        $Python = Join-Path $prefix 'python.exe'
    }
}
$previousPython = $env:FPBENCH_PYTHON
Push-Location $repoRoot
try {
    $env:FPBENCH_PYTHON = $Python
    $arguments = @((Join-Path $PSScriptRoot 'workbench.py'), $Command, '--env-file', $EnvFile)
    if ($TestProfile) { $arguments += '--test' }
    & $Python -s @arguments
    if ($LASTEXITCODE -ne 0) { throw "Workbench $Command failed; see the preceding diagnostic." }
}
finally {
    [Environment]::SetEnvironmentVariable('FPBENCH_PYTHON', $previousPython, 'Process')
    Pop-Location
}
