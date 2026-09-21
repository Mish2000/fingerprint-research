[CmdletBinding()]
param([string] $HostName = "127.0.0.1", [int] $Port = 8765, [switch] $SkipBuild)
$ErrorActionPreference = "Stop"
if ($HostName -ne "127.0.0.1") { throw "The local sidecar must use loopback." }
$python = if ($env:FPBENCH_PYTHON) { $env:FPBENCH_PYTHON } else { (Get-Command python -ErrorAction Stop).Source }
$command = Join-Path $PSScriptRoot "workbench.py"
$previousUrl = $env:SOURCEAFIS_SERVICE_URL
try {
    $env:SOURCEAFIS_SERVICE_URL = "http://127.0.0.1:$Port"
    if (-not $SkipBuild) {
        & $python $command build-java
        if ($LASTEXITCODE -ne 0) { throw "SourceAFIS build failed." }
    }
    & $python $command serve-java
    if ($LASTEXITCODE -ne 0) { throw "SourceAFIS execution failed." }
}
finally {
    [Environment]::SetEnvironmentVariable("SOURCEAFIS_SERVICE_URL", $previousUrl, "Process")
}
