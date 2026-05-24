[CmdletBinding()]
param(
    [string] $HostName = "127.0.0.1",
    [int] $Port = 8765,
    [switch] $SkipBuild
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$serviceDir = Join-Path $repoRoot "apps\sourceafis-service"
$jarPath = Join-Path $serviceDir "target\sourceafis-service-0.1.0.jar"

if (-not (Get-Command java -ErrorAction SilentlyContinue)) {
    throw "Java is required to run the SourceAFIS sidecar."
}

$needsBuild = -not (Test-Path $jarPath)
if (-not $needsBuild -and -not $SkipBuild) {
    $jar = Get-Item $jarPath
    $inputs = @(
        Get-Item (Join-Path $serviceDir "pom.xml")
        Get-ChildItem (Join-Path $serviceDir "src") -Recurse -File
    )
    $needsBuild = ($inputs | Where-Object { $_.LastWriteTimeUtc -gt $jar.LastWriteTimeUtc } | Select-Object -First 1) -ne $null
}

if ($needsBuild -and -not $SkipBuild) {
    if (-not (Get-Command mvn -ErrorAction SilentlyContinue)) {
        throw "Maven is required to build the SourceAFIS sidecar. Install Maven or run with an existing packaged jar."
    }
    Push-Location $serviceDir
    try {
        & mvn package
        if ($LASTEXITCODE -ne 0) {
            throw "Maven package failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path $jarPath)) {
    throw "Packaged sidecar jar not found at $jarPath."
}

$previousHost = $env:SOURCEAFIS_HOST
$previousPort = $env:SOURCEAFIS_PORT
try {
    $env:SOURCEAFIS_HOST = $HostName
    $env:SOURCEAFIS_PORT = [string] $Port
    & java -jar $jarPath
}
finally {
    if ($null -eq $previousHost) {
        Remove-Item Env:\SOURCEAFIS_HOST -ErrorAction SilentlyContinue
    }
    else {
        $env:SOURCEAFIS_HOST = $previousHost
    }
    if ($null -eq $previousPort) {
        Remove-Item Env:\SOURCEAFIS_PORT -ErrorAction SilentlyContinue
    }
    else {
        $env:SOURCEAFIS_PORT = $previousPort
    }
}
