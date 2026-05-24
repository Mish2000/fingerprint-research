[CmdletBinding()]
param(
    [string] $HostName = "127.0.0.1",
    [int] $Port = 8765
)

$ErrorActionPreference = "Stop"

function Invoke-SourceAfisHealth {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Uri
    )

    try {
        return Invoke-RestMethod -Method Get -Uri $Uri -TimeoutSec 2 -ErrorAction Stop
    }
    catch {
        return $null
    }
}

function Wait-SourceAfisHealth {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Uri,
        [int] $TimeoutSeconds = 20
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $health = Invoke-SourceAfisHealth -Uri $Uri
        if ($null -ne $health) {
            return $health
        }
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)

    return $null
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$serviceDir = Join-Path $repoRoot "apps\sourceafis-service"
$jarPath = Join-Path $serviceDir "target\sourceafis-service-0.1.0.jar"
$healthUri = "http://${HostName}:$Port/health"

if (-not (Get-Command java -ErrorAction SilentlyContinue)) {
    Write-Error "Java is required to build and run the SourceAFIS sidecar. Install a JDK, open a new PowerShell, and run java -version."
    exit 1
}

if (-not (Get-Command mvn -ErrorAction SilentlyContinue)) {
    Write-Host "Maven is required to validate the SourceAFIS sidecar build."
    Write-Host "Install Maven with:"
    Write-Host "  winget install Apache.Maven"
    Write-Host "Then open a new PowerShell and run:"
    Write-Host "  mvn -v"
    exit 1
}

$startedProcess = $null
$previousHost = $env:SOURCEAFIS_HOST
$previousPort = $env:SOURCEAFIS_PORT

try {
    Push-Location $serviceDir
    try {
        Write-Host "Running mvn test..."
        & mvn test
        if ($LASTEXITCODE -ne 0) {
            throw "mvn test failed with exit code $LASTEXITCODE."
        }

        Write-Host "Running mvn package..."
        & mvn package
        if ($LASTEXITCODE -ne 0) {
            throw "mvn package failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }

    if (-not (Test-Path $jarPath)) {
        throw "Packaged sidecar jar not found at $jarPath after mvn package."
    }

    $health = Invoke-SourceAfisHealth -Uri $healthUri
    if ($null -eq $health) {
        Write-Host "No SourceAFIS sidecar is reachable at $healthUri. Starting a temporary validation process..."
        $env:SOURCEAFIS_HOST = $HostName
        $env:SOURCEAFIS_PORT = [string] $Port

        $javaPath = (Get-Command java).Source
        $startedProcess = Start-Process `
            -FilePath $javaPath `
            -ArgumentList @("-jar", $jarPath) `
            -WorkingDirectory $serviceDir `
            -WindowStyle Hidden `
            -PassThru

        $health = Wait-SourceAfisHealth -Uri $healthUri -TimeoutSeconds 20
        if ($null -eq $health) {
            throw "SourceAFIS sidecar did not become reachable at $healthUri."
        }
    }
    else {
        Write-Host "SourceAFIS sidecar is already reachable at $healthUri. Reusing the existing process."
    }

    if ($health.status -ne "ok" -or
        $health.provider_id -ne "sourceafis_open" -or
        $health.engine -ne "SourceAFIS" -or
        $health.template_format -ne "sourceafis" -or
        $health.supports_verification -ne $true -or
        $health.supports_identification -ne $true -or
        $health.supports_quality -ne $false) {
        throw "SourceAFIS sidecar health response does not match the expected contract."
    }

    Write-Host "Health response:"
    $health | ConvertTo-Json -Depth 8
}
finally {
    if ($null -ne $startedProcess -and -not $startedProcess.HasExited) {
        Write-Host "Stopping temporary SourceAFIS sidecar process $($startedProcess.Id)..."
        Stop-Process -Id $startedProcess.Id -ErrorAction SilentlyContinue
        $startedProcess.WaitForExit(5000) | Out-Null
    }

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
