[CmdletBinding()]
param(
    [ValidateSet('cpu', 'cuda128')] [string] $Profile = 'cpu',
    [string] $EnvironmentName = '',
    [string] $CondaExe = ''
)
$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (-not $EnvironmentName) { $EnvironmentName = "fingerprint_research_$Profile" }
if (-not $CondaExe) {
    $command = Get-Command conda -ErrorAction SilentlyContinue
    if ($command) { $CondaExe = $command.Source }
    else {
        $candidates = @((Join-Path $env:ProgramData 'miniconda3\Scripts\conda.exe'), (Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe'))
        $CondaExe = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    }
}
if (-not $CondaExe) { throw 'Provide the existing Miniconda executable with -CondaExe.' }
$info = (& $CondaExe env list --json | ConvertFrom-Json)
if ($info.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvironmentName }) {
    throw 'This environment already exists. Use another -EnvironmentName for a clean installation; existing environments are preserved.'
}
& $CondaExe create -y -n $EnvironmentName --override-channels -c conda-forge --file (Join-Path $repoRoot 'requirements\windows-native.explicit.txt')
if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed.' }
$info = (& $CondaExe env list --json | ConvertFrom-Json)
$prefix = $info.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvironmentName } | Select-Object -First 1
if (-not $prefix) { throw 'Could not locate the newly created environment.' }
$python = Join-Path $prefix 'python.exe'
$index = if ($Profile -eq 'cpu') { 'cpu' } else { 'cu128' }
$lock = Join-Path $repoRoot "requirements\windows-$Profile.lock"
& $python -s -m pip install --no-deps --extra-index-url "https://download.pytorch.org/whl/$index" -r $lock
if ($LASTEXITCODE -ne 0) { throw 'Locked Python dependency installation failed.' }
& $python -s -m pip check
if ($LASTEXITCODE -ne 0) { throw 'The installed Python dependencies are inconsistent.' }
$nodeVersion = (& node --version)
if ($nodeVersion -ne 'v24.18.0') { throw 'Install Node 24.18.0 from the official Node.js distribution, then run npm ci in apps/ui.' }
Push-Location (Join-Path $repoRoot 'apps\ui')
try {
    & npm.cmd ci
    if ($LASTEXITCODE -ne 0) { throw 'UI dependency installation failed.' }
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'UI build failed.' }
}
finally { Pop-Location }
Write-Output "Installed $EnvironmentName. Activate it, then run workbench.py init-config, build-java and prepare-models."
