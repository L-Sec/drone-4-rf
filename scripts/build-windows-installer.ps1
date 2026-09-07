param(
    [string]$Python = ".venv39\Scripts\python.exe",
    [string]$PothosRoot = "C:\Program Files\PothosSDR",
    [string]$InnoCompiler = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Resolve-SafePath([string]$Path) {
    $full = [IO.Path]::GetFullPath((Join-Path $ProjectRoot $Path))
    $prefix = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\') + '\'
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Build path escapes the project directory: $full"
    }
    return $full
}

$PythonPath = (Resolve-Path $Python).Path
$hasPyInstaller = & $PythonPath -c "import importlib.util; print('yes' if importlib.util.find_spec('PyInstaller') else 'no')"
if ($hasPyInstaller -ne "yes") {
    throw "PyInstaller is missing. Install it with: $Python -m pip install pyinstaller"
}

$versionMatch = Select-String -LiteralPath "pyproject.toml" -Pattern '^version\s*=\s*"([^"]+)"$'
if (-not $versionMatch) { throw "Could not read the project version." }
$Version = $versionMatch.Matches[0].Groups[1].Value

$BuildRoot = Resolve-SafePath "build\windows-installer"
$AppDistRoot = Join-Path $BuildRoot "app"
$PyInstallerRoot = Join-Path $BuildRoot "pyinstaller"
$OutputRoot = Resolve-SafePath "dist\windows"
$AppRoot = Join-Path $AppDistRoot "Drone4RF"

foreach ($target in @($BuildRoot, $OutputRoot)) {
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}
New-Item -ItemType Directory -Force -Path $AppDistRoot, $PyInstallerRoot, $OutputRoot | Out-Null

$arguments = @(
    "-m", "PyInstaller",
    "--noconfirm", "--clean", "--windowed", "--onedir",
    "--name", "Drone4RF",
    "--distpath", $AppDistRoot,
    "--workpath", $PyInstallerRoot,
    "--specpath", $PyInstallerRoot,
    "--collect-data", "drone4rf",
    "--hidden-import", "drone4rf.sdr.hackrf",
    "--add-data", ((Join-Path $ProjectRoot "config\default.yaml") + ";config"),
    "--add-data", ((Join-Path $ProjectRoot "config\local.example.yaml") + ";config"),
    (Join-Path $ProjectRoot "scripts\windows_launcher.py")
)

$SoapySite = Join-Path $PothosRoot "lib\python3.9\site-packages"
if (-not (Test-Path -LiteralPath (Join-Path $SoapySite "_SoapySDR.pyd"))) {
    throw "PothosSDR Python 3.9 bindings were not found under $PothosRoot."
}
$env:PATH = (Join-Path $PothosRoot "bin") + ";" + $env:PATH
$arguments += @(
    "--add-data", ((Join-Path $SoapySite "SoapySDR.py") + ";."),
    "--add-binary", ((Join-Path $SoapySite "_SoapySDR.pyd") + ";.")
)
foreach ($binary in @("SoapySDR.dll", "hackrf.dll", "libusb-1.0.dll")) {
    $source = Join-Path $PothosRoot "bin\$binary"
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing runtime file: $source" }
    $arguments += @("--add-binary", ($source + ";pothos\bin"))
}
$module = Join-Path $PothosRoot "lib\SoapySDR\modules0.8\HackRFSupport.dll"
if (-not (Test-Path -LiteralPath $module)) { throw "Missing HackRF support module: $module" }
$arguments += @("--add-binary", ($module + ";pothos\lib\SoapySDR\modules0.8"))

& $PythonPath @arguments
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

Copy-Item -LiteralPath "LICENSE" -Destination (Join-Path $AppRoot "LICENSE.txt")
$LicenseRoot = Join-Path $AppRoot "THIRD-PARTY-LICENSES"
New-Item -ItemType Directory -Force -Path $LicenseRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $PothosRoot "licenses\hackRF\COPYING") -Destination (Join-Path $LicenseRoot "hackRF-COPYING.txt")
Copy-Item -LiteralPath (Join-Path $PothosRoot "licenses\libusb\COPYING") -Destination (Join-Path $LicenseRoot "libusb-COPYING.txt")
Copy-Item -LiteralPath (Join-Path $PothosRoot "licenses\SoapyHackRF\LICENSE") -Destination (Join-Path $LicenseRoot "SoapyHackRF-LICENSE.txt")
$SoapyLicense = Get-ChildItem -LiteralPath (Join-Path $PothosRoot "licenses") -Directory | Where-Object Name -eq "SoapySDR" | Select-Object -First 1
if ($SoapyLicense) {
    Copy-Item -Path (Join-Path $SoapyLicense.FullName "*") -Destination $LicenseRoot -Recurse
}

$SelfTestData = Join-Path $BuildRoot "self-test-data"
$PreviousLocalAppData = $env:LOCALAPPDATA
try {
    $env:LOCALAPPDATA = $SelfTestData
    $SelfTestInfo = [Diagnostics.ProcessStartInfo]::new()
    $SelfTestInfo.FileName = Join-Path $AppRoot "Drone4RF.exe"
    $SelfTestInfo.Arguments = "--self-test"
    $SelfTestInfo.UseShellExecute = $false
    $SelfTestInfo.CreateNoWindow = $true
    $SelfTestProcess = [Diagnostics.Process]::Start($SelfTestInfo)
    $SelfTestProcess.WaitForExit()
    if ($SelfTestProcess.ExitCode -ne 0) {
        throw "Packaged application self-test failed with exit code $($SelfTestProcess.ExitCode)."
    }
}
finally {
    $env:LOCALAPPDATA = $PreviousLocalAppData
}

if (-not $InnoCompiler) {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe"
    )
    $InnoCompiler = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $InnoCompiler -or -not (Test-Path -LiteralPath $InnoCompiler)) {
    throw "Inno Setup 6 was not found. Install it with: winget install --id JRSoftware.InnoSetup"
}

$iss = Join-Path $ProjectRoot "packaging\windows\Drone4RF.iss"
& $InnoCompiler "/DAppVersion=$Version" "/DSourceDir=$AppRoot" "/DOutputDir=$OutputRoot" $iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed with exit code $LASTEXITCODE" }

$Installer = Join-Path $OutputRoot "Drone-4-RF-$Version-Windows-x64-Setup.exe"
if (-not (Test-Path -LiteralPath $Installer)) { throw "Installer output was not created." }
$InstallerHash = Get-FileHash -Algorithm SHA256 -LiteralPath $Installer
$ChecksumPath = "$Installer.sha256"
$ChecksumLine = $InstallerHash.Hash.ToLowerInvariant() + "  " + (Split-Path -Leaf $Installer) + "`n"
[IO.File]::WriteAllText($ChecksumPath, $ChecksumLine, [Text.UTF8Encoding]::new($false))
Write-Host "Created installer: $Installer"
Write-Host "Created checksum: $ChecksumPath"
Write-Host "User data is stored separately under %LOCALAPPDATA%\Drone4RF."
