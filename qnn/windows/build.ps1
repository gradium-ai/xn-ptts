param(
    [Parameter(Mandatory=$true)][string]$QairtRoot,
    [string]$BuildDir = "$PSScriptRoot/build",
    [string]$CMake = 'cmake',
    [string]$Cargo = 'cargo'
)
$ErrorActionPreference = 'Stop'
$runner = [System.IO.Path]::GetFullPath("$PSScriptRoot/../runner")
$QairtRoot = (Resolve-Path -LiteralPath $QairtRoot).Path
if (!(Test-Path -LiteralPath "$QairtRoot/include/QNN/QnnInterface.h")) {
    throw 'QairtRoot must contain the QAIRT include/QNN headers.'
}

& $CMake -S $runner -B $BuildDir -A ARM64 "-DQAIRT_ROOT=$QairtRoot" -DPHONON_BUILD_BENCHMARK=ON
if ($LASTEXITCODE) { throw 'CMake configuration failed.' }
& $CMake --build $BuildDir --config Release
if ($LASTEXITCODE) { throw 'C++ build failed.' }
& $Cargo build --manifest-path "$PSScriptRoot/../text-ffi/Cargo.toml" --target aarch64-pc-windows-msvc --release --locked
if ($LASTEXITCODE) { throw 'Rust text frontend build failed.' }
Copy-Item -LiteralPath "$PSScriptRoot/../text-ffi/target/aarch64-pc-windows-msvc/release/ptts_text.dll" -Destination "$BuildDir/Release/ptts_text.dll"

& $CMake -S "$runner/tests" -B "$BuildDir/tests" -A ARM64
if ($LASTEXITCODE) { throw 'Test configuration failed.' }
& $CMake --build "$BuildDir/tests" --config Release
if ($LASTEXITCODE) { throw 'Test build failed.' }
$ctest = Join-Path (Split-Path (Get-Command $CMake).Source) 'ctest.exe'
& $ctest --test-dir "$BuildDir/tests" -C Release --output-on-failure
if ($LASTEXITCODE) { throw 'C++ tests failed.' }
Write-Output "Built Windows ARM64 runner, strict HTP benchmark and text frontend in $BuildDir/Release"
