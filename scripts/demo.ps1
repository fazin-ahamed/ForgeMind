[CmdletBinding()]
param(
    [string]$Archive = "examples/showcase-repository",
    [string]$Database = "artifacts/showcase.sqlite",
    [string]$Summary = "benchmark-results/m5-showcase/summary.json",
    [string]$AssetRoot = "",
    [int]$Port = 8088,
    [string]$Question = "Why did sessions fail after the April migration?"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $AssetRoot) {
    $AssetRoot = $RepoRoot
}
$HostAddress = "127.0.0.1"
$Uv = Get-Command uv -ErrorAction Stop

function Assert-NativeSuccess([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

Push-Location $RepoRoot
try {
    if (-not (Test-Path -LiteralPath $Archive -PathType Container)) {
        throw "Demo archive not found: $Archive"
    }
    if (-not (Test-Path -LiteralPath $Summary -PathType Leaf)) {
        throw "Frozen fallback summary not found: $Summary"
    }
    if ($HostAddress -ne "127.0.0.1") {
        throw "Demo host must remain on loopback"
    }

    $env:HF_HUB_OFFLINE = "1"
    & $Uv.Source run python scripts/verify_assets.py assets/manifest.json --root $AssetRoot
    Assert-NativeSuccess "Asset verification"

    & $Uv.Source run forgemind doctor
    Assert-NativeSuccess "forgemind doctor"

    & $Uv.Source run forgemind ingest $Archive --db $Database
    Assert-NativeSuccess "Incremental showcase ingestion"

    $Arguments = @(
        "run", "forgemind", "web",
        "--db", $Database,
        "--port", $Port,
        "--summary", $Summary
    )
    $Process = Start-Process `
        -FilePath $Uv.Source `
        -ArgumentList $Arguments `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -PassThru

    Write-Output "ForgeMind demo PID: $($Process.Id)"
    Write-Output "Open http://$HostAddress`:$Port"
    Write-Output "Ask: $Question"
    Write-Output "If the live model fails, use the clearly labelled Previously frozen run table."
} finally {
    Pop-Location
}
