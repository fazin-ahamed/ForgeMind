param(
    [string]$Destination = (Join-Path ([System.IO.Path]::GetTempPath()) "forgemind-clean-check")
)

$ErrorActionPreference = "Stop"

function Assert-NativeSuccess([string]$Operation) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE"
    }
}

$tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$resolvedDestination = [System.IO.Path]::GetFullPath($Destination)
$comparison = if ($IsWindows) {
    [System.StringComparison]::OrdinalIgnoreCase
} else {
    [System.StringComparison]::Ordinal
}

if (
    $resolvedDestination -eq $tempRoot -or
    -not $resolvedDestination.StartsWith($tempRoot, $comparison)
) {
    throw "Clean-check destination must be a child of the OS temp directory: $tempRoot"
}

if (Test-Path -LiteralPath $resolvedDestination) {
    Remove-Item -LiteralPath $resolvedDestination -Recurse -Force
}

git clone --no-local . $resolvedDestination
Assert-NativeSuccess "git clone"
Push-Location $resolvedDestination
try {
    uv sync --frozen --extra dev
    Assert-NativeSuccess "uv sync"
    uv run ruff check .
    Assert-NativeSuccess "Ruff"
    uv run mypy src
    Assert-NativeSuccess "mypy"
    uv run python -m pytest -q -m "not model and not benchmark"
    Assert-NativeSuccess "pytest"
} finally {
    Pop-Location
}
