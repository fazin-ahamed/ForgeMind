param(
    [string]$Destination = (Join-Path ([System.IO.Path]::GetTempPath()) "forgemind-clean-check")
)

$ErrorActionPreference = "Stop"
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
Push-Location $resolvedDestination
try {
    uv sync --frozen --extra dev
    uv run ruff check .
    uv run mypy src
    uv run pytest -q -m "not model and not benchmark"
} finally {
    Pop-Location
}
