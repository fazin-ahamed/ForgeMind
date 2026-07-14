[CmdletBinding()]
param(
    [ValidateSet("pdf", "html", "all")]
    [string]$Format = "all"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PaperRoot = Join-Path $RepoRoot ".forgemind-private\paper"
$Source = Join-Path $PaperRoot "paper.md"
$Bibliography = Join-Path $PaperRoot "references.bib"
$OutputRoot = Join-Path $PaperRoot "build"

if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
    throw "Private paper source not found: $Source"
}

$Pandoc = Get-Command pandoc -ErrorAction Stop
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$CommonArguments = @(
    $Source,
    "--standalone",
    "--toc",
    "--number-sections",
    "--metadata", "title=ForgeMind: An Evidence-First Context Compiler for Consumer GPUs"
)

if (Test-Path -LiteralPath $Bibliography -PathType Leaf) {
    $CommonArguments += @("--citeproc", "--bibliography", $Bibliography)
}

if ($Format -in @("html", "all")) {
    & $Pandoc.Source @CommonArguments --output (Join-Path $OutputRoot "forgemind-paper.html")
    if ($LASTEXITCODE -ne 0) {
        throw "pandoc HTML export failed with exit code $LASTEXITCODE"
    }
}

if ($Format -in @("pdf", "all")) {
    $XeLaTeX = Get-Command xelatex -ErrorAction Stop
    & $Pandoc.Source @CommonArguments --pdf-engine $XeLaTeX.Source `
        --output (Join-Path $OutputRoot "forgemind-paper.pdf")
    if ($LASTEXITCODE -ne 0) {
        throw "pandoc PDF export failed with exit code $LASTEXITCODE"
    }
}

Write-Output "Paper export completed in $OutputRoot"
