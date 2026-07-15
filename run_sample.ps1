param(
    [string]$Sample,
    [string]$Prefix = "2p",
    [string]$Output
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv312\Scripts\python.exe"

if ([string]::IsNullOrWhiteSpace($Sample)) {
    $Sample = Join-Path $Root "sample"
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $Root "results\$Prefix"
}

$Bok = Join-Path $Sample "${Prefix}_bok.jpg"
$Edof = Join-Path $Sample "${Prefix}_edof.jpg"
$Raw = Join-Path $Output "raw"
$Base = Join-Path $Output "base"
$Fine = Join-Path $Output "final"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment missing. Run .\setup.ps1 first."
}
if (-not (Test-Path -LiteralPath $Bok)) {
    throw "BOK input missing: $Bok"
}
if (-not (Test-Path -LiteralPath $Edof)) {
    throw "EDOF input missing: $Edof"
}

# When both model snapshots are already cached, prevent harmless Hub HEAD requests.
$BiRefCache = Join-Path $Root "models\models--ZhengPeng7--BiRefNet_HR-matting\refs\main"
$SapiensCache = Join-Path $Root "models\models--facebook--sapiens2-seg-0.4b\refs\main"
if ((Test-Path -LiteralPath $BiRefCache) -and (Test-Path -LiteralPath $SapiensCache)) {
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
} else {
    $env:HF_HUB_DISABLE_XET = "1"
}

& $Python (Join-Path $Root "run_pipeline.py") `
  --bok $Bok `
  --edof $Edof `
  --output $Raw

& $Python (Join-Path $Root "prepare_base_images.py") `
  --bok $Bok `
  --edof $Edof `
  --hair (Join-Path $Raw "${Prefix}_bok_hair.png") `
  --hair-probability (Join-Path $Raw "${Prefix}_bok_sapiens2_hair_probability_16bit.png") `
  --sapiens2-labels (Join-Path $Raw "${Prefix}_bok_sapiens2_labels.png") `
  --matte (Join-Path $Raw "${Prefix}_bok_biref_bok_mat4k.png") `
  --output $Base `
  --prefix $Prefix

& $Python (Join-Path $Root "extract_fine_hair.py") `
  --bok (Join-Path $Base "${Prefix}_bok.png") `
  --edof (Join-Path $Base "${Prefix}_edof.png") `
  --hair (Join-Path $Base "${Prefix}_bok_hair.png") `
  --hair-probability (Join-Path $Base "${Prefix}_bok_hair_probability_16bit.png") `
  --sapiens2-labels (Join-Path $Base "${Prefix}_bok_sapiens2_labels.png") `
  --matte (Join-Path $Base "${Prefix}_bok_mat4k.png") `
  --output $Fine

& $Python (Join-Path $Root "validate_outputs.py") --base $Base --fine $Fine --prefix $Prefix
