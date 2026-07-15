$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Uv = Join-Path $Root "tools\uv\uv.exe"
if (-not (Test-Path -LiteralPath $Uv)) {
    $Zip = Join-Path $env:TEMP "uv-hairmask.zip"
    Invoke-WebRequest "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" -OutFile $Zip
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Uv) | Out-Null
    Expand-Archive -LiteralPath $Zip -DestinationPath (Split-Path -Parent $Uv) -Force
}
& $Uv venv (Join-Path $Root ".venv312") --python 3.12
$Python = Join-Path $Root ".venv312\Scripts\python.exe"
& $Uv pip install --python $Python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
& $Uv pip install --python $Python -r (Join-Path $Root "requirements.txt")
& $Python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
