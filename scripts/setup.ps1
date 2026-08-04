param(
    [string]$PythonExe = "",
    [switch]$WithLegacyDiffusers
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "python_runtime.ps1")

$PythonExe = Resolve-H3Python312 -Requested $PythonExe
$Venv = Join-Path $Root ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$WebRequirements = Join-Path $Root "requirements.webui.txt"
$Upstream = Join-Path $Root ".upstream\diffusers"
$DiffusersSha = "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"

Write-Host "Using Python 3.12: $PythonExe"
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    & $PythonExe -m venv $Venv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the isolated Web UI environment: $Venv"
    }
}
if (-not (Test-H3Python312 -Path $VenvPython)) {
    throw "The existing .venv is not a 64-bit Python 3.12 environment. Rename or remove '$Venv', then rerun setup."
}

& $VenvPython -m pip install --upgrade pip==26.2
if ($LASTEXITCODE -ne 0) {
    throw "Web UI pip bootstrap failed."
}
& $VenvPython -m pip install --requirement $WebRequirements
if ($LASTEXITCODE -ne 0) {
    throw "Web UI dependency installation failed."
}
& $VenvPython -c "import av, fastapi, multipart, psutil, uvicorn; print('webui python', __import__('sys').version.split()[0]); print('av', av.__version__); print('fastapi', fastapi.__version__); print('uvicorn', uvicorn.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "Web UI runtime validation failed."
}

if ($WithLegacyDiffusers) {
    Assert-H3Command -Name "git.exe" -InstallHint "Install Git for Windows from https://git-scm.com/download/win and reopen this terminal."
    & $VenvPython -m pip install --index-url https://download.pytorch.org/whl/cu130 `
        torch==2.13.0+cu130 torchvision==0.28.0+cu130
    if ($LASTEXITCODE -ne 0) {
        throw "Legacy CUDA PyTorch installation failed."
    }

    if (-not (Test-Path -LiteralPath (Join-Path $Upstream ".git"))) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $Upstream) -Force | Out-Null
        & git clone --filter=blob:none --no-checkout https://github.com/huggingface/diffusers.git $Upstream
        if ($LASTEXITCODE -ne 0) {
            throw "Diffusers clone failed."
        }
    }

    & git -C $Upstream fetch --depth 1 origin $DiffusersSha
    if ($LASTEXITCODE -ne 0) {
        throw "Diffusers fetch failed for $DiffusersSha."
    }
    & git -C $Upstream checkout --detach $DiffusersSha
    if ($LASTEXITCODE -ne 0) {
        throw "Diffusers checkout failed for $DiffusersSha."
    }
    $ActualSha = (& git -C $Upstream rev-parse HEAD | Select-Object -Last 1).Trim()
    if ($ActualSha -ne $DiffusersSha) {
        throw "Diffusers SHA mismatch: expected $DiffusersSha, got $ActualSha"
    }

    & $VenvPython -m pip install --editable $Upstream
    & $VenvPython -m pip install --requirement (Join-Path $Root "requirements.runtime.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "Legacy Diffusers dependency installation failed."
    }
    & $VenvPython -c "import torch, diffusers, transformers, torchao, av; assert torch.cuda.is_available(); print('torch', torch.__version__); print('cuda', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0)); print('diffusers', diffusers.__version__); print('transformers', transformers.__version__); print('torchao', torchao.__version__); print('av', av.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "Legacy Diffusers runtime validation failed."
    }
    Write-Host "Legacy Diffusers comparison environment is ready."
}

Write-Host "H3 Studio Web UI environment is ready."
