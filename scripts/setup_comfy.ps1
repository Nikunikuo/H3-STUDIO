param(
    [string]$PythonExe = "",
    [switch]$VerifyOnly,
    [switch]$SkipModelHash,
    [switch]$AcceptMiniMaxH3License
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "python_runtime.ps1")
Assert-H3Command -Name "git.exe" -InstallHint "Install Git for Windows from https://git-scm.com/download/win and reopen this terminal."

$AttentionBackend = if ([string]::IsNullOrWhiteSpace($env:H3_ATTENTION_BACKEND)) {
    "sage"
} else {
    $env:H3_ATTENTION_BACKEND.Trim().ToLowerInvariant()
}
if ($AttentionBackend -notin @("sage", "pytorch")) {
    throw "H3_ATTENTION_BACKEND must be 'sage' or 'pytorch'."
}

$ComfyVenv = Join-Path $Root ".comfy-venv"
$ComfyPython = Join-Path $ComfyVenv "Scripts\python.exe"
$ComfySource = Join-Path $Root ".upstream\ComfyUI"
$Requirements = Join-Path $Root "requirements.comfy.txt"
$ModelLockPath = Join-Path $Root "comfy_models.lock.json"
$ComfySha = "14b05228cef127ce529bc0c08660770d4af3e9a8"
$SageVersion = "2.2.0+cu130torch2.10.0andhigher.post6"
$SageWheelName = "sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl"
$SageWheelUrl = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0%2Bcu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl"
$SageWheelBytes = [int64]16656067
$SageWheelSha256 = "1635283f5c01ec3cda58a784d0d7eabbcaffaf9511d1b263db4750e1ed7958bb"

function Test-FixedCheckout {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedSha,
        [Parameter(Mandatory = $true)][string]$Name
    )

    if (-not (Test-Path -LiteralPath (Join-Path $Path ".git"))) {
        throw "$Name checkout is missing: $Path"
    }
    $actual = (& git -C $Path rev-parse HEAD | Select-Object -Last 1).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read the $Name checkout revision."
    }
    if ($actual -ne $ExpectedSha) {
        throw "$Name SHA mismatch: expected $ExpectedSha, got $actual"
    }
    $changes = @(& git -C $Path status --porcelain --untracked-files=all)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect the $Name checkout."
    }
    if ($changes.Count -gt 0) {
        throw "$Name checkout has local or untracked files: $Path"
    }
}

function Set-FixedCheckout {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$ExpectedSha,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $gitDirectory = Join-Path $Path ".git"
    $isNewCheckout = $false
    if (-not (Test-Path -LiteralPath $gitDirectory)) {
        if (Test-Path -LiteralPath $Path) {
            $existing = @(Get-ChildItem -LiteralPath $Path -Force)
            if ($existing.Count -gt 0) {
                throw "$Name target exists but is not a Git checkout: $Path"
            }
        } else {
            New-Item -ItemType Directory -Path (Split-Path -Parent $Path) -Force | Out-Null
        }
        & git clone --filter=blob:none --no-checkout $Repository $Path
        if ($LASTEXITCODE -ne 0) {
            throw "$Name clone failed."
        }
        $isNewCheckout = $true
    }

    if (-not $isNewCheckout) {
        $changes = @(& git -C $Path status --porcelain --untracked-files=all)
        if ($LASTEXITCODE -ne 0) {
            throw "Could not inspect the $Name checkout."
        }
        if ($changes.Count -gt 0) {
            throw "$Name checkout has local changes. Preserve or remove them before changing revisions: $Path"
        }
    }

    & git -C $Path fetch --depth 1 origin $ExpectedSha
    if ($LASTEXITCODE -ne 0) {
        throw "$Name fetch failed for $ExpectedSha."
    }
    & git -C $Path checkout --detach $ExpectedSha
    if ($LASTEXITCODE -ne 0) {
        throw "$Name checkout failed for $ExpectedSha."
    }
    Test-FixedCheckout -Path $Path -ExpectedSha $ExpectedSha -Name $Name
}

function Read-ModelLock {
    if (-not (Test-Path -LiteralPath $ModelLockPath)) {
        throw "Comfy model lock is missing: $ModelLockPath"
    }
    return Get-Content -LiteralPath $ModelLockPath -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Test-ComfyModels {
    param(
        [Parameter(Mandatory = $true)]$Lock,
        [switch]$FullHash
    )

    $total = [int64]0
    foreach ($entry in $Lock.files) {
        $path = Join-Path $Root ([string]$entry.path)
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Comfy model file is missing: $path"
        }
        $item = Get-Item -LiteralPath $path
        $expectedSize = [int64]$entry.size
        if ($item.Length -ne $expectedSize) {
            throw "Comfy model size mismatch for $path`: expected $expectedSize, got $($item.Length)"
        }
        $total += $item.Length

        if ($FullHash) {
            Write-Host "SHA-256: $($entry.path)"
            $actualHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
            $expectedHash = ([string]$entry.sha256).ToLowerInvariant()
            if ($actualHash -ne $expectedHash) {
                throw "Comfy model SHA-256 mismatch for $path`: expected $expectedHash, got $actualHash. Preserve it for inspection, then remove or replace it before rerunning setup."
            }
        }
    }

    $expectedTotal = [int64]$Lock.verification.total_bytes
    if ($total -ne $expectedTotal) {
        throw "Comfy model total size mismatch: expected $expectedTotal, got $total"
    }
    Write-Host "Comfy models verified: $($Lock.files.Count) files, $total bytes$(if ($FullHash) { ', SHA-256 matched' } else { ', sizes matched' })."
}

function Get-MissingModelBytes {
    param([Parameter(Mandatory = $true)]$Lock)

    $missingBytes = [int64]0
    foreach ($entry in $Lock.files) {
        $path = Join-Path $Root ([string]$entry.path)
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $actualSize = (Get-Item -LiteralPath $path).Length
            if ($actualSize -ne [int64]$entry.size) {
                throw "An existing Comfy model has the wrong size: $path. Preserve it for inspection, then remove or replace it before rerunning setup."
            }
        } else {
            $missingBytes += [int64]$entry.size
        }
    }
    return $missingBytes
}

function Assert-H3NvidiaPreflight {
    Assert-H3Command -Name "nvidia-smi.exe" -InstallHint "Install or update the NVIDIA display driver, then reboot."
    $rows = @(& nvidia-smi.exe --id=0 --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>&1)
    if ($LASTEXITCODE -ne 0 -or $rows.Count -eq 0) {
        throw "nvidia-smi could not inspect CUDA GPU 0. Install or update the NVIDIA display driver, then reboot."
    }
    $parts = @($rows[-1] -split ',' | ForEach-Object { $_.Trim() })
    if ($parts.Count -lt 3) {
        throw "Unexpected nvidia-smi response: $($rows[-1])"
    }
    $memoryMiB = [int64]0
    if (-not [int64]::TryParse($parts[1], [ref]$memoryMiB)) {
        throw "Could not parse GPU memory from nvidia-smi: $($rows[-1])"
    }
    if ($memoryMiB -lt 30720) {
        throw "CUDA GPU 0 requires at least 30 GiB VRAM; detected $memoryMiB MiB on $($parts[0])."
    }
    Write-Host "GPU preflight: $($parts[0]), $memoryMiB MiB VRAM, NVIDIA driver $($parts[2])."
}

function Test-ComfyRuntime {
    if (-not (Test-Path -LiteralPath $ComfyPython -PathType Leaf)) {
        throw "ComfyUI Python environment is missing: $ComfyPython"
    }
    $runtimeProbe = @'
import importlib.metadata
import pathlib
import sys

import torch
import torchaudio
import torchao
import torchvision

root = pathlib.Path(sys.argv[1])
attention_backend = sys.argv[2]
sys.path.insert(0, str(root))
import comfy

assert tuple(sys.version_info[:2]) == (3, 12), sys.version
assert torch.__version__ == '2.13.0+cu130', torch.__version__
assert torchvision.__version__ == '0.28.0+cu130', torchvision.__version__
assert torchaudio.__version__ == '2.11.0+cu130', torchaudio.__version__
assert torchao.__version__ == '0.17.0', torchao.__version__
assert torch.version.cuda == '13.0', torch.version.cuda
assert torch.cuda.is_available()
gpu_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
assert gpu_memory_gib >= 30.0, f'{torch.cuda.get_device_name(0)} has only {gpu_memory_gib:.1f} GiB VRAM'

probe = torch.zeros((1, 2, 64), dtype=torch.float32)
resampled = torchaudio.functional.resample(probe, 16000, 32000)
assert resampled.shape == (1, 2, 128), resampled.shape

if attention_backend == 'sage':
    from sageattention import sageattn

    assert importlib.metadata.version('sageattention') == '2.2.0+cu130torch2.10.0andhigher.post6'
    assert importlib.metadata.version('triton-windows') == '3.7.1.post27'
    q = torch.randn((1, 128, 4, 64), device='cuda', dtype=torch.float16)
    sage_output = sageattn(q, q, q, tensor_layout='NHD')
    assert sage_output.shape == q.shape
    assert torch.isfinite(sage_output).all()

print('python', sys.version.split()[0])
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('torchaudio', torchaudio.__version__)
print('torchao', torchao.__version__)
print('attention_backend', attention_backend)
if attention_backend == 'sage':
    print('sageattention', importlib.metadata.version('sageattention'))
    print('triton-windows', importlib.metadata.version('triton-windows'))
print('cuda', torch.version.cuda)
print('gpu', torch.cuda.get_device_name(0))
print('gpu_memory_gib', f'{gpu_memory_gib:.1f}')
print('comfy_source', root)
print('audio_resample', 'ok')
print('sageattention_kernel', 'ok' if attention_backend == 'sage' else 'skipped')
'@
    & $ComfyPython -c $runtimeProbe $ComfySource $AttentionBackend
    if ($LASTEXITCODE -ne 0) {
        throw "ComfyUI runtime validation failed."
    }
}

$modelLock = Read-ModelLock
$missingBytes = Get-MissingModelBytes -Lock $modelLock

if ($env:OS -ne "Windows_NT") {
    throw "This pinned H3 Studio setup currently supports 64-bit Windows only."
}

if ($VerifyOnly) {
    Test-FixedCheckout -Path $ComfySource -ExpectedSha $ComfySha -Name "ComfyUI"
    Test-ComfyRuntime
    Test-ComfyModels -Lock $modelLock -FullHash:(-not $SkipModelHash)
    Write-Host "ComfyUI verification completed without modifying the environment."
    return
}

if ($missingBytes -gt 0 -and -not $AcceptMiniMaxH3License) {
    $licenseUrl = [string]$modelLock.source.license_url
    throw @"
MiniMax H3 model files are not included in this repository.
Before downloading $missingBytes bytes, review the MiniMax H3 Community License:
$licenseUrl

The license has territory, use, redistribution, output, and commercial restrictions.
If you are eligible and accept it, rerun with -AcceptMiniMaxH3License or use Setup-H3-Studio.cmd.
"@
}

if ($missingBytes -gt 0) {
    $driveName = [IO.Path]::GetPathRoot($Root).TrimEnd('\').TrimEnd(':')
    $drive = Get-PSDrive -Name $driveName
    $requiredFree = $missingBytes + 25GB
    if ([int64]$drive.Free -lt $requiredFree) {
        throw "Not enough free disk space for the runtime, Comfy models, and safety margin: need $requiredFree bytes, available $($drive.Free)."
    }
    Write-Host "MiniMax H3 license accepted for this setup invocation."
    Write-Host "Model download: $missingBytes bytes from fixed revision $($modelLock.source.revision)."
}

Assert-H3NvidiaPreflight

$PythonExe = Resolve-H3Python312 -Requested $PythonExe
Write-Host "Using Python 3.12: $PythonExe"

Set-FixedCheckout `
    -Path $ComfySource `
    -Repository "https://github.com/Comfy-Org/ComfyUI.git" `
    -ExpectedSha $ComfySha `
    -Name "ComfyUI"

if (-not (Test-Path -LiteralPath $ComfyPython -PathType Leaf)) {
    & $PythonExe -m venv $ComfyVenv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the isolated ComfyUI environment: $ComfyVenv"
    }
}
if (-not (Test-H3Python312 -Path $ComfyPython)) {
    throw "The existing .comfy-venv is not a 64-bit Python 3.12 environment. Rename or remove '$ComfyVenv', then rerun setup."
}

& $ComfyPython -m pip install --upgrade pip==26.2
if ($LASTEXITCODE -ne 0) {
    throw "ComfyUI pip bootstrap failed."
}
& $ComfyPython -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu130 `
    torch==2.13.0+cu130 torchvision==0.28.0+cu130 torchaudio==2.11.0+cu130
if ($LASTEXITCODE -ne 0) {
    throw "CUDA PyTorch installation failed."
}
& $ComfyPython -m pip install --requirement $Requirements
if ($LASTEXITCODE -ne 0) {
    throw "ComfyUI dependency installation failed."
}

if ($AttentionBackend -eq "sage") {
    & $ComfyPython -m pip install triton-windows==3.7.1.post27
    if ($LASTEXITCODE -ne 0) {
        throw "Pinned Triton Windows installation failed."
    }

    $SageWheelDirectory = Join-Path $Root ".cache\wheels"
    $SageWheelPath = Join-Path $SageWheelDirectory $SageWheelName
    New-Item -ItemType Directory -Path $SageWheelDirectory -Force | Out-Null
    $downloadSageWheel = -not (Test-Path -LiteralPath $SageWheelPath -PathType Leaf)
    if (-not $downloadSageWheel) {
        $sageWheelItem = Get-Item -LiteralPath $SageWheelPath
        $sageWheelHash = (Get-FileHash -LiteralPath $SageWheelPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $downloadSageWheel = $sageWheelItem.Length -ne $SageWheelBytes -or $sageWheelHash -ne $SageWheelSha256
    }
    if ($downloadSageWheel) {
        $SagePartialPath = "$SageWheelPath.partial"
        Remove-Item -LiteralPath $SagePartialPath -Force -ErrorAction SilentlyContinue
        Invoke-WebRequest -Uri $SageWheelUrl -OutFile $SagePartialPath -UseBasicParsing
        $downloadedSageWheel = Get-Item -LiteralPath $SagePartialPath
        $downloadedSageHash = (Get-FileHash -LiteralPath $SagePartialPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($downloadedSageWheel.Length -ne $SageWheelBytes -or $downloadedSageHash -ne $SageWheelSha256) {
            throw "Pinned SageAttention wheel failed size or SHA-256 verification."
        }
        Remove-Item -LiteralPath $SageWheelPath -Force -ErrorAction SilentlyContinue
        Move-Item -LiteralPath $SagePartialPath -Destination $SageWheelPath -Force
    }
    $installedSage = @(& $ComfyPython -c "import importlib.metadata; print(importlib.metadata.version('sageattention'))" 2>$null)
    if ($LASTEXITCODE -ne 0 -or $installedSage.Count -eq 0 -or $installedSage[-1].Trim() -ne $SageVersion) {
        & $ComfyPython -m pip install --no-deps --force-reinstall $SageWheelPath
        if ($LASTEXITCODE -ne 0) {
            throw "Pinned SageAttention installation failed."
        }
    } else {
        Write-Host "Pinned SageAttention is already installed; reinstall skipped."
    }
} else {
    Write-Host "PyTorch attention selected; SageAttention wheel installation skipped."
}

if ($missingBytes -gt 0) {
    $modelRoot = Join-Path $Root "models\comfy"
    New-Item -ItemType Directory -Path $modelRoot -Force | Out-Null
    $patterns = @($modelLock.files | ForEach-Object {
        ([string]$_.path).Substring("models/comfy/".Length).Replace('\', '/')
    })
    $repoId = [string]$modelLock.source.repo_id
    $revision = [string]$modelLock.source.revision
    & $ComfyPython -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], allow_patterns=sys.argv[4:], max_workers=2)" `
        $repoId $revision $modelRoot @patterns
    if ($LASTEXITCODE -ne 0) {
        throw "Fixed-revision Comfy model download failed. It is safe to rerun this script to resume."
    }
} else {
    Write-Host "All fixed-revision Comfy model files are already present; download skipped."
}

Test-ComfyRuntime
Test-ComfyModels -Lock $modelLock -FullHash:(-not $SkipModelHash)

Write-Host "ComfyUI setup complete."
Write-Host "ComfyUI SHA: $ComfySha"
Write-Host "Model revision: $($modelLock.source.revision)"
Write-Host "Attention backend: $AttentionBackend$(if ($AttentionBackend -eq 'sage') { " ($SageVersion; verified wheel)" } else { '' })"
