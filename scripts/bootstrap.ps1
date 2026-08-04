param(
    [string]$PythonExe = "",
    [switch]$AcceptMiniMaxH3License,
    [switch]$AcceptPromptTranslatorLicense,
    [switch]$SkipPromptTranslator
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$LicenseUrl = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/af0fe5abe6fd50d632b65a82fef321c4c5c1f249/LICENSE"
$PromptTranslatorLicenseUrl = "https://huggingface.co/LiquidAI/LFM2-350M-ENJP-MT/blob/80367784d525777ad7565b24534ba5810eeac59f/LICENSE"

if ($AcceptPromptTranslatorLicense -and $SkipPromptTranslator) {
    throw "Choose either -AcceptPromptTranslatorLicense or -SkipPromptTranslator, not both."
}

Write-Host ""
Write-Host "H3 Studio first-time setup" -ForegroundColor Cyan
Write-Host "Project: $Root"
Write-Host ""
Write-Warning "MiniMax H3 weights are NOT included in this Git repository."
Write-Host "Setup downloads five fixed-revision model files (about 59.08 GiB)."
Write-Host "Review the controlling model license before continuing:"
Write-Host $LicenseUrl -ForegroundColor Cyan
Write-Host ""
Write-Host "Important: the license excludes the EU, United Kingdom, South Korea, and United States"
Write-Host "from its Applicable Territory and contains use, output, redistribution, and commercial restrictions."
Write-Host "See MODEL_TERMS.md for a non-exhaustive guide; the official license controls."
Write-Host ""

if (-not $AcceptMiniMaxH3License) {
    $answer = Read-Host "Type ACCEPT if you reviewed, are eligible to use, and accept the MiniMax H3 license"
    if ($answer.Trim() -cne "ACCEPT") {
        throw "License acknowledgement was not provided. No model download was started."
    }
}

Write-Host ""
Write-Host "The required community prompt planner is also downloaded automatically:" -ForegroundColor Cyan
Write-Host "  Qwen/Qwen3-4B-Instruct-2507"
Write-Host "  fixed revision cdbee75f17c01a7cc42f958dc650907174af0554"
Write-Host "  9 runtime files, 8,056,459,158 bytes (about 7.50 GiB)"
Write-Host "License: Apache-2.0 (no additional click-through acknowledgement)."
Write-Host "The planner runs in a separate process before H3, then exits so it is not resident with ComfyUI generation."

if (-not $AcceptPromptTranslatorLicense -and -not $SkipPromptTranslator) {
    # The old LFM route is not exposed by the normal UI. Keep first-time setup
    # focused on the required community planner; experts can opt in explicitly.
    $SkipPromptTranslator = $true
}

Write-Host ""
if ($AcceptPromptTranslatorLicense) {
    Write-Warning "The optional legacy LFM A/B translator weights are NOT included in this Git repository."
    Write-Host "This opt-in downloads eight fixed-revision files (713,822,225 bytes; about 680.75 MiB)."
    Write-Host "The LFM Open License v1.0 controls this separate model:"
    Write-Host $PromptTranslatorLicenseUrl -ForegroundColor Cyan
    Write-Host "Review MODEL_TERMS.md and the official license; the command-line opt-in is your acknowledgement."
} else {
    Write-Host "Optional legacy LFM A/B translator skipped by default." -ForegroundColor DarkGray
    Write-Host "It is not used by the normal UI. Experts can review MODEL_TERMS.md and rerun"
    Write-Host "scripts/bootstrap.ps1 with -AcceptPromptTranslatorLicense to install it later."
}

. (Join-Path $PSScriptRoot "python_runtime.ps1")
$PythonExe = Resolve-H3Python312 -Requested $PythonExe

& (Join-Path $PSScriptRoot "setup.ps1") -PythonExe $PythonExe
if ($LASTEXITCODE -ne 0) {
    throw "H3 Studio Web UI setup failed."
}

$setupComfyArguments = @{
    PythonExe = $PythonExe
    AcceptMiniMaxH3License = $true
    AcceptPromptTranslatorLicense = [bool]$AcceptPromptTranslatorLicense
    SkipPromptTranslator = [bool]$SkipPromptTranslator
}
& (Join-Path $PSScriptRoot "setup_comfy.ps1") @setupComfyArguments
if ($LASTEXITCODE -ne 0) {
    throw "H3 Studio ComfyUI setup failed. It is safe to rerun Setup-H3-Studio.cmd."
}

Write-Host ""
Write-Host "H3 Studio setup completed successfully." -ForegroundColor Green
Write-Host "For normal use, double-click Start-H3-WebUI.cmd."
