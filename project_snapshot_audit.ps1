$ErrorActionPreference = "Continue"
$root = (Get-Location).Path
$reportsDir = Join-Path $root "reports"
if (-not (Test-Path $reportsDir)) { New-Item -ItemType Directory -Path $reportsDir | Out-Null }
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $reportsDir "project_snapshot_audit_$timestamp.txt"

& {
    Write-Output "DYNAMIC PRICING E-COMMERCE — READ-ONLY PROJECT SNAPSHOT"
    Write-Output "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Output "Working directory: $root"
    Write-Output ""

    Write-Output "========== 1. GIT STATUS =========="
    if (Get-Command git -ErrorAction SilentlyContinue) {
        git status --short
        git branch --show-current
    } else { Write-Output "Git executable not found." }
    Write-Output ""

    Write-Output "========== 2. PYTHON ENVIRONMENT =========="
    python --version
    python -c "import numpy,pandas,torch; print('NumPy:',numpy.__version__); print('Pandas:',pandas.__version__); print('PyTorch:',torch.__version__)"
    Write-Output ""

    Write-Output "========== 3. CORE SOURCE FILES =========="
    $files = @(
        "src\models\itanet.py",
        "src\models\sales_predictor.py",
        "src\models\ppo_agent.py",
        "src\environment\pricing_environment.py",
        "src\environment\pricing_economics.py",
        "src\environment\reward.py",
        "src\environment\counterfactual_simulator.py",
        "scripts\build_state_train_only.py",
        "scripts\train_sales_predictor_train_only.py",
        "scripts\train_ppo.py",
        "scripts\evaluate_ppo_reproduction.py",
        "scripts\audit_topk_matched_cohort_v4.py"
    )
    foreach ($f in $files) {
        if (Test-Path $f) {
            $item = Get-Item $f
            Write-Output ("FOUND`t{0}`t{1} bytes`tModified: {2}" -f $item.FullName, $item.Length, $item.LastWriteTime)
        } else { Write-Output "MISSING`t$f" }
    }
    Write-Output ""

    Write-Output "========== 4. KEY MODEL AND STATE ARTIFACTS =========="
    $artifacts = @(
        "data\processed\products_clean.parquet",
        "data\processed\state_train_only\state.npy",
        "data\processed\state_train_only\state_row_indices.npy",
        "data\processed\state\state.npy",
        "data\processed\state\state_row_indices.npy",
        "models\sales_predictor_train_only\best_model.pth",
        "models\sales_predictor\best_model.pth",
        "models\ppo\ppo_batched_economic_50_updates.pth",
        "models\ppo\ppo_batched_supported_8action_500_updates.pth"
    )
    foreach ($f in $artifacts) {
        if (Test-Path $f) {
            $item = Get-Item $f
            Write-Output ("FOUND`t{0}`t{1} bytes`tModified: {2}" -f $item.FullName, $item.Length, $item.LastWriteTime)
        } else { Write-Output "MISSING`t$f" }
    }
    Write-Output ""

    Write-Output "========== 5. RECENT REPORT DIRECTORIES =========="
    if (Test-Path ".\reports") {
        Get-ChildItem ".\reports" -Directory |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 15 Name, LastWriteTime |
            Format-Table -AutoSize
    }
    Write-Output ""
    Write-Output "========== AUDIT COMMAND FINISHED =========="
} *> $logPath

$bytes = (Get-Item $logPath).Length
Write-Host ""
Write-Host "Project snapshot audit finished."
Write-Host "Full output saved to: $logPath"
Write-Host ("Log size: {0:N1} KB" -f ($bytes / 1KB))
Write-Host ""
Write-Host "Please attach this file in ChatGPT or paste its contents."
