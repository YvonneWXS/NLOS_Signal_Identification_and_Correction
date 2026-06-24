@echo off
REM ============================================================
REM Frankfurt P0 Retraining Script (Module 2 v3)
REM ============================================================
REM Trains exp_038 (frankfurt1) and exp_039 (frankfurt2) with
REM dataset-specific config overrides for p_los/NLL fix.
REM
REM Estimated time: ~30-60 min per dataset on RTX 5060
REM ============================================================

set CONDA_ENV=smartLoc
set PYTHON=D:\1_developTool\4_conda\envs\%CONDA_ENV%\python.exe
set WORKDIR=D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\model

cd /d "%WORKDIR%"

echo ============================================================
echo PART 1/2: Training exp_038 (frankfurt1_maintower)
echo Overrides: LAMBDA_ENTROPY=0.005, SIGMA_NLOS_CLAMP_LOG_MAX=3.5,
echo            LAMBDA_SIGMA_REG=0.02, SIGMA_GAP_TARGET=1.0
echo ============================================================
%PYTHON% run_full_training.py --exp-name exp_038 --dataset frankfurt1_maintower
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] exp_038 training failed!
    pause
    exit /b 1
)
echo exp_038 training complete.

echo.
echo ============================================================
echo PART 2/2: Training exp_039 (frankfurt2_westendtower)
echo Overrides: LAMBDA_ENTROPY=0.005, SIGMA_NLOS_CLAMP_LOG_MAX=3.5,
echo            LAMBDA_SIGMA_REG=0.02, SIGMA_GAP_TARGET=1.0
echo ============================================================
%PYTHON% run_full_training.py --exp-name exp_039 --dataset frankfurt2_westendtower
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] exp_039 training failed!
    pause
    exit /b 1
)
echo exp_039 training complete.

echo.
echo ============================================================
echo Analyzing exp_038 and exp_039 with analyze_mog.py...
echo ============================================================
%PYTHON% analyze_mog.py --exp exp_038 --dataset frankfurt1_maintower
%PYTHON% analyze_mog.py --exp exp_039 --dataset frankfurt2_westendtower

echo.
echo ============================================================
echo Frankfurt P0 retraining complete!
echo Now run: python run_fusion.py in part2_FactorGraphLocalizationFusion/model/
echo ============================================================
pause
