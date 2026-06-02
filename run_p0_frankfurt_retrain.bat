@echo off
REM ============================================================
REM P0 Frankfurt p_los/NLL fix ? Module 1 Retraining Script
REM ============================================================
REM Run this from: D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\model
REM Expected time: ~35 min per dataset (RTX 5060, AMP, block-diag bs=32)
REM ============================================================

set PYTHON=D:\1_developTool\4_conda\envs\smartLoc\python.exe
set WORKDIR=D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\model

cd /d "%WORKDIR%"

echo ============================================================
echo P0 Frankfurt Retraining
echo ============================================================
echo.
echo Changes applied (per-dataset overrides in config.py):
echo   - LAMBDA_ENTROPY: 0.03 -> 0.005  (less entropy reg, better p_los discrimination)
echo   - SIGMA_NLOS_CLAMP_LOG_MAX: 2.5 -> 3.5  (sigma_nlos max: 12.2 -> 33.1 km)
echo   - LAMBDA_SIGMA_REG: 0.01 -> 0.02  (stronger sigma regularization)
echo   - SIGMA_GAP_TARGET: 0.5 -> 1.0  (larger gap target)
echo.

echo [1/2] Training frankfurt1_maintower (exp_038)...
%PYTHON% run_full_training.py --exp-name exp_038 --dataset frankfurt1_maintower
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: frankfurt1 training failed!
    goto :error
)
echo frankfurt1 training complete.
echo.

echo [2/2] Training frankfurt2_westendtower (exp_039)...
%PYTHON% run_full_training.py --exp-name exp_039 --dataset frankfurt2_westendtower
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: frankfurt2 training failed!
    goto :error
)
echo frankfurt2 training complete.
echo.

echo ============================================================
echo P0 Frankfurt Retraining Complete!
echo ============================================================
echo Results: result\exp_038 and result\exp_039
echo.
echo Next: Run Module 2 evaluation with retrained models.
echo   cd D:\3_document\4_research\NLOS Signal Identification and Correction\model\part2_FactorGraphLocalizationFusion\model
echo   python run_fusion.py
goto :end

:error
echo Training failed. Check the console output above.
exit /b 1

:end
pause
