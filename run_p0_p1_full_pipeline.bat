@echo off
REM ============================================================
REM P0 + P1 Complete Pipeline Script
REM ============================================================
REM Run from: D:\3_document\4_research\NLOS Signal Identification and Correction
REM ============================================================

set PYTHON=D:\1_developTool\4_conda\envs\smartLoc\python.exe
set M1_DIR=D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\model
set M2_DIR=D:\3_document\4_research\NLOS Signal Identification and Correction\model\part2_FactorGraphLocalizationFusion\model

echo ============================================================
echo P0/P1 Complete Pipeline
echo ============================================================
echo.
echo Phase 1: P0 Frankfurt Retraining (Module 1)
echo   - LAMBDA_ENTROPY: 0.03 -> 0.005
echo   - SIGMA_NLOS_CLAMP_LOG_MAX: 2.5 -> 3.5
echo.

cd /d "%M1_DIR%"
echo [P0.1] Training frankfurt1_maintower (exp_038)...
%PYTHON% run_full_training.py --exp-name exp_038 --dataset frankfurt1_maintower
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: frankfurt1 training failed!
    goto :error
)

echo [P0.2] Training frankfurt2_westendtower (exp_039)...
%PYTHON% run_full_training.py --exp-name exp_039 --dataset frankfurt2_westendtower
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: frankfurt2 training failed!
    goto :error
)

echo.
echo Phase 2: P1 TCN Training (Module 2)
echo   - Build sequence caches + train TCN for all 4 datasets
echo.

cd /d "%M2_DIR%"
echo [P1.1] Training TCN models...
%PYTHON% fusion/train_tcn.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: TCN training failed, continuing...
)

echo.
echo Phase 3: Module 2 Evaluation
echo   - Run 6 methods (Standard LS, WLS-elev, WLS-MoG, Hard-threshold,
echo     FactorGraph-MoG, FactorGraph-MoG+2A)
echo.

echo [P1.2] Running full evaluation...
%PYTHON% run_fusion.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Evaluation failed, check logs...
)

echo.
echo ============================================================
echo Pipeline Complete!
echo ============================================================
echo Results:
echo   Module 1: result\exp_038, result\exp_039
echo   Module 2: result\exp_XXX (latest)
echo   TCN models: models\tcn_*.pth
echo.
goto :end

:error
echo Pipeline failed. Check the console output above.
pause
exit /b 1

:end
pause
