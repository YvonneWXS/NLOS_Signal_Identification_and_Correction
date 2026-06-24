@echo off
REM ============================================================
REM Module 2 v5 ? Full Training Pipeline
REM ============================================================
REM Trains exp_040-043 (Module 1 with Supervised Mu Regression)
REM Then runs full 12-method evaluation.
REM
REM Estimated time: ~4-6 hours total (4 x ~60 min training + ~30 min eval)
REM ============================================================

set PYTHON=D:\1_developTool\4_conda\envs\smartLoc\python.exe
set M1DIR=D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\model
set M2DIR=D:\3_document\4_research\NLOS Signal Identification and Correction\model\part2_FactorGraphLocalizationFusion\model

echo ============================================================
echo STEP 1: Train Module 1 with Supervised Mu Regression (4 datasets)
echo ============================================================

cd /d "%M1DIR%"

echo [1/4] berlin1_potsdamer_platz -> exp_040
%PYTHON% run_full_training.py --exp-name exp_040 --dataset berlin1_potsdamer_platz
if %ERRORLEVEL% NEQ 0 (echo FAILED & pause & exit /b 1)

echo [2/4] berlin2_gendarmenmarkt -> exp_041
%PYTHON% run_full_training.py --exp-name exp_041 --dataset berlin2_gendarmenmarkt
if %ERRORLEVEL% NEQ 0 (echo FAILED & pause & exit /b 1)

echo [3/4] frankfurt1_maintower -> exp_042
%PYTHON% run_full_training.py --exp-name exp_042 --dataset frankfurt1_maintower
if %ERRORLEVEL% NEQ 0 (echo FAILED & pause & exit /b 1)

echo [4/4] frankfurt2_westendtower -> exp_043
%PYTHON% run_full_training.py --exp-name exp_043 --dataset frankfurt2_westendtower
if %ERRORLEVEL% NEQ 0 (echo FAILED & pause & exit /b 1)

echo.
echo ============================================================
echo STEP 2: Analyze all 4 models
echo ============================================================
%PYTHON% analyze_mog.py --exp exp_040 --dataset berlin1_potsdamer_platz --batch-size 64
%PYTHON% analyze_mog.py --exp exp_041 --dataset berlin2_gendarmenmarkt --batch-size 64
%PYTHON% analyze_mog.py --exp exp_042 --dataset frankfurt1_maintower --batch-size 64
%PYTHON% analyze_mog.py --exp exp_043 --dataset frankfurt2_westendtower --batch-size 64

echo.
echo ============================================================
echo STEP 3: Rebuild MoG inference caches
echo ============================================================
del /Q "D:\3_document\4_research\NLOS Signal Identification and Correction\model\part2_FactorGraphLocalizationFusion\cache\*_mog_outputs.pkl" 2>nul

echo.
echo ============================================================
echo STEP 4: Run full 12-method evaluation
echo ============================================================
cd /d "%M2DIR%"
%PYTHON% run_fusion.py

echo.
echo ============================================================
echo v5 Training Pipeline Complete!
echo ============================================================
pause
