# scripts/run_all_experiments.ps1 — One-click full experiment pipeline
param(
    [string]$Dataset = "berlin1_potsdamer_platz",
    [string]$OutputDir = "results/run_all"
)
$Python = "D:\1_developTool\4_conda\envs\smartLoc\python.exe"
$Root = "D:\3_document\4_research\NLOS Signal Identification and Correction\model"

Write-Host "Running full pipeline for $Dataset..."
Write-Host "Output: $OutputDir"

# M1: Inference
Write-Host "[1/4] Module 1 inference..."
& $Python -m module1_nlos.run --dataset $Dataset --mode train --output "$OutputDir/module1"

Write-Host "Pipeline complete."
