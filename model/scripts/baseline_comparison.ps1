# Baseline comparison: 4 datasets x all methods
param(
    [string] = "berlin1_potsdamer_platz,berlin2_gendarmenmarkt,frankfurt1_maintower,frankfurt2_westendtower",
    [string] = "results/baseline"
)
 = "D:\1_developTool\4_conda\envs\smartLoc\python.exe"
Set-Location "D:\3_document\4_research\NLOS Signal Identification and Correction\model"
Write-Host "Running baseline comparison on: "
&  -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'module1_nlos'); sys.path.insert(0,'common'); sys.path.insert(0,'module4_experiments'); from baseline_runner import run_all_datasets; run_all_datasets(datasets=''.split(','), methods='all', output_dir='')"
Write-Host "Done. Results: "
