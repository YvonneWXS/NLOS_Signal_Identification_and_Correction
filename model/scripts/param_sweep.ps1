# param_sweep.ps1 -- Batch script (ponytail: minimal)
param(
    [string] = "berlin1_potsdamer_platz",
    [string] = "results/sweep"
)
 = "python"
Write-Host "[param_sweep.ps1] Running on ..."
&  -m model.module4_experiments.run --dataset  --output 
Write-Host "[param_sweep.ps1] Done."
