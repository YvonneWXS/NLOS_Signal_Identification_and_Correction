# generate_report.ps1 -- Batch script (ponytail: minimal)
param(
    [string] = "berlin1_potsdamer_platz",
    [string] = "results/sweep"
)
 = "python"
Write-Host "[generate_report.ps1] Running on ..."
&  -m model.module4_experiments.run --dataset  --output 
Write-Host "[generate_report.ps1] Done."
