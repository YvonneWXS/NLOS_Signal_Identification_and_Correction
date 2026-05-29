$outputFile = "D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\_analysis_output.txt"
$pythonScript = "D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\_run_analysis.py"

& python $pythonScript *>&1 | Out-File -FilePath $outputFile -Encoding UTF8
Write-Host "Done. Output saved to $outputFile"