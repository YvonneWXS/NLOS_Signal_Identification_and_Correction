"""
run_serial.py -- Serial training script for 4 datasets
========================================================
Runs each dataset one at a time using existing checkpoints.
After each dataset finishes, generates analysis report.
"""
import subprocess
import os
import sys
import time

PYTHON = r"D:\1_developTool\4_conda\envs\smartLoc\python.exe"
SRC_DIR = r"D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\RadioGAT-Multi-band-Radiomap-Reconstruction"

EXPERIMENTS = [
    ("berlin1_potsdamer_platz", "exp_025"),
    ("berlin2_gendarmenmarkt", "exp_026"),
    ("frankfurt1_maintower", "exp_027"),
    ("frankfurt2_westendtower", "exp_028"),
]


def run_cmd(cmd, description=""):
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"CWD: {SRC_DIR}")
    print()
    
    start = time.time()
    result = subprocess.run(cmd, cwd=SRC_DIR)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"\nSUCCESS ({elapsed/60:.1f} min)")
    else:
        print(f"\nFAILED with code {result.returncode} ({elapsed/60:.1f} min)")
    
    return result.returncode


def main():
    print("=" * 60)
    print("Serial Training & Analysis Pipeline")
    print("=" * 60)
    print(f"Python: {PYTHON}")
    print(f"Source: {SRC_DIR}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print()
    
    total_start = time.time()
    completed = 0
    failed = []
    
    for dataset, exp_name in EXPERIMENTS:
        print(f"\n{'#'*60}")
        print(f"# Experiment: {exp_name} -- {dataset}")
        print(f"{'#'*60}")
        
        # Step 1: Training
        train_cmd = [
            PYTHON, "run_full_training.py",
            "--exp-name", exp_name,
            "--dataset", dataset,
        ]
        rc = run_cmd(train_cmd, f"Training {exp_name}")
        
        if rc != 0:
            print(f"\n*** WARNING: {exp_name} training failed, skipping analysis ***")
            failed.append(exp_name)
            continue
        
        completed += 1
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Pipeline Complete")
    print(f"  Completed: {completed}/{len(EXPERIMENTS)}")
    if failed:
        print(f"  Failed: {failed}")
    print(f"  Total time: {total_elapsed/3600:.1f} hours")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
