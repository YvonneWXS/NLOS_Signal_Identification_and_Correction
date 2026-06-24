# -*- coding: utf-8 -*-
"""
Module 1: NLOS Perception & Error Modeling — CLI entry point
============================================================
Usage:
  python -m module1_nlos.run --dataset berlin1_potsdamer_platz --mode train
  python -m module1_nlos.run --dataset berlin1_potsdamer_platz --mode inference --checkpoint path/to/best_model.pth
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from Data_read import load_and_process_dataset, print_data_statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 1: NLOS GAT training/inference")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name (e.g., exp_001)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for inference")
    parser.add_argument("--epochs", type=int, default=None, help="Override NUM_EPOCHS")
    args = parser.parse_args()

    config = get_config(DATASETS=[args.dataset])
    config.ensure_dirs()

    print("=" * 60)
    print("Step 1: Processing dataset...")
    print("=" * 60)

    t0 = time.time()
    epochs = load_and_process_dataset(args.dataset, config)
    elapsed = time.time() - t0
    print_data_statistics(epochs, args.dataset)
    print(f"  Processing time: {elapsed:.1f}s")

    if args.mode == "train":
        print("\n" + "=" * 60)
        print(f"Step 2: Starting {config.NUM_EPOCHS}-epoch training")
        print("=" * 60)
        from GAT_V2026 import main
        main(dataset_name=args.dataset, exp_name=args.exp_name, num_epochs=args.epochs)
    elif args.mode == "inference":
        print("\n" + "=" * 60)
        print("Step 2: Running inference")
        print("=" * 60)
        from GAT_V2026 import main
        if args.checkpoint:
            main(dataset_name=args.dataset, exp_name=args.exp_name, resume_from=args.checkpoint, num_epochs=0)
        else:
            print("ERROR: --checkpoint required for inference mode")
            sys.exit(1)
