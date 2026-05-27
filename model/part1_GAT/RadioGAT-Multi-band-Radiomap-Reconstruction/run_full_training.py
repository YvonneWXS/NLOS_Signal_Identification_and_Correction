"""
Full training script — processes dataset and runs training epochs
Usage:
  Single:  python run_full_training.py --dataset berlin1_potsdamer_platz
  Parallel (4 terminals):
    python run_full_training.py --dataset berlin1_potsdamer_platz
    python run_full_training.py --dataset berlin2_gendarmenmarkt
    python run_full_training.py --dataset frankfurt1_maintower
    python run_full_training.py --dataset frankfurt2_westendtower
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from Data_read import load_and_process_dataset, print_data_statistics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (e.g., exp_001)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., berlin1_potsdamer_platz)')
    args = parser.parse_args()

    config = get_config(DATASETS=[args.dataset], USE_TENSORBOARD=False)
    config.ensure_dirs()

    print("=" * 60)
    print("Step 1: Processing all datasets...")
    print("=" * 60)

    for ds_name in config.DATASETS:
        t0 = time.time()
        epochs = load_and_process_dataset(ds_name, config)
        elapsed = time.time() - t0
        print_data_statistics(epochs, ds_name)
        print(f"  Processing time: {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print(f"Step 2: Starting {config.NUM_EPOCHS}-epoch training")
    print("=" * 60)

    from GAT_V2025 import main
    main(dataset_name=args.dataset, exp_name=args.exp_name)
