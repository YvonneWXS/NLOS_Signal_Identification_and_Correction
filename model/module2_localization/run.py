# module2_localization/run.py — Module 2 CLI entry point
"""Module 2: Fusion Localization — run all methods on a dataset."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Import all method modules to register with factory
import module2_localization.standard_ls
import module2_localization.wls
import module2_localization.hard_threshold
import module2_localization.factor_graph
import module2_localization.cno_weighted
import module2_localization.snr_weighted
import module2_localization.raim
import module2_localization.irls
import module2_localization.kalman
import module2_localization.dnn
import module2_localization.gat_e2e
import module2_localization.ins_gnss

from module2_localization.factory import LocalizationFactory

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Module 2: Fusion Localization")
    parser.add_argument("--methods", type=str, default="all", help="Comma-separated method names or 'all'")
    parser.add_argument("--dataset", type=str, default="berlin1_potsdamer_platz")
    args = parser.parse_args()

    methods = LocalizationFactory.list_methods() if args.methods == "all" else args.methods.split(",")
    print(f"Module 2: {len(methods)} methods available: {methods}")
    print("Ready. Call LocalizationFactory.create(name).solve(obs, svp, ...) for each epoch.")
