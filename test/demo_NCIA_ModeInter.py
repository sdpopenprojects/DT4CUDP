"""Revised permutation-importance experiment.

Run from the repository root:

    python test/demo_NCIA_ModeInter.py --check
    python test/demo_NCIA_ModeInter.py --smoke
    python test/demo_NCIA_ModeInter.py

Five clusterers, six transformations, 22 projects, 50 repetitions.
Each repetition refits the clusterer after every feature shuffle.
"""
import argparse
import os
import sys
import warnings

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utilities.CrossValidataion import out_of_sample_bootstrap
from utilities.File import create_dir, save_results
from utilities.ModeInter import G2PC
from utilities.labelraw_run import (
    ALL_NORMS,
    ALL_PROJECTS,
    DATA_DIR,
    INTERPRET_CLF,
    N_CLUSTERS,
    apply_transform,
    count_finished_rounds,
    label_predictions,
    load_project,
    missing_datasets,
    parse_names,
    run_cluster,
)

SAVE_ROOT = os.path.join(ROOT, "result", "test_ModeInter_labelRaw")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the revised interpretation experiment.")
    parser.add_argument("--check", action="store_true", help="Verify the 22 datasets and exit.")
    parser.add_argument("--smoke", action="store_true", help="Short check: Kmeans, original data, EQ, 1 run.")
    parser.add_argument("--clf", default="", help="Comma-separated clusterers. Default: Kmeans,Kmedoids,Gmeans,AP,SC.")
    parser.add_argument("--rep", type=int, default=50, help="Bootstrap repetitions. Default: 50.")
    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    missing = missing_datasets()
    if missing:
        raise SystemExit("Missing datasets:\n  " + "\n  ".join(missing))
    print(f"datasets = {len(ALL_PROJECTS)} under {DATA_DIR}")
    if args.check:
        print("dataset check passed")
        return

    if args.smoke:
        clusterers = ["Kmeans"]
        projects = ["EQ"]
        norms = ["O"]
        repeats = 1
    else:
        clusterers = parse_names(args.clf, INTERPRET_CLF)
        projects = ALL_PROJECTS
        norms = ALL_NORMS
        repeats = args.rep

    print("protocol = permutation importance; refit after each shuffle; labelCluster_v4_raw")
    print("save_path =", SAVE_ROOT)
    print("clusterers =", clusterers)
    print("transformations =", norms)
    print("projects =", len(projects))
    print("Rep =", repeats)

    for model_name in clusterers:
        for normalization_model in norms:
            for project_name in projects:
                folder = create_dir(os.path.join(SAVE_ROOT, model_name, normalization_model))
                csv_path = folder + project_name + ".csv"
                done = count_finished_rounds(csv_path)
                if done >= repeats:
                    print(f"[skip] {model_name} {normalization_model} {project_name} {done}/{repeats}")
                    continue

                data, _bugs, _locs = load_project(project_name)
                if done > 0:
                    print(
                        f"[resume] {model_name} {normalization_model} "
                        f"{project_name} from {done + 1}/{repeats}"
                    )

                for loop in range(done, repeats):
                    print(f"{model_name}-> {normalization_model} {project_name} {loop + 1}/{repeats}")
                    _train, _train_label, test_data, test_label, _train_idx, _test_idx = (
                        out_of_sample_bootstrap(data, loop)
                    )
                    test_orig = np.asarray(test_data, dtype=float)
                    _train_t, test_cluster = apply_transform(normalization_model, _train, test_data)
                    predict_y = run_cluster(model_name, test_cluster, N_CLUSTERS, loop)
                    predict_y = label_predictions(test_orig, predict_y)
                    test_label = np.asarray(test_label).flatten()
                    row_means = G2PC(
                        predict_y, test_cluster, test_orig, test_label, model_name, N_CLUSTERS, loop
                    )
                    save_results(folder + project_name, np.asarray(row_means).tolist())

    print("done")
    print("Results under:", SAVE_ROOT)


if __name__ == "__main__":
    main()
