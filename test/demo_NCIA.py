"""Revised main experiment: clustering-based unsupervised defect prediction.

Run from the repository root:

    python test/demo_NCIA.py --check
    python test/demo_NCIA.py --smoke
    python test/demo_NCIA.py

Clustering uses the transformed out-of-bag test features. Labels use the
original metrics and labelCluster_v4_raw. Finished CSV files are skipped,
so the same command can be resumed.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utilities.CrossValidataion import out_of_sample_bootstrap
from utilities.File import create_dir, save_results
from utilities.PerformanceMeasure import get_measure
from utilities.RankMeasure import rank_measure
from utilities.labelraw_run import (
    ALL_CLF,
    ALL_NORMS,
    ALL_PROJECTS,
    DATA_DIR,
    N_CLUSTERS,
    apply_transform,
    count_finished_rounds,
    label_predictions,
    load_project,
    missing_datasets,
    parse_names,
    run_cluster,
)

SAVE_ROOT = os.path.join(ROOT, "result", "test_labelRaw")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the revised performance experiment.")
    parser.add_argument("--check", action="store_true", help="Verify the 22 datasets and exit.")
    parser.add_argument("--smoke", action="store_true", help="Short check: Kmeans, original data, EQ, 2 runs.")
    parser.add_argument("--clf", default="", help="Comma-separated clusterers. Default: all 23.")
    parser.add_argument("--rep", type=int, default=100, help="Bootstrap repetitions. Default: 100.")
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
        repeats = 2
    else:
        clusterers = parse_names(args.clf, ALL_CLF)
        projects = ALL_PROJECTS
        norms = ALL_NORMS
        repeats = args.rep

    print("protocol = cluster on transformed features; label on original metrics (labelCluster_v4_raw)")
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

                data, bugs, locs = load_project(project_name)
                if done > 0:
                    print(
                        f"[resume] {model_name} {normalization_model} "
                        f"{project_name} from {done + 1}/{repeats}"
                    )

                for loop in range(done, repeats):
                    print(f"{model_name}-> {normalization_model} {project_name} {loop + 1}/{repeats}")
                    train_data, _train_label, test_data, test_label, _train_idx, test_idx = (
                        out_of_sample_bootstrap(data, loop)
                    )
                    loc = locs[test_idx]
                    bug = bugs[test_idx]
                    test_orig = np.asarray(test_data, dtype=float)
                    _train_t, test_cluster = apply_transform(normalization_model, train_data, test_data)

                    start = time.perf_counter()
                    predict_y = run_cluster(model_name, test_cluster, N_CLUSTERS, loop)
                    predict_y = label_predictions(test_orig, predict_y)
                    elapsed = time.perf_counter() - start

                    if not isinstance(bug, np.ndarray):
                        bug = bug.to_numpy().flatten()
                    test_label = np.asarray(test_label).flatten()
                    precision, recall, pf, f_measure, auc, g_measure, g_mean, bal, mcc = get_measure(
                        test_label, predict_y
                    )
                    popt, erecall, eprecision, efmeasure, pmi, ifa = rank_measure(predict_y, loc, test_label)
                    save_results(
                        folder + project_name,
                        [
                            precision, recall, pf, f_measure, auc, g_measure, g_mean, bal, mcc,
                            popt, erecall, eprecision, efmeasure, pmi, ifa, elapsed,
                        ],
                    )

    print("done")
    print("Results under:", SAVE_ROOT)


if __name__ == "__main__":
    main()
