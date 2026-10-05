import os
import pickle

import pandas as pd


def create_dir(dirname):
    path = dirname
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    except OSError as err:
        print(err)
    return path + "/"


def save_results(save_path, score):
    temp_res = pd.DataFrame(score).T
    temp_res.to_csv(save_path + ".csv", index=False, header=False, mode="a")


def save_results_pickle(save_path, results):
    with open(save_path + ".pkl", "ab") as f:
        pickle.dump(results, f)


def load_results_pickle(save_path):
    results = []
    with open(save_path + ".pkl", "rb") as f:
        result = pickle.load(f)
        if type(result) == list:
            results = result
        else:
            results.append(result)
            while True:
                try:
                    result = pickle.load(f)
                    results.append(result)
                except EOFError:
                    break
    return results
