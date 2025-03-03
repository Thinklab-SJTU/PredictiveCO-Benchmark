import os

import numpy as np
import pandas as pd


def get_results(data_name, model_name, prefix_name):
    log_path = os.path.join(
        "saved_records", data_name, model_name, prefix_name, "log.txt"
    )
    if not os.path.exists(log_path):
        return np.zeros(4)
    # print(data_name, "-", model_name)
    with open(log_path, "r") as f:
        last_line = f.readlines()[-1].strip()
        result = last_line.split("  ")[-4:]
    try:
        result = [float(x) for x in result]
    except Exception:
        print(result)
        return np.zeros(4)
    return np.array(result)


def collect_results(data_name, prefix_name, model_names):
    pd.options.display.float_format = "{:.6f}".format
    results = list()
    for model_name in model_names:
        result = get_results(data_name, model_name, prefix_name)
        results.append(result)
    results = np.vstack(results)
    data_dict = dict()
    for idx in range(len(model_names)):
        data_dict[model_names[idx]] = results[idx, :]
    df = pd.DataFrame(data_dict)
    return df


global_data_names = [
    "knapsack-gen",
    "knapsack-energy",
    "energy-energy",
    "budgetalloc-real",
    "cubic-gen",
    "bipartitematching-cora",
    "shortestpath-warcraft" "TSP-gen",
]
global_model_names = [
    "mse",
    "dfl",
    "blackbox",
    "identity",
    "qptl",
    "spo",
    "nce",
    "pointLTR",
    "pairLTR",
    "listLTR",
    "lodl",
]


def collect_benchmarks():
    prefix_name = "default"
    for data_name in global_data_names:
        df = collect_results(data_name, "default", global_model_names)
        print("-" * 130)
        print(data_name, prefix_name)
        print(df)
        df.to_excel(
            os.path.join(
                "saved_records", data_name, f"{data_name}-benchmark-results.xlsx"
            ),
            index=False,
            float_format="%.6f",
        )


def collect_prefixs(prob_name, ood_name, prefix_names, model_name, collect_name):
    pd.options.display.float_format = "{:.6f}".format
    results = list()
    for prefix_name in prefix_names:
        result = get_results(prob_name, model_name, prefix_name)
        results.append(result)
    results = np.vstack(results)
    data_dict = dict()
    for idx in range(len(prefix_names)):
        data_dict[prefix_names[idx]] = results[idx, :]
    df = pd.DataFrame(data_dict)
    print(df)
    save_path = os.path.join(
        "saved_records",
        prob_name,
        f"{prob_name}-{model_name}-{ood_name}-{collect_name}-results.xlsx",
    )
    print("saving to ", save_path)
    df.to_excel(
        save_path,
        index=False,
        float_format="%.6f",
    )


if __name__ == "__main__":
    collect_benchmarks()
