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


def collect_ptr_ftn(data_name, prefix_name):
    if prefix_name == "default":
        ptr_ftn_prefix_name = "ptr-ftn"
    else:
        ptr_ftn_prefix_name = prefix_name + "-ptr-ftn"
    return collect_results(data_name, ptr_ftn_prefix_name, ["blackbox", "identity"])


global_data_names = [
    "knapsack-gen",
    "knapsack-energy",
    "energy-energy",
    "budgetalloc-real",
    "cubic-gen",
    "bipartitematching-cora",
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
        df_main = collect_results(data_name, "default", global_model_names)
        df_add = collect_ptr_ftn(data_name, "default")
        df = pd.concat((df_main, df_add), axis=1)
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


def collect_cap():
    cap_prefix_names = ["cap60", "cap90", "cap120"]
    for prefix_name in cap_prefix_names:
        df_main = collect_results("knapsack-gen", prefix_name, global_model_names)
        df_add = collect_ptr_ftn("knapsack-gen", prefix_name)
        df = pd.concat((df_main, df_add), axis=1)
        print("-" * 130)
        print("knapsack gen", prefix_name)
        print(df)
        df.to_excel(
            os.path.join(
                "saved_records",
                "knapsack-gen",
                f"knapsack-gen-{prefix_name}-results.xlsx",
            ),
            index=False,
            float_format="%.6f",
        )


def collect_size():
    cap_prefix_names = ["new-size40", "new-size60", "new-size80", "new-size100"]
    for prefix_name in cap_prefix_names:
        df_main = collect_results("knapsack-gen", prefix_name, global_model_names)
        df_add = collect_ptr_ftn("knapsack-gen", prefix_name)
        df = pd.concat((df_main, df_add), axis=1)
        print("-" * 130)
        print("knapsack gen", prefix_name)
        print(df)
        df.to_excel(
            os.path.join(
                "saved_records",
                "knapsack-gen",
                f"knapsack-gen-{prefix_name}-results.xlsx",
            ),
            index=False,
            float_format="%.6f",
        )


def collect_ad():
    prefix_name = "default"
    ad_model_names = [
        "bce",
        # "dfl",
        "blackbox",
        "identity",
        # "qptl",
        "spo",
    ]
    for data_name in ["advertising-real"]:
        df = collect_results(data_name, "default", ad_model_names)
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


def collect_gen():
    cap_prefix_names = ["gen60", "gen90", "gen120"]
    for prefix_name in cap_prefix_names:
        df = collect_results("knapsack-gen", prefix_name, global_model_names)
        print("-" * 130)
        print("knapsack gen", prefix_name)
        print(df)
        df.to_excel(
            os.path.join(
                "saved_records",
                "knapsack-gen",
                f"knapsack-gen-{prefix_name}-results.xlsx",
            ),
            index=False,
            float_format="%.6f",
        )


def collect_other_benchmarks():
    prefix_name = "linear"
    for data_name in ["cubic-gen"]:
        df_main = collect_results(data_name, prefix_name, global_model_names)
        df_add = collect_ptr_ftn(data_name, prefix_name)
        df = pd.concat((df_main, df_add), axis=1)
        print("-" * 130)
        print(data_name, prefix_name)
        print(df)
        df.to_excel(
            os.path.join(
                "saved_records",
                data_name,
                f"{data_name}-{prefix_name}-benchmark-results.xlsx",
            ),
            index=False,
            float_format="%.6f",
        )


def collect_time():
    prefix_name = "time"
    all_data_names = [
        "knapsack-gen",
        "knapsack-energy",
        "energy-energy",
        "budgetalloc-real",
        "cubic-gen",
        "portfolio-real",
        "bipartitematching-cora",
        # "advertise-real",
    ]
    for data_name in all_data_names:
        print("-" * 130)
        df = collect_results(data_name, prefix_name, global_model_names)
        print(data_name, prefix_name)
        print(df)
        df.to_excel(
            os.path.join("saved_records", data_name, f"{data_name}-time-results.xlsx"),
            index=False,
            float_format="%.6f",
        )


def collect_prefixs(prob_name, prefix_names, model_name, collect_name):
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
        f"{prob_name}-{collect_name}-results.xlsx",
    )
    df.to_excel(
        save_path,
        index=False,
        float_format="%.6f",
    )


if __name__ == "__main__":
    collect_benchmarks()
    collect_other_benchmarks()
    collect_cap()
    collect_size()
    collect_ad()
    collect_gen()
    collect_time()
