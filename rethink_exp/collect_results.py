import os

import numpy as np
import pandas as pd


def get_results(data_name, model_name, prefix_name):
    log_path = os.path.join(
        "saved_records", data_name, model_name, prefix_name, "log.txt"
    )
    if not os.path.exists(log_path):
        return np.zeros(4)
    with open(log_path, "r") as f:
        last_line = f.readlines()[-1].strip()
        result = last_line.split("  ")[-4:]
    try:
        result = [float(x) for x in result]
    except Exception:
        print(result)
        return np.zeros(4)
    return np.array(result)


data_names = ["knapsack-gen", "budgetalloc-real", "cubic-gen"]
prefix_names = ["default"]
model_names = [
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


# all_path_list.append(("blackbox", "ptr-ftn"))
# all_path_list.append(("identity", "ptr-ftn"))

pd.options.display.float_format = "{:.6f}".format
results = list()
for data_name in data_names:
    for prefix_name in prefix_names:
        results = list()
        for model_name in model_names:
            result = get_results(data_name, model_name, prefix_name)
            results.append(result)
        results = np.vstack(results)
        data_dict = dict()
        for idx in range(len(model_names)):
            data_dict[model_names[idx]] = results[idx, :]
        df = pd.DataFrame(data_dict)
        print(data_name, prefix_name)
        print(df)
        df.to_excel(
            os.path.join("saved_records", data_name, f"{prefix_name}-results.xlsx"),
            index=False,
            float_format="%.6f",
        )
