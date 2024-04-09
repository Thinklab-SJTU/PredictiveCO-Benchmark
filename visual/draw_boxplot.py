import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def draw_boxplot(prob_name, prefix):
    model_names = [
        "mse",
        "dfl",
        "blackbox",
        "identity",
        "cpLayer",
        "spo",
        "nce",
        "pointLTR",
        "pairLTR",
        "listLTR",
        "lodl",
    ]
    results_list = []
    for model_name in model_names:
        # try:
        res = np.load(
            f"/mnt/nas/home/genghaoyu/OR/PTO/Rethink1.0/saved_records/{prob_name}/{model_name}/{prefix}/results.npy"
        )
        # size = res.shape
        # except:
        #     res = np.ones(size)
        #     res[1] = 0.1
        results_list.append(res)

    regret_list = np.array([res[1] / res[0] for res in results_list])

    regret_dict = {}
    formal_names = [
        "TwoStage",
        "DFL",
        "BB",
        "ID",
        "CPLayer",
        "SPO",
        "NCE",
        "Point",
        "Pair",
        "List",
        "LODL",
    ]
    for idx in range(len(model_names)):
        regret_dict[formal_names[idx]] = regret_list[idx]

    df = pd.DataFrame(regret_dict)
    plt.figure()
    ax = sns.boxplot(data=df, orient="h", palette="Set2")

    # 可选：旋转x轴标签
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    # plt.title("Boxplot with Standard Deviation")
    plt.yticks(size=16)
    plt.xticks(size=16)
    # save
    scatter_fig = ax.get_figure()
    fig_path = f"./visual/figs/{prob_name}_{prefix}.pdf"
    plt.tight_layout()
    plt.show()
    scatter_fig.savefig(fig_path, dpi=400)


draw_boxplot("knapsack-gen", "default")
draw_boxplot("knapsack-gen", "cap60")
draw_boxplot("knapsack-gen", "cap90")
draw_boxplot("knapsack-gen", "cap120")

draw_boxplot("knapsack-gen", "new-size40")
draw_boxplot("knapsack-gen", "new-size60")
draw_boxplot("knapsack-gen", "new-size80")
draw_boxplot("knapsack-gen", "new-size100")

draw_boxplot("knapsack-gen", "gen60")
draw_boxplot("knapsack-gen", "gen90")
draw_boxplot("knapsack-gen", "gen120")
