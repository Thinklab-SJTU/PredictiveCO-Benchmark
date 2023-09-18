# CONFIG_PATH=shells/configs/knapsack/knapsack-${EPOCHS}0item.yaml
CONFIG_PATH=""
EPOCHS=30

# Prediction-focused learning
python rethink_exp/main_results.py --problem=knapsack --opt_model mse      --solver gurobi --n_epochs ${EPOCHS} --gpu 0

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=knapsack --opt_model dfl      --solver gurobi --n_epochs ${EPOCHS} --gpu 0
python rethink_exp/main_results.py --problem=knapsack --opt_model blackbox --solver gurobi --n_epochs ${EPOCHS} --gpu 0
python rethink_exp/main_results.py --problem=knapsack --opt_model identity --solver gurobi --n_epochs ${EPOCHS} --gpu 0
python rethink_exp/main_results.py --problem=knapsack --opt_model spo      --solver gurobi --n_epochs ${EPOCHS} --gpu 0
python rethink_exp/main_results.py --problem=knapsack --opt_model nce      --solver gurobi --n_epochs ${EPOCHS} --gpu 0
python rethink_exp/main_results.py --problem=knapsack --opt_model pointLTR --solver gurobi --n_epochs ${EPOCHS} --gpu 0
python rethink_exp/main_results.py --problem=knapsack --opt_model pairLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu 0
python rethink_exp/main_results.py --problem=knapsack --opt_model listLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu 0
python rethink_exp/main_results.py --problem=knapsack --opt_model lodl     --solver gurobi --n_epochs ${EPOCHS} --gpu -1

# python rethink_exp/main_results.py --problem=knapsack --opt_model  --solver gurobi --n_epochs ${EPOCHS} --gpu 0

# prediction + decision
python rethink_exp/main_results.py --problem=knapsack --opt_model blackbox --solver gurobi --n_ptr_epochs 20 --n_epochs 10 --gpu 0
python rethink_exp/main_results.py --problem=knapsack --opt_model identity --solver gurobi --n_ptr_epochs 20 --n_epochs 10 --gpu 0

# # problem size:
# python rethink_exp/main_results.py --problem=knapsack --opt_model mse --solver gurobi --n_epochs 30 --gpu 0 \
#     --config_path ${CONFIG_PATH}
