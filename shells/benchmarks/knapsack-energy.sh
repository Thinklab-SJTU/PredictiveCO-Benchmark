GPU=0
EPOCHS=300
PTH="./openpto/config/probs/knapsack-real.yaml"
PREFIX="bench"

# Prediction-focused learning
python rethink_exp/main_results.py --problem=knapsack --opt_model mse      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX}

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=knapsack --opt_model dfl      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX}
python rethink_exp/main_results.py --problem=knapsack --opt_model blackbox --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX}
python rethink_exp/main_results.py --problem=knapsack --opt_model identity --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX}
python rethink_exp/main_results.py --problem=knapsack --opt_model cpLayer  --solver cvxpy  --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX}
python rethink_exp/main_results.py --problem=knapsack --opt_model spo      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX}
python rethink_exp/main_results.py --problem=knapsack --opt_model nce      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX}
python rethink_exp/main_results.py --problem=knapsack --opt_model pointLTR --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX}
python rethink_exp/main_results.py --problem=knapsack --opt_model listLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX}
python rethink_exp/main_results.py --problem=knapsack --opt_model pairLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX}
python rethink_exp/main_results.py --problem=knapsack --opt_model lodl     --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX}



