GPU=0
EPOCHS=300

# # Prediction-focused learning
python rethink_exp/main_results.py --problem=knapsack --opt_model mse      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench"  --loadnew True

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=knapsack --opt_model dfl      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 10000
python rethink_exp/main_results.py --problem=knapsack --opt_model blackbox --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench"
python rethink_exp/main_results.py --problem=knapsack --opt_model identity --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench"
python rethink_exp/main_results.py --problem=knapsack --opt_model cpLayer  --solver cvxpy  --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench"
python rethink_exp/main_results.py --problem=knapsack --opt_model spo      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=knapsack --opt_model nce      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=knapsack --opt_model pointLTR --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=knapsack --opt_model listLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=knapsack --opt_model pairLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=knapsack --opt_model lodl     --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1 --method_path "openpto/config/models/lodl/quad5.yaml"

