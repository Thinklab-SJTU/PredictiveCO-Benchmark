# CONFIG_PATH=shells/configs/knapsack/knapsack-${EPOCHS}0item.yaml
CONFIG_PATH=""
EPOCHS=100
GPU=0

# Prediction-focused learning
python rethink_exp/main_results.py --problem=cubic --opt_model mse      --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU}

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=cubic --opt_model blackbox --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=cubic --opt_model identity --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=cubic --opt_model spo      --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=cubic --opt_model nce      --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=cubic --opt_model pointLTR --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=cubic --opt_model pairLTR  --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=cubic --opt_model listLTR  --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=cubic --opt_model lodl     --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU}
