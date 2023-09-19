# CONFIG_PATH=shells/configs/energy/energy-${EPOCHS}0item.yaml
CONFIG_PATH=""
EPOCHS=30

# Prediction-focused learning
python rethink_exp/main_results.py --problem=energy --opt_model mse      --solver gurobi --n_epochs ${EPOCHS} --gpu -1

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=energy --opt_model blackbox --solver gurobi --n_epochs ${EPOCHS} --gpu -1
python rethink_exp/main_results.py --problem=energy --opt_model identity --solver gurobi --n_epochs ${EPOCHS} --gpu -1
python rethink_exp/main_results.py --problem=energy --opt_model spo      --solver gurobi --n_epochs ${EPOCHS} --gpu -1
python rethink_exp/main_results.py --problem=energy --opt_model nce      --solver gurobi --n_epochs ${EPOCHS} --gpu -1
python rethink_exp/main_results.py --problem=energy --opt_model pointLTR --solver gurobi --n_epochs ${EPOCHS} --gpu -1
python rethink_exp/main_results.py --problem=energy --opt_model pairLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu -1
python rethink_exp/main_results.py --problem=energy --opt_model listLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu -1
python rethink_exp/main_results.py --problem=energy --opt_model lodl     --solver gurobi --n_epochs ${EPOCHS} --gpu -1

# python rethink_exp/main_results.py --problem=energy --opt_model  --solver gurobi --n_epochs ${EPOCHS} --gpu -1

# prediction + decision
python rethink_exp/main_results.py --problem=energy --opt_model blackbox --solver gurobi --n_ptr_epochs 20 --n_epochs 10 --gpu -1
python rethink_exp/main_results.py --problem=energy --opt_model identity --solver gurobi --n_ptr_epochs 20 --n_epochs 10 --gpu -1

# # problem size:
# python rethink_exp/main_results.py --problem=energy --opt_model mse --solver gurobi --n_epochs 30 --gpu -1 \
#     --config_path ${CONFIG_PATH}
