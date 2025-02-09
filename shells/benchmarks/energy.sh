GPU=0
EPOCHS=300
PREFIX="bench"

# Prediction-focused learning
python rethink_exp/main_results.py --problem=energy --opt_model mse      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX}

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=energy --opt_model dfl      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX}
python rethink_exp/main_results.py --problem=energy --opt_model blackbox --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX}
python rethink_exp/main_results.py --problem=energy --opt_model identity --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX}
python rethink_exp/main_results.py --problem=energy --opt_model spo      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX}
python rethink_exp/main_results.py --problem=energy --opt_model nce      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX}
python rethink_exp/main_results.py --problem=energy --opt_model pointLTR --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX}
python rethink_exp/main_results.py --problem=energy --opt_model pairLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX}
python rethink_exp/main_results.py --problem=energy --opt_model listLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX}
python rethink_exp/main_results.py --problem=energy --opt_model lodl     --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX}

