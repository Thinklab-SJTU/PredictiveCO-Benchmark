GPU=0
EPOCHS=300

# Prediction-focused learning
python rethink_exp/main_results.py --problem=energy --opt_model mse      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --data_dir ${DIR}

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=energy --opt_model blackbox --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=energy --opt_model identity --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=energy --opt_model spo      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=energy --opt_model nce      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=energy --opt_model pointLTR --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=energy --opt_model pairLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=energy --opt_model listLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=energy --opt_model lodl     --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}

# prediction + decision
python rethink_exp/main_results.py --problem=energy --opt_model blackbox --solver gurobi --n_ptr_epochs 150 --n_epochs 150 --gpu ${GPU}
python rethink_exp/main_results.py --problem=energy --opt_model identity --solver gurobi --n_ptr_epochs 150 --n_epochs 150 --gpu ${GPU}

