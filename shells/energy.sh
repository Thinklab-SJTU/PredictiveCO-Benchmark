GPU=0
EPOCHS=100

# Prediction-focused learning
python rethink_exp/main_results.py --problem=energy --opt_model mse      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=energy --opt_model blackbox --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=energy --opt_model identity --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=energy --opt_model spo      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=energy --opt_model nce      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=energy --opt_model pointLTR --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=energy --opt_model pairLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=energy --opt_model listLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=energy --opt_model lodl  --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}

# prediction + decision
python rethink_exp/main_results.py --problem=energy --opt_model blackbox --solver gurobi --n_ptr_epochs 20 --n_epochs 10 --gpu ${GPU}
python rethink_exp/main_results.py --problem=energy --opt_model identity --solver gurobi --n_ptr_epochs 20 --n_epochs 10 --gpu ${GPU}


#python rethink_exp/main_results.py --problem=energy --opt_model mse --solver gurobi --n_epochs 1 --gpu -1 --loadnew True


python rethink_exp/main_results.py --problem=energy --opt_model blackbox --solver gurobi --n_epochs 1 --gpu 0
python rethink_exp/main_results.py --problem=energy --opt_model pointLTR --solver gurobi --n_epochs 1 --gpu 0