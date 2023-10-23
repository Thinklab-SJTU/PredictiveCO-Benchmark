GPU=0
EPOCHS=300

# # Prediction-focused learning
python rethink_exp/main_results.py --problem=portfolio --opt_model mse      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU}

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=portfolio --opt_model dfl      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=portfolio --opt_model blackbox --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=portfolio --opt_model identity --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=portfolio --opt_model spo      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=portfolio --opt_model nce      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=portfolio --opt_model pointLTR --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=portfolio --opt_model listLTR  --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=portfolio --opt_model pairLTR  --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=portfolio --opt_model lodl     --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU}

# prediction + decision
python rethink_exp/main_results.py --problem=portfolio --opt_model blackbox --solver cvxpy --prefix "ptr-ftn" --n_ptr_epochs 150 --n_epochs 150 --gpu ${GPU}
python rethink_exp/main_results.py --problem=portfolio --opt_model identity --solver cvxpy --prefix "ptr-ftn" --n_ptr_epochs 150 --n_epochs 150 --gpu ${GPU}


