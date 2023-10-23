GPU=0
EPOCHS=300

# # Prediction-focused learning
python rethink_exp/main_results.py --problem=budgetalloc --opt_model mse      --solver neural --n_epochs ${EPOCHS} --gpu ${GPU}

# ## Decisoin-focused learning
python rethink_exp/main_results.py --problem=budgetalloc --opt_model dfl      --solver neural --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=budgetalloc --opt_model blackbox --solver neural --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=budgetalloc --opt_model identity --solver neural --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=budgetalloc --opt_model spo      --solver neural --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=budgetalloc --opt_model pointLTR --solver neural --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=budgetalloc --opt_model listLTR  --solver neural --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=budgetalloc --opt_model pairLTR  --solver neural --n_epochs ${EPOCHS} --gpu ${GPU}
python rethink_exp/main_results.py --problem=budgetalloc --opt_model lodl     --solver neural --n_epochs ${EPOCHS} --gpu ${GPU}

# prediction + decision
python rethink_exp/main_results.py --problem=budgetalloc --opt_model blackbox --solver nerual --prefix "ptr-ftn" --n_ptr_epochs 150 --n_epochs 150 --gpu ${GPU}
python rethink_exp/main_results.py --problem=budgetalloc --opt_model identity --solver neural --prefix "ptr-ftn" --n_ptr_epochs 150 --n_epochs 150 --gpu ${GPU}

