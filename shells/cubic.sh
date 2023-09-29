EPOCHS=100
GPU=0

# Prediction-focused learning
python rethink_exp/main_results.py --problem=cubic --opt_model mse      --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=cubic --opt_model dfl      --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2
python rethink_exp/main_results.py --problem=cubic --opt_model blackbox --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2
python rethink_exp/main_results.py --problem=cubic --opt_model identity --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2
python rethink_exp/main_results.py --problem=cubic --opt_model spo      --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2
python rethink_exp/main_results.py --problem=cubic --opt_model nce      --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2
python rethink_exp/main_results.py --problem=cubic --opt_model pointLTR --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2
python rethink_exp/main_results.py --problem=cubic --opt_model listLTR  --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2
python rethink_exp/main_results.py --problem=cubic --opt_model pairLTR  --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2
python rethink_exp/main_results.py --problem=cubic --opt_model lodl     --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2
# python rethink_exp/main_results.py --problem=cubic --opt_model perturb  --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2

# prediction + decision
python rethink_exp/main_results.py --problem=cubic --opt_model blackbox --solver heuristic --prefix "ptr-ftn" --n_ptr_epochs 50 --n_epochs 50 --gpu ${GPU} --lr 5e-2
python rethink_exp/main_results.py --problem=cubic --opt_model identity --solver heuristic --prefix "ptr-ftn" --n_ptr_epochs 50 --n_epochs 50 --gpu ${GPU} --lr 5e-2

