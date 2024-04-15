GPU=0
EPOCHS=300

# # Prediction-focused learning
python rethink_exp/main_results.py --problem=TSP --opt_model mse      --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench"

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=TSP --opt_model dfl      --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench"
python rethink_exp/main_results.py --problem=TSP --opt_model blackbox --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench"
python rethink_exp/main_results.py --problem=TSP --opt_model identity --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench"
python rethink_exp/main_results.py --problem=TSP --opt_model spo      --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench"
# python rethink_exp/main_results.py --problem=TSP --opt_model nce      --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench"
# python rethink_exp/main_results.py --problem=TSP --opt_model pointLTR --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench"
# python rethink_exp/main_results.py --problem=TSP --opt_model listLTR  --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench"
# python rethink_exp/main_results.py --problem=TSP --opt_model pairLTR  --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench"
# python rethink_exp/main_results.py --problem=TSP --opt_model lodl     --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --method_path "openpto/config/models/lodl50k.yaml"

# prediction + decision
# python rethink_exp/main_results.py --problem=TSP --opt_model blackbox --solver heuristic --prefix "ptr-ftn" --n_ptr_epochs 150 --n_epochs 150 --gpu ${GPU} --prefix "bench"
# python rethink_exp/main_results.py --problem=TSP --opt_model identity --solver heuristic --prefix "ptr-ftn" --n_ptr_epochs 150 --n_epochs 150 --gpu ${GPU} --prefix "bench"

