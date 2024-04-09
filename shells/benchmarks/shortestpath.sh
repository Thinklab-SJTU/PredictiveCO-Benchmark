GPU=0
EPOCHS=300
P_MODEL="cv_mlp"

# # Prediction-focused learning
python rethink_exp/main_results.py --problem=shortestpath --opt_model bce      --solver heuristic --pred_model ${P_MODEL} --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench"

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=shortestpath --opt_model dfl      --solver heuristic --pred_model ${P_MODEL} --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench"
python rethink_exp/main_results.py --problem=shortestpath --opt_model blackbox --solver heuristic --pred_model ${P_MODEL} --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench"
python rethink_exp/main_results.py --problem=shortestpath --opt_model identity --solver heuristic --pred_model ${P_MODEL} --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench"
python rethink_exp/main_results.py --problem=shortestpath --opt_model spo      --solver heuristic --pred_model ${P_MODEL} --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=shortestpath --opt_model nce      --solver heuristic --pred_model ${P_MODEL} --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=shortestpath --opt_model pointLTR --solver heuristic --pred_model ${P_MODEL} --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=shortestpath --opt_model listLTR  --solver heuristic --pred_model ${P_MODEL} --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=shortestpath --opt_model pairLTR  --solver heuristic --pred_model ${P_MODEL} --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=shortestpath --opt_model lodl     --solver heuristic --pred_model ${P_MODEL} --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --method_path "openpto/config/models/lodl50k.yaml"

# prediction + decision
python rethink_exp/main_results.py --problem=shortestpath --opt_model blackbox --solver heuristic --pred_model ${P_MODEL} --prefix "ptr-ftn" --n_ptr_epochs 150 --n_epochs 150 --gpu ${GPU} --prefix "bench"
python rethink_exp/main_results.py --problem=shortestpath --opt_model identity --solver heuristic --pred_model ${P_MODEL} --prefix "ptr-ftn" --n_ptr_epochs 150 --n_epochs 150 --gpu ${GPU} --prefix "bench"

