GPU=0
EPOCHS=300

# Prediction-focused learning
python rethink_exp/main_results.py --problem=cubic --opt_model mse      --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2 --instances 250 --testinstances 400  --prefix "bench"  --loadnew True

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=cubic --opt_model dfl      --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2 --instances 250 --testinstances 400 --prefix "bench"
python rethink_exp/main_results.py --problem=cubic --opt_model blackbox --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2 --instances 250 --testinstances 400 --prefix "bench"
python rethink_exp/main_results.py --problem=cubic --opt_model identity --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2 --instances 250 --testinstances 400 --prefix "bench"
python rethink_exp/main_results.py --problem=cubic --opt_model spo      --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2 --instances 250 --testinstances 400 --prefix "bench"
python rethink_exp/main_results.py --problem=cubic --opt_model nce      --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2 --instances 250 --testinstances 400 --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=cubic --opt_model pointLTR --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2 --instances 250 --testinstances 400 --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=cubic --opt_model listLTR  --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2 --instances 250 --testinstances 400 --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=cubic --opt_model pairLTR  --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2 --instances 250 --testinstances 400 --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=cubic --opt_model lodl     --solver heuristic --n_epochs ${EPOCHS} --gpu ${GPU} --lr 5e-2 --instances 250 --testinstances 400 --prefix "bench" --method_path "openpto/config/models/lodl/lodl5k.yaml"
