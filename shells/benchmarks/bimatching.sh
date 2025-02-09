GPU=-1
EPOCHS=300

# Prediction-focused learning
python rethink_exp/main_results.py --problem=bipartitematching --opt_model bce       --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6  --prefix "bench"

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=bipartitematching --opt_model dfl      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --lr 0.01 --n_layers 1 --n_hidden 32 --prefix "bench"
python rethink_exp/main_results.py --problem=bipartitematching --opt_model blackbox --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --lr 0.01 --n_layers 1 --n_hidden 32 --prefix "bench"
python rethink_exp/main_results.py --problem=bipartitematching --opt_model identity --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --lr 0.01 --n_layers 1 --n_hidden 32 --prefix "bench"
python rethink_exp/main_results.py --problem=bipartitematching --opt_model cpLayer  --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --lr 0.01 --n_layers 1 --n_hidden 32 --prefix "bench"
python rethink_exp/main_results.py --problem=bipartitematching --opt_model spo      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --lr 0.01 --n_layers 1 --n_hidden 32 --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=bipartitematching --opt_model nce      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --lr 0.01 --n_layers 1 --n_hidden 32 --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=bipartitematching --opt_model pointLTR --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --lr 0.01 --n_layers 1 --n_hidden 32 --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=bipartitematching --opt_model listLTR  --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --lr 0.01 --n_layers 1 --n_hidden 32 --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=bipartitematching --opt_model pairLTR  --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --lr 0.01 --n_layers 1 --n_hidden 32 --prefix "bench" --batch_size 1
python rethink_exp/main_results.py --problem=bipartitematching --opt_model lodl     --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --lr 0.01 --n_layers 1 --n_hidden 32 --prefix "bench" --batch_size 1 --method_path "openpto/config/models/lodl5k.yaml"


