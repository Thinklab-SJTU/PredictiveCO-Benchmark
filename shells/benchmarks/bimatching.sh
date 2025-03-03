GPU=0
EPOCHS=300
DIR="./openpto/data/"

# Prediction-focused learning
python rethink_exp/main_results.py --problem=bipartitematching --opt_model bce      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --prefix "bench" --data_dir ${DIR}  --loadnew True

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=bipartitematching --opt_model dfl      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --prefix "bench" --data_dir ${DIR}
python rethink_exp/main_results.py --problem=bipartitematching --opt_model blackbox --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --prefix "bench" --data_dir ${DIR}
python rethink_exp/main_results.py --problem=bipartitematching --opt_model identity --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --prefix "bench" --data_dir ${DIR}
python rethink_exp/main_results.py --problem=bipartitematching --opt_model cpLayer  --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --prefix "bench" --data_dir ${DIR}
python rethink_exp/main_results.py --problem=bipartitematching --opt_model spo      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --prefix "bench" --batch_size 1 --data_dir ${DIR}
python rethink_exp/main_results.py --problem=bipartitematching --opt_model nce      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --prefix "bench" --batch_size 1 --data_dir ${DIR}
python rethink_exp/main_results.py --problem=bipartitematching --opt_model pointLTR --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --prefix "bench" --batch_size 1 --data_dir ${DIR}
python rethink_exp/main_results.py --problem=bipartitematching --opt_model listLTR  --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --prefix "bench" --batch_size 1 --data_dir ${DIR}
python rethink_exp/main_results.py --problem=bipartitematching --opt_model pairLTR  --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --prefix "bench" --batch_size 1 --data_dir ${DIR}
python rethink_exp/main_results.py --problem=bipartitematching --opt_model lodl     --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --prefix "bench" --batch_size 1 --data_dir ${DIR} --method_path "openpto/config/models/lodl/lodl5k.yaml" 


