GPU=0
EPOCHS=300
DIR="./openpto/data/"
# # Prediction-focused learning
python rethink_exp/main_results.py --problem=portfolio --opt_model mse      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --data_dir ${DIR} --loadnew True

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=portfolio --opt_model dfl      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=portfolio --opt_model blackbox --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=portfolio --opt_model identity --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=portfolio --opt_model cpLayer  --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=portfolio --opt_model spo      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1 --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=portfolio --opt_model nce      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1 --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=portfolio --opt_model pointLTR --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1 --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=portfolio --opt_model listLTR  --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1 --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=portfolio --opt_model pairLTR  --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1 --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=portfolio --opt_model lodl     --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1 --method_path "openpto/config/models/lodl50k.yaml" --data_dir ${DIR}

