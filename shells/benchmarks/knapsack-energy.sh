GPU=0
EPOCHS=1
PTH="./openpto/config/probs/knapsack-real.yaml"
DIR="/mnt/nas/home/genghaoyu/OR/PTO/Rethink1.0/openpto/data/"
PREFIX="bench"

# Prediction-focused learning
python rethink_exp/main_results.py --problem=knapsack --opt_model mse      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX} --data_dir ${DIR} --loadnew True

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=knapsack --opt_model dfl      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX} --data_dir ${DIR}  --batch_size 10000
python rethink_exp/main_results.py --problem=knapsack --opt_model blackbox --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX} --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=knapsack --opt_model identity --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX} --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=knapsack --opt_model cpLayer  --solver cvxpy  --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX} --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=knapsack --opt_model spo      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX} --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=knapsack --opt_model nce      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX} --data_dir ${DIR}   --batch_size 1
python rethink_exp/main_results.py --problem=knapsack --opt_model pointLTR --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX} --data_dir ${DIR}   --batch_size 1
python rethink_exp/main_results.py --problem=knapsack --opt_model listLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX} --data_dir ${DIR}   --batch_size 1
python rethink_exp/main_results.py --problem=knapsack --opt_model pairLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX} --data_dir ${DIR}   --batch_size 1
python rethink_exp/main_results.py --problem=knapsack --opt_model lodl     --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --config_path ${PTH} --prefix ${PREFIX} --data_dir ${DIR}   --batch_size 1



