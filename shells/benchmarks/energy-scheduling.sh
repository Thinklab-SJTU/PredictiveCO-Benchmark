GPU=0
EPOCHS=300
PREFIX="bench"
DIR="./openpto/data/"

# Prediction-focused learning
python rethink_exp/main_results.py --problem=energy --opt_model mse      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX} --data_dir ${DIR} --loadnew True

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=energy --opt_model dfl      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX} --data_dir ${DIR} --batch_size 10000 
python rethink_exp/main_results.py --problem=energy --opt_model blackbox --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX} --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=energy --opt_model identity --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX} --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=energy --opt_model spo      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX} --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=energy --opt_model nce      --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX} --batch_size 1 --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=energy --opt_model pointLTR --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX} --batch_size 1 --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=energy --opt_model pairLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX} --batch_size 1 --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=energy --opt_model listLTR  --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX} --batch_size 1 --data_dir ${DIR} 
python rethink_exp/main_results.py --problem=energy --opt_model lodl     --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --prefix ${PREFIX} --batch_size 1 --data_dir ${DIR} --method_path "openpto/config/models/lodl/lodl5k.yaml"
