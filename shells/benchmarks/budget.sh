GPU=0
EPOCHS=300
DIR="./openpto/data/"

# # Prediction-focused learning
python rethink_exp/main_results.py --problem=budgetalloc --opt_model mse      --solver neural --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --data_dir ${DIR}

# ## Decisoin-focused learning
python rethink_exp/main_results.py --problem=budgetalloc --opt_model dfl      --solver neural --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --data_dir ${DIR} --batch_size 10000
python rethink_exp/main_results.py --problem=budgetalloc --opt_model blackbox --solver neural --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --data_dir ${DIR}
python rethink_exp/main_results.py --problem=budgetalloc --opt_model identity --solver neural --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --data_dir ${DIR}
python rethink_exp/main_results.py --problem=budgetalloc --opt_model spo      --solver neural --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --data_dir ${DIR}
python rethink_exp/main_results.py --problem=budgetalloc --opt_model nce      --solver neural --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --data_dir ${DIR} --batch_size 1
python rethink_exp/main_results.py --problem=budgetalloc --opt_model pointLTR --solver neural --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --data_dir ${DIR} --batch_size 1
python rethink_exp/main_results.py --problem=budgetalloc --opt_model listLTR  --solver neural --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --data_dir ${DIR} --batch_size 1
python rethink_exp/main_results.py --problem=budgetalloc --opt_model pairLTR  --solver neural --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --data_dir ${DIR} --batch_size 1
python rethink_exp/main_results.py --problem=budgetalloc --opt_model lodl     --solver neural --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --data_dir ${DIR} --batch_size 1 --method_path "openpto/config/models/lodl5k.yaml"

