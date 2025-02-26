GPU=0
PTR_EPS=1
EPOCHS=1
DIR="/mnt/nas/home/genghaoyu/OR/PTO/Rethink1.0/openpto/data/"

# Prediction-focused learning
python rethink_exp/main_results.py --problem=advertising --pred_model cvr --opt_model bce      --solver ortools --n_ptr_epochs ${PTR_EPS} --n_epochs ${EPOCHS} --gpu ${GPU} --prefix "bench" --batch_size 1 --data_dir ${DIR} --loadnew True

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=advertising --pred_model cvr --opt_model dfl      --solver ortools --n_ptr_epochs ${PTR_EPS} --n_epochs ${EPOCHS} --gpu ${GPU}  --prefix "bench" --batch_size 1 --data_dir ${DIR}
python rethink_exp/main_results.py --problem=advertising --pred_model cvr --opt_model identity --solver ortools --n_ptr_epochs ${PTR_EPS} --n_epochs ${EPOCHS} --gpu ${GPU}  --prefix "bench" --batch_size 1 --data_dir ${DIR}
python rethink_exp/main_results.py --problem=advertising --pred_model cvr --opt_model blackbox --solver ortools --n_ptr_epochs ${PTR_EPS} --n_epochs ${EPOCHS} --gpu ${GPU}  --prefix "bench" --batch_size 1 --data_dir ${DIR}

