GPU=-1
EPOCHS=1

DIR="/mnt/nas/home/genghaoyu/OR/PTO/Rethink1.0/openpto/data/"
# # Prediction-focused learning
python rethink_exp/main_results.py --problem=knapsack --opt_model qptl --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU} --data_dir ${DIR}
