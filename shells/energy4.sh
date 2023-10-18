DIR="/mnt/nas/home/genghaoyu/OR/PTO/Rethink1.0/openpto/data/"
GPU=0
EPOCHS=100

python rethink_exp/main_results.py --problem=energy --opt_model lodl  --solver gurobi --n_epochs ${EPOCHS} --gpu ${GPU}  --data_dir ${DIR}
