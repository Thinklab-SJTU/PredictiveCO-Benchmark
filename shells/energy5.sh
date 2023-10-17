DIR="/mnt/nas/home/genghaoyu/OR/PTO/Rethink1.0/openpto/data/"
GPU=0
EPOCHS=100

python rethink_exp/main_results.py --problem=energy --opt_model blackbox --solver gurobi --n_ptr_epochs 20 --n_epochs 10 --gpu ${GPU} --data_dir ${DIR}
python rethink_exp/main_results.py --problem=energy --opt_model identity --solver gurobi --n_ptr_epochs 20 --n_epochs 10 --gpu ${GPU} --data_dir ${DIR}
