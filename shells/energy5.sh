GPU=0
EPOCHS=100

python rethink_exp/main_results.py --problem=energy --opt_model blackbox --solver gurobi --n_ptr_epochs 20 --n_epochs 10 --gpu ${GPU}
python rethink_exp/main_results.py --problem=energy --opt_model identity --solver gurobi --n_ptr_epochs 20 --n_epochs 10 --gpu ${GPU}
