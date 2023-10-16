DIR="/home/genghaoyu/OR/PTO/PTO-xinye/data/"
GPU=0
PTR_EPS=50
EPOCHS=50

# Prediction-focused learning
python rethink_exp/main_results.py --problem=advertising --opt_model bce      --solver ortools --n_ptr_epochs ${PTR_EPS} --n_epochs ${EPOCHS} --gpu ${GPU} 

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=advertising --opt_model dfl      --solver ortools --n_ptr_epochs ${PTR_EPS} --n_epochs ${EPOCHS} --gpu ${GPU} 
python rethink_exp/main_results.py --problem=advertising --opt_model identity --solver ortools --n_ptr_epochs ${PTR_EPS} --n_epochs ${EPOCHS} --gpu ${GPU} 
python rethink_exp/main_results.py --problem=advertising --opt_model blackbox --solver ortools --n_ptr_epochs ${PTR_EPS} --n_epochs ${EPOCHS} --gpu ${GPU} 
python rethink_exp/main_results.py --problem=advertising --opt_model spo      --solver ortools --n_ptr_epochs ${PTR_EPS} --n_epochs ${EPOCHS} --gpu ${GPU} 
python rethink_exp/main_results.py --problem=advertising --opt_model nce      --solver ortools --n_ptr_epochs ${PTR_EPS} --n_epochs ${EPOCHS} --gpu ${GPU} 
python rethink_exp/main_results.py --problem=advertising --opt_model pointLTR --solver ortools --n_ptr_epochs ${PTR_EPS} --n_epochs ${EPOCHS} --gpu ${GPU} 
python rethink_exp/main_results.py --problem=advertising --opt_model listLTR  --solver ortools --n_ptr_epochs ${PTR_EPS} --n_epochs ${EPOCHS} --gpu ${GPU} 
python rethink_exp/main_results.py --problem=advertising --opt_model pairLTR  --solver ortools --n_ptr_epochs ${PTR_EPS} --n_epochs ${EPOCHS} --gpu ${GPU} 
python rethink_exp/main_results.py --problem=advertising --opt_model lodl     --solver ortools --n_ptr_epochs ${PTR_EPS} --n_epochs ${EPOCHS} --gpu ${GPU} 
# python rethink_exp/main_results.py --problem=advertising --opt_model perturb  --solver ortools --n_epochs ${EPOCHS} --gpu ${GPU} 

