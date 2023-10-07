EPOCHS=100
GPU=0

# Prediction-focused learning【[]
python rethink_exp/main_results.py --problem=bipartitematching --opt_model mse      --solver cvxpy --n_epochs ${EPOCHS}  --instances 20 --testinstances 6 

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=bipartitematching --opt_model dfl      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 
python rethink_exp/main_results.py --problem=bipartitematching --opt_model blackbox --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 
python rethink_exp/main_results.py --problem=bipartitematching --opt_model identity --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 
python rethink_exp/main_results.py --problem=bipartitematching --opt_model nce      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 
python rethink_exp/main_results.py --problem=bipartitematching --opt_model pointLTR --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 
python rethink_exp/main_results.py --problem=bipartitematching --opt_model listLTR  --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 
python rethink_exp/main_results.py --problem=bipartitematching --opt_model pairLTR  --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 
python rethink_exp/main_results.py --problem=bipartitematching --opt_model lodl     --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 
python rethink_exp/main_results.py --problem=bipartitematching --opt_model spo      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 
# prediction + decision
python rethink_exp/main_results.py --problem=bipartitematching --opt_model blackbox --solver cvxpy --prefix "ptr-ftn" --n_ptr_epochs 50 --n_epochs 50 --gpu ${GPU} --instances 20 --testinstances 6 
python rethink_exp/main_results.py --problem=bipartitematching --opt_model identity --solver cvxpy --prefix "ptr-ftn" --n_ptr_epochs 50 --n_epochs 50 --gpu ${GPU} --instances 20 --testinstances 6 