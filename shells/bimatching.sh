GPU=0
EPOCHS=100
<<<<<<< HEAD
<<<<<<< HEAD:shells/bipartitematching.sh

# Prediction-focused learning【[]
python rethink_exp/main_results.py --problem=bipartitematching --opt_model ce       --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=bipartitematching --opt_model dfl      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --losslr 0.01 --n_layers 1 --n_hidden 32
python rethink_exp/main_results.py --problem=bipartitematching --opt_model blackbox --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --losslr 0.01 --n_layers 1 --n_hidden 32
python rethink_exp/main_results.py --problem=bipartitematching --opt_model spo      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --losslr 0.01 --n_layers 1 --n_hidden 32
=======
=======
>>>>>>> 252b7c78851c07b58831e055cb49beb3a19b44d0

# Prediction-focused learning
python rethink_exp/main_results.py --problem=bipartitematching --opt_model ce       --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 

# Decisoin-focused learning
python rethink_exp/main_results.py --problem=bipartitematching --opt_model dfl      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --losslr 0.01 --n_layers 1 --n_hidden 32
python rethink_exp/main_results.py --problem=bipartitematching --opt_model blackbox --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --losslr 0.01 --n_layers 1 --n_hidden 32
python rethink_exp/main_results.py --problem=bipartitematching --opt_model spo      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --losslr 0.01 --n_layers 1 --n_hidden 32
<<<<<<< HEAD
python rethink_exp/main_results.py --problem=bipartitematching --opt_model identity --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --losslr 0.01 --n_layers 1 --n_hidden 32
python rethink_exp/main_results.py --problem=bipartitematching --opt_model nce      --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --losslr 0.01 --n_layers 1 --n_hidden 32
python rethink_exp/main_results.py --problem=bipartitematching --opt_model pointLTR --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --losslr 0.01 --n_layers 1 --n_hidden 32
python rethink_exp/main_results.py --problem=bipartitematching --opt_model listLTR  --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --losslr 0.01 --n_layers 1 --n_hidden 32
python rethink_exp/main_results.py --problem=bipartitematching --opt_model pairLTR  --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --losslr 0.01 --n_layers 1 --n_hidden 32
python rethink_exp/main_results.py --problem=bipartitematching --opt_model lodl     --solver cvxpy --n_epochs ${EPOCHS} --gpu ${GPU} --instances 20 --testinstances 6 --losslr 0.01 --n_layers 1 --n_hidden 32


>>>>>>> 840ead186bc3d78407952c78ffd6d83fe221e67d:shells/bimatching.sh
=======
>>>>>>> 252b7c78851c07b58831e055cb49beb3a19b44d0
# prediction + decision
python rethink_exp/main_results.py --problem=bipartitematching --opt_model blackbox --solver cvxpy --prefix "ptr-ftn" --n_ptr_epochs 50 --n_epochs 50 --gpu ${GPU} --instances 20 --testinstances 6 --losslr 0.01 --n_layers 1 --n_hidden 32
python rethink_exp/main_results.py --problem=bipartitematching --opt_model identity --solver cvxpy --prefix "ptr-ftn" --n_ptr_epochs 50 --n_epochs 50 --gpu ${GPU} --instances 20 --testinstances 6 --losslr 0.01 --n_layers 1 --n_hidden 32


