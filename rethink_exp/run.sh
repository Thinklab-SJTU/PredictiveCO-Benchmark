'''
# Supported problems: 
    knapsack (--prob_version=gen),

# Supported models: 
    spo, 

'''

# budget allocation
python rethink_exp/main_results.py --problem=budgetalloc --opt_model mse --epochs 1 --gpu 0 
# cubic
python rethink_exp/main_results.py --problem=cubic --opt_model mse --epochs 1 --gpu 0 
# bipartitematching
python rethink_exp/main_results.py --problem=bipartitematching --opt_model mse --epochs 1 --gpu 0 
# rmab
python rethink_exp/main_results.py --problem=rmab --opt_model mse --epochs 1 --gpu 0 
# portfolio
python rethink_exp/main_results.py --problem=portfolio --opt_model mse --epochs 1 --gpu 0 
# Knapsack
python rethink_exp/main_results.py --problem=knapsack --opt_model blackbox --solver gurobi --epochs 1 --gpu 0