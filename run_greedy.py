import time
import json
from greedy import RGF

# Files
files = [{"train": "../data/01_c_skin_train.csv", "test": "../data/01_c_skin_test.csv", "task": "classification"},
         {"train": "../data/01_r_air_train.csv", "test": "../data/01_r_air_test.csv", "task": "regression"},
         {"train": "../data/02_r_bike_train.csv", "test": "../data/02_r_bike_test.csv", "task": "regression"},
         {"train": "../data/03_r_gas_train.csv", "test": "../data/03_r_gas_test.csv", "task": "regression"}]

# RGF Grids
# max_leaf: Maximum number of leaf nodes in forest
# algorithm: RGF = L2 regularization on leaf-only models,
#            RGF_Opt = Min-penalty regularization
#            RGF_Sid = Min-penalty regularization with sum-to-zero sibling constrains
# l2: degree of L2 regularization
param_grid_normal = {'max_leaf': [1000, 5000, 10000],
                     'algorithm': ['RGF_Sib'],
                     'l2': [1.0, 0.1, 0.01]}

# n_estimators: The number of trees in the forest
# l1: L1 regularization
# l2: L2 regularization
param_grid_fast = {'n_estimators': [500, 1000, 2500],
                   'l1': [0.0, 1.0, 10.0],
                   'l2': [100.0, 1000.0, 10000.0]}

# Use fast RGF implementation?
use_fast = True
if use_fast:
    param_grid = param_grid_fast
else:
    param_grid = param_grid_normal

# Results container
results = []

# Run RGF models
for case in files:

    # Status
    print("   ")
    print("   ")
    print("Starting training (" + str(case["task"]) + ")")
    print("     Training file: " + str(case["train"]))
    print("     Test file: " + str(case["test"]))

    # Time tracking
    start_time = time.time()

    # Model
    mod = RGF(task=case["task"], fast=use_fast)

    # Load data
    mod.load_data(path_train=case["train"], path_test=case["test"])

    # Tune model (5-fold CV)
    mod.tune(grid=param_grid, folds=5, cores=5)

    # Score on test
    mod.score()

    # Results
    info = {"train": case["train"],
            "test": case["test"],
            "task": case["task"],
            "all_param": param_grid,
            "opt_param": mod.grid_search.best_params_,
            "cv_score": mod.grid_search.best_score_,
            "test_score": mod.test_score,
            "metric": mod.metric,
            "time": time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}
    results.append(info)

# Save model results to json
with open("../models/" + time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime(time.time())) + "_rgf_model.json", "w") as outfile:
    json.dump(results, outfile, sort_keys=True, indent=4)