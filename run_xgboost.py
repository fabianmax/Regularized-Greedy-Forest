import time
import json
from xgb import XGB

# Files
files = [{"train": "../data/01_c_skin_train.csv", "test": "../data/01_c_skin_test.csv", "task": "classification"},
         {"train": "../data/01_r_air_train.csv", "test": "../data/01_r_air_test.csv", "task": "regression"},
         {"train": "../data/02_r_bike_train.csv", "test": "../data/02_r_bike_test.csv", "task": "regression"},
         {"train": "../data/03_r_gas_train.csv", "test": "../data/03_r_gas_test.csv", "task": "regression"}]

# XGB Grid
# n_estimators: Number of boosted trees to fit
# max_depth: Maximum tree depth for base learners
# l1: L1 regularization
# l2: L2 regularization
param_grid = {'n_estimators': [100, 250, 500],
              'max_depth': [3, 6, 9],
              'reg_alpha': [0.0, 0.1, 1.0, 10.0],
              'reg_lambda': [0.0, 0.1, 1.0, 10.0]}

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
    mod = XGB(task=case["task"])

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
with open("../models/" + time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime(time.time())) + "_xgb_model.json", "w") as outfile:
    json.dump(results, outfile, sort_keys=True, indent=4)