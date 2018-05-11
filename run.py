import time
from greedy import RGF

# Files
files = [{"train": "../data/01_c_skin_train.csv", "test": "../data/01_c_skin_test.csv", "task": "classification"},
         {"train": "../data/01_r_air_train.csv", "test": "../data/01_r_air_test.csv", "task": "regression"},
         {"train": "../data/02_r_bike_train.csv", "test": "../data/02_r_bike_test.csv", "task": "regression"},
         {"train": "../data/03_r_gas_train.csv", "test": "../data/03_r_gas_test.csv", "task": "regression"}]

# Grid
# max_leaf: Maximum number of leaf nodes in forest
# algorithm: RGF = L2 regularization on leaf-only models,
#            RGF_Opt = Min-penalty regularization
#            RGF_Sod = Min-penalty regularization with sum-to-zero sibling constrains
# l2: degree of L2 regularization
param_grid = {'max_leaf': [1000, 5000, 10000],
              'algorithm': ['RGF', 'RGF_Opt', 'RGF_Sib'],
              'l2': [1.0, 0.1, 0.01]}

# Results container
results = []

# Run models
for case in files:

    # Status
    print("Starting training (" + str(case["task"]) + ")")
    print("     Training file: " + str(case["train"]))
    print("     Test file: " + str(case["test"]))

    # Time tracking
    start_time = time.time()

    # Model
    mod = RGF(task=case["task"])

    # Load data
    mod.load_data(path_train=case["train"], path_test=case["test"])

    # Tune model
    mod.tune(grid=param_grid)

    # Score on test
    mod.score()

    # Results
    info = {"train": case["train"],
            "test": case["test"],
            "task": case["task"],
            "opt_param": mod.grid_search.best_params_,
            "cv_score": mod.grid_search.best_score_,
            "test_score": mod.test_score,
            "metric": mod.metric,
            "time": time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}
    results.append(info)
