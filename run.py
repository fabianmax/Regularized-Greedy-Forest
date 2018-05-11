from greedy import RGF


# Files
files = [{"train": "../data/01_c_skin_train.csv", "test": "../data/01_c_skin_test.csv", "task": "classification"},
         {"train": "../data/01_r_air_train.csv", "test": "../data/01_r_air_test.csv", "task": "regression"},
         {"train": "../data/02_r_bike_train.csv", "test": "../data/02_r_bike_test.csv", "task": "regression"},
         {"train": "../data/03_r_gas_train.csv", "test": "../data/03_r_gas_test.csv", "task": "regression"}]

files = files[1:3]

# Grid
# max_leaf: Maximum number of leaf nodes in forest
# algorithm: RGF = L2 regularization on leaf-only models,
#            RGF_Opt = Min-penalty regularization
#            RGF_Sod = Min-penalty regularization with sum-to-zero sibling constrains
# l2: degree of L2 regularization
param_grid = {'max_leaf': [1000, 5000, 10000],
              'algorithm': ['RGF', 'RGF_Opt', 'RGF_Sib'],
              'l2': [1.0, 0.1, 0.01]}

for case in files:

    # Status
    print("Starting training (" + str(case["task"]) + ")")
    print("     Training file: " + str(case["train"]))
    print("     Test file: " + str(case["test"]))

    # Model
    mod = RGF(task=case["task"])

    # Load data
    mod.load_data(path_train=case["train"], path_test=case["test"])

    # Tune model
    mod.tune(grid=param_grid)

    # Score on test
    mod.score()


























# Change dir
os.chdir(os.path.expanduser("~/Intern/Projekte/Blog/RGF"))

# Load data
df_train = pd.read_csv("data/01_c_skin_train.csv")
df_test = pd.read_csv("data/01_c_skin_test.csv")

# Keep only columns available in test set
ava_in_test = df_test.columns.tolist()
df_train = df_train.loc[:, ava_in_test]

# Target and features
y_train = df_train.loc[:, "label"]
X_train = df_train.drop("label", axis=1)

y_test = df_test.loc[:, "label"]
X_test = df_test.drop("label", axis=1)

# Transform target to 0/1
y_train = y_train - 1
y_test = y_test - 1

# Regularized Greedy Forest
rgf = RGFClassifier()

# Parameter grid
param_grid = {'max_leaf': [500, 1000],
              'algorithm': ['RGF_Sib'],
              'l2': [1.0, 0.1, 0.01]}

# CV Object
grid_search = GridSearchCV(estimator=rgf,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=5,
                           n_jobs=4,
                           verbose=2)

# Run CV
grid_search.fit(X=X_train, y=y_train)

# Results from CV
grid_search.best_estimator_
grid_search.best_params_
grid_search.best_score_

# Prediction on test sample
y_predict = grid_search.predict(X=X_test)

# Accuracy on test sample
test_score = accuracy_score(y_true=y_test, y_pred=y_predict)
print('RGF Classfier score: {0:.5f}'.format(test_score))
