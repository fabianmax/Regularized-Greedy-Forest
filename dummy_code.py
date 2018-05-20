import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from rgf.sklearn import RGFClassifier

# Load data
df_train = pd.read_csv("path_to_training.csv")
df_test = pd.read_csv("path_to_test.csv")

# Target and features
y_train = df_train.loc[:, "label"]
X_train = df_train.drop("label", axis=1)

y_test = df_test.loc[:, "label"]
X_test = df_test.drop("label", axis=1)

# Parameter grid
param_grid = {'max_leaf': [1000, 5000, 10000],
              'algorithm': ['RGF_Sib'],
              'l2': [1.0, 0.1, 0.01]}

# Grid search
grid_search = GridSearchCV(estimator=RGFClassifier(),
                           param_grid=param_grid,
                           scoring="roc_auc",
                           cv=10,
                           n_jobs=4,
                           verbose=3)

# Run grid search
grid_search.fit(X=X_train, y=y_train)

# Predict on test set
y_predict = grid_search.predict(X=X_test)

# Calculate AUC on test set
test_score = roc_auc_score(y_true=y_test, y_score=y_predict)

