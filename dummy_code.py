import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from rgf.sklearn import RGFClassifier

# Load data
df_train = pd.read_csv("../data/01_c_skin_train.csv")
df_test = pd.read_csv("../data/01_c_skin_test.csv")

# Columns
cols_train = df_train.columns.tolist()
cols_test = df_test.columns.tolist()

# Subset relevant columns in data
use_this_cols = set(cols_train).intersection(cols_test)
df_train = df_train.loc[:, use_this_cols]
df_test = df_test.loc[:, use_this_cols]

# Target and features
y_train = df_train.loc[:, "label"]
X_train = df_train.drop("label", axis=1)

y_test = df_test.loc[:, "label"]
X_test = df_test.drop("label", axis=1)

# Label encoding
la_encoder_train = LabelEncoder()
y_train = pd.Series(la_encoder_train.fit_transform(y_train))

la_encoder_test = LabelEncoder()
y_test = pd.Series(la_encoder_test.fit_transform(y_test))


# Parameter grid
param_grid = {'max_leaf': [1000],
              'algorithm': ['RGF_Sib'],
              'l2': [1.0]}

# Grid search
grid_search = GridSearchCV(estimator=RGFClassifier(),
                           param_grid=param_grid,
                           scoring="roc_auc",
                           cv=5,
                           n_jobs=5,
                           verbose=3)

# Run grid search
grid_search.fit(X=X_train, y=y_train)

# Predict on test set
y_predict = grid_search.predict(X=X_test)

# Calculate AUC on test set
test_score = roc_auc_score(y_true=y_test, y_score=y_predict)

