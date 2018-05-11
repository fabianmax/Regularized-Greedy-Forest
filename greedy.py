import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

from rgf.sklearn import RGFClassifier, RGFRegressor


class RGF:

    """
    Wrapper for Regularized Greedy Forest
    based on RGFClassifier (classification) and RGFRegressor (regression)
    https://github.com/RGF-team/rgf_python

    Parameters
    ----------
    task: string ("classification", "regression")
        Either regression of classification task

    """

    def __init__(self, task):
        if task == 'classification':
            self.model = RGFClassifier(loss="Log")
            self.metric = 'roc_auc'
            self.task = "classification"
        else:
            self.model = RGFRegressor(loss="LS", normalize=True)
            self.metric = 'neg_mean_squared_error'
            self.task = "regression"
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_train = None
        self.grid_search = None
        self.y_predict = None
        self.test_score = None

    def load_data(self, path_train, path_test):
        """
        Method for loading data from path

        :param path_train: path to training csv
        :param path_test: path to test csv
        :return: None
        """

        # Load data
        df_train = pd.read_csv(path_train)
        df_test = pd.read_csv(path_test)

        # Columns
        cols_train = df_train.columns.tolist()
        cols_test = df_test.columns.tolist()

        # Subset relevant columns in data
        use_this_cols = set(cols_train).intersection(cols_test)
        df_train = df_train.loc[:, use_this_cols]
        df_test = df_test.loc[:, use_this_cols]

        # Target and features
        self.y_train = df_train.loc[:, "label"]
        self.X_train = df_train.drop("label", axis=1)

        self.y_test = df_test.loc[:, "label"]
        self.X_test = df_test.drop("label", axis=1)

        # Label encoding
        if self.task == 'classification':
            la_encoder_train = LabelEncoder()
            self.y_train = pd.Series(la_encoder_train.fit_transform(self.y_train))

            la_encoder_test = LabelEncoder()
            self.y_test = pd.Series(la_encoder_test.fit_transform(self.y_test))

    def tune(self, grid):
        """
        Method for parameter optimization via grid search

        :param grid: dict of parameters
        :return: None
        """

        # CV object
        self.grid_search = GridSearchCV(estimator=self.model,
                                        param_grid=grid,
                                        scoring=self.metric,
                                        cv=5,
                                        n_jobs=4,
                                        verbose=2)

        # Run CV
        self.grid_search.fit(X=self.X_train, y=self.y_train)

    def score(self):
        """
        Method for scoring test dataset

        :return: None
        """

        # Prediction on test sample
        self.y_predict = self.grid_search.predict(X=self.X_test)

        # Score
        if self.task == 'classification':
            self.test_score = roc_auc_score(y_true=self.y_test, y_score=self.y_predict)
        else:
            self.test_score = np.sqrt(mean_squared_error(y_true=self.y_test, y_pred=self.y_predict))