from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest, f_regression
# from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA


class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = Pipeline([('scaler', StandardScaler()),
                             ('anova', SelectKBest(f_regression, k=1500)),
                             ('KernelPCA', KernelPCA(n_components=10)),
                             ("RF", RandomForestRegressor(n_estimators=200, max_depth=15))])

    def fit(self, X, y):
        y = y.ravel()
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
