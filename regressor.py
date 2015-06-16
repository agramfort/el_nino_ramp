from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction.image import grid_to_graph


N_JOBS = -1


class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = Pipeline([
            ("RF", RandomForestRegressor(n_estimators=200, max_depth=15,
                                         n_jobs=N_JOBS))])
        self.scaler = StandardScaler()
        self.agglo = FeatureAgglomeration(n_clusters=500)

    def fit(self, X, y):
        y = y.ravel()
        n_samples, n_lags, n_lats, n_lons = X.shape
        self.scaler.fit(X[:, -1].reshape(n_samples, -1))
        X = X.reshape(n_lags * n_samples, -1)
        connectivity = grid_to_graph(n_lats, n_lons)
        self.agglo.connectivity = connectivity
        X = self.scaler.transform(X)
        X = self.agglo.fit_transform(X)
        X = X.reshape(n_samples, -1)
        self.clf.fit(X, y)

    def predict(self, X):
        n_samples, n_lags, n_lats, n_lons = X.shape
        X = X.reshape(n_lags * n_samples, -1)
        X = self.scaler.transform(X)
        X = self.agglo.transform(X)
        X = X.reshape(n_samples, -1)
        return self.clf.predict(X)
