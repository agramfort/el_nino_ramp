from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest, f_regression
# from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def apply_mask(X, mask):
    X = X[:, :, mask]
    X = X.reshape(len(X), -1)
    return X

N_JOBS = -1


class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = Pipeline([
            # ('scaler', StandardScaler()),
            # ('pca', PCA(n_components=20)),
            ("RF", RandomForestRegressor(n_estimators=100, max_depth=15,
                                         n_jobs=N_JOBS))])

    def fit(self, X, y):
        y = y.ravel()
        n_samples, n_lags, n_clusters = X.shape
        X1 = X[:, -1].reshape(n_samples, n_clusters)
        select = SelectKBest(f_regression, k=min(1500, n_clusters))
        select.fit(X1, y)
        mask = select._get_support_mask()
        X = apply_mask(X, mask)
        self.mask_ = mask
        self.clf.fit(X, y)

    def predict(self, X):
        X = apply_mask(X, self.mask_)
        return self.clf.predict(X)
