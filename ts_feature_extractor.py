import numpy as np
from sklearn.feature_extraction.image import grid_to_graph
from sklearn import cluster
from sklearn.preprocessing import StandardScaler

random_state = 61
n_burn_in = 120
n_lookahead = 6

en_lat_bottom = -5
en_lat_top = 5
en_lon_left = 360. - 170.
en_lon_right = 360. - 120.


class FeatureExtractor(object):

    def __init__(self):
        pass

    def fit(self, temperatures_xray, n_burn_in, n_lookahead):
        pass

    def transform(self, temperatures_xray, n_burn_in=n_burn_in,
                  n_lookahead=n_lookahead, skf_is=None):
        """Combine two variables: the montly means corresponding to the
        month of the target and the current mean temperature in the
        El Nino 3.4 region."""
        data = temperatures_xray['tas'].values
        n_times, n_lats, n_lons = data.shape
        # n_locations = n_lats * n_lons
        valid_range = np.arange(n_burn_in, n_times - n_lookahead)

        data = data.reshape(n_times, -1)
        connectivity = grid_to_graph(n_lats, n_lons)
        agglo = cluster.FeatureAgglomeration(connectivity=connectivity,
                                             n_clusters=500)
                                             # n_clusters=2664)
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        data = agglo.fit_transform(data)
        self.agglo_ = agglo

        X = []
        for k in valid_range:
            X.append(data[k - 2:k + 1])
        X = np.array(X)
        return X

if __name__ == '__main__':
    import xray
    resampled_xray = xray.open_dataset('resampled_xray.nc')

    random_state = 61
    n_burn_in = 120
    n_lookahead = 6

    idx = np.arange(n_burn_in, resampled_xray['time'].shape[0] - n_lookahead)
    y_array = resampled_xray['target'][idx].values
    y = y_array.reshape((y_array.shape[0], 1))
    fe = FeatureExtractor()
    X = fe.transform(resampled_xray, n_burn_in, n_lookahead, None)

    # Look at univariate scores
    from sklearn.feature_selection import f_regression
    F, pv = f_regression(X[:, -1, :], y)
    F = fe.agglo_.inverse_transform(F)

    from viz import plot_map

    ds = resampled_xray
    _, n_lats, n_lons = ds['tas'].values.shape
    F_ds = xray.Dataset(
        {'tas': (['time', 'lat', 'lon'], F.reshape(1, n_lats, n_lons))},
        coords={'time': ds['time'][:1], 'lat': ds['lat'], 'lon': ds['lon']})
    plot_map(F_ds['tas'], 0)
