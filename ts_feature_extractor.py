import numpy as np

random_state = 61
n_burn_in = 120
n_lookahead = 6

en_lat_bottom = -5
en_lat_top = 5
en_lon_left = 360. - 170.
en_lon_right = 360. - 120.

en_f_lat_bottom = -30
en_f_lat_top = 30
en_f_lon_left = 0
en_f_lon_right = 360


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
        tas = temperatures_xray['tas']

        data = tas.values
        n_times, n_lats, n_lons = data.shape
        valid_range = np.arange(n_burn_in, n_times - n_lookahead)

        # Removing running mean
        data_month = data.reshape(-1, 12, n_lats, n_lons)
        count = np.ones_like(data_month)
        data_mean = np.cumsum(data_month, axis=0) / np.cumsum(count, axis=0)
        data_mean = data_mean.reshape(-1, n_lats, n_lons)
        data = data - data_mean

        X = []
        for k in valid_range:
            X.append(data[k - 2:k + 1])
        X = np.array(X)

        return X


# if __name__ == '__main__':
#     import xray
#     resampled_xray = xray.open_dataset('resampled_xray.nc')

#     random_state = 61
#     n_burn_in = 120
#     n_lookahead = 6

#     idx = np.arange(n_burn_in, resampled_xray['time'].shape[0] - n_lookahead)
#     y_array = resampled_xray['target'][idx].values
#     y = y_array.reshape((y_array.shape[0], 1))
#     fe = FeatureExtractor()
#     X = fe.transform(resampled_xray, n_burn_in, n_lookahead, None)

#     # Look at univariate scores
#     for lag in [-3, -2, -1]:
#         from sklearn.feature_selection import f_regression
#         F, pv = f_regression(X[:, lag].reshape(len(X), -1), y)

#         from viz import plot_map

#         ds = resampled_xray
#         tas = select_box(ds['tas'])
#         _, n_lats, n_lons = tas.values.shape
#         F_ds = xray.Dataset(
#             {'tas': (['time', 'lat', 'lon'], F.reshape(1, n_lats, n_lons))},
#             coords={'time': ds['time'][:1], 'lat': ds['lat'], 'lon': ds['lon']})
#         fig = plot_map(F_ds['tas'], 0, clim=[0, 1200.])
