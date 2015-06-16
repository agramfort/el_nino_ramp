import numpy as np

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
        n_locations = n_lats * n_lons
        valid_range = np.arange(n_burn_in, n_times - n_lookahead)
        X = []
        for k in valid_range:
            X.append(data[k - 2:k + 1])
        X = np.array(X)
        return X

if __name__ == '__main__':
    import xray
    resampled_xray = xray.open_dataset('resampled_xray.nc')
# resampled_xray = xray.open_dataset('COLA_data/resampled_tas_Amon_CCSM4_piControl_r1i1p1_080001-130012.nc')

    random_state = 61
    n_burn_in = 120
    n_lookahead = 6

    valid_range = range(n_burn_in, resampled_xray['time'].shape[0] - n_lookahead)
    y_array = resampled_xray['target'][valid_range].values
    y = y_array.reshape((y_array.shape[0], 1))
    fe = FeatureExtractor()
    X = fe.transform(resampled_xray, n_burn_in, n_lookahead, None)

