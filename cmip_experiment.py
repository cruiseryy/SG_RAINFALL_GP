
import numpy as np
import time 
import tensorflow as tf
import tensorflow_probability as tfp
import gc
from utils import *

# i have to wrap things up in a class to avoid out-of-memory error
class mc_generator:
    def __init__(self, nn = 1000) -> None:
        self.nn = nn
        self.mask = np.loadtxt('mask.txt') # i just hardcoded it here
        return
    def gpr_init(self, obs, sim, sim_full, P = 14):
        self.gpr = gp_interpolator(P = P)
        self.gpr.read_rainfall(obs, sim)
        self.gpr.sn_converge()
        mu, self.cov = self.gpr.predict(sim_full)
        self.mu = mu.T
        return
    def mc_generate(self, i = 0):
        self.res = []
        for yr in range(40):
            t1 = time.time()
            mu_tf = tf.convert_to_tensor(self.mu[yr, :], dtype=tf.float32)
            tmp_cov = self.cov + np.eye(self.cov.shape[1]) * np.mean(np.diag(self.cov)) * 1e-4
            cov_tf = tf.convert_to_tensor(tmp_cov, dtype=tf.float32)
            mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=mu_tf, covariance_matrix=cov_tf)
            data_sample1 = mvn.sample(sample_shape=(self.nn,))
            data_sample = data_sample1.numpy()
            data_sample[data_sample < 0] = 0
            data_map = np.reshape(data_sample, (self.nn, 120, 160))
            sg_rain_samples = np.nanmean(np.multiply(data_map, self.mask), axis = (1, 2))
            self.res.append(sg_rain_samples)
            print(f'{i+1}th month {yr+1}th year: {time.time() - t1:.2f} seconds')
        return
    def write_(self, i = 0, path = 'cmip_era5_comparison/era5/'):
        np.savetxt(path + f'sample_{i}.txt', np.array(self.res))
        return

# -------------------------------------------------------------------------------------------------
# DATA PREPARATION
# the station-based rainfall data is already organized in a N by P matrix
rain_obs = np.loadtxt('data/sta_monthly.csv')

# the simulated rainfall data is reshaped to a 2d matrix of size N by (W x L)
rain_sim_flatten = np.loadtxt('data/wrf_monthly.csv')
rain_sim = rain_sim_flatten.reshape(rain_sim_flatten.shape[0], 120, 160)

# sim_sel constains a P by 4 matrix
# the first two columns are the row and column indices of grids corresponding to the stations
# the last two columns are the lons and lats of grids corresponding to the stations
sim_sel = np.loadtxt('data/wrf_loc.csv')
sim_idx = sim_sel[:, :2].astype(int)

# select simulated rainfall at stations
wrf_sta = np.array([rain_sim[:, i, j] for (i, j) in sim_idx]).T

# lats and lons of all grids are stored in a 2d matrix of 2 by (W x L)
# the first row is lons and the second row is lats
longlat = np.loadtxt('data/lonlat.txt')
lons = longlat[0, :].reshape(120, 160)
lats = longlat[1, :].reshape(120, 160)

# read station lons (3rd column) and lats (4th column)
sta_loc = np.genfromtxt('data/sta_lookup_new.csv', delimiter=',')[:, 2:]

# read wrf simulated rainfall forced by cmip6 future climate
cmip_rain_f = np.loadtxt('data/wrf_monthly_cmip.txt')[-40:, :]
cmip_rain0 = cmip_rain_f.reshape(cmip_rain_f.shape[0], 12, 120, 160)
cmip_rain = np.zeros(cmip_rain0.shape)
cmip_rain[:, 0:6, :, :] = cmip_rain0[:, 6:12, :, :]
cmip_rain[:, 6:12, :, :] = cmip_rain0[:, 0:6, :, :]
cmip_rain_flatten = np.reshape(cmip_rain, (40, 12, 120*160))
wrf_sta_fut = np.zeros((40, 12, sim_idx.shape[0]))
for idx, (i, j) in enumerate(sim_idx):
    wrf_sta_fut[:, :, idx] = cmip_rain[:, :, i, j]

# -------------------------------------------------------------------------------------------------
# interpolation using ERA5-forced WRF simulations
for i in range(12):
    mc_ = mc_generator()
    mc_.gpr_init(rain_obs[i::12, :], wrf_sta[i::12, :], rain_sim_flatten[i::12, :])
    mc_.mc_generate(i)
    mc_.write_(i, path = 'cmip_era5_comparison/era5/')
    del mc_
    gc.collect()

# -------------------------------------------------------------------------------------------------
# interpolation using CMIP6-forced WRF simulations
for i in range(12):
    mc_ = mc_generator()
    mc_.gpr_init(rain_obs[i::12, :], wrf_sta_fut[:, i, :], cmip_rain_flatten[:, i, :])
    mc_.mc_generate(i)
    mc_.write_(i, path = 'cmip_era5_comparison/cmip6/')
    del mc_
    gc.collect()

# -------------------------------------------------------------------------------------------------
# collect obs rainfall
obs_all = []
for i in range(12):
    obs_all.append(np.mean(rain_obs[i::12, :], axis = 1))
np.savetxt('cmip_era5_comparison/obs.txt', np.array(obs_all))