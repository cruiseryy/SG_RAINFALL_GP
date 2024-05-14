import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt 

import geopandas as gpd
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
import cartopy.io.shapereader as shpreader
import matplotlib.ticker as mticker

from matplotlib import gridspec
import time 
import scipy.stats as stats
import tensorflow as tf
import tensorflow_probability as tfp

plt.rcParams['font.family'] = 'Myriad Pro'

# the Kling-Gupta Efficiency (KGE) score
def kge(true_, fit):
    r = np.corrcoef(true_, fit)[0, 1]
    alpha = np.std(fit, ddof = 1) / np.std(true_, ddof= 1)
    beta = np.mean(fit) / np.mean(true_)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def nse(true_, fit):
    return 1 - np.sum((true_ - fit) ** 2) / np.sum((true_ - np.mean(true_)) ** 2)

# given prior info: mu_x, mu_y, kxx, kyy, kxy (derived from simulated rainfall)
# conditioned on some observation of x: xx_obs
# update the posterior distribution of y: mu_y, cov_y
def gp_infer(xx_obs, mu_x, kyx, kxx, mu_y, kyy):
    mu_y = mu_y + kyx @ np.linalg.inv(kxx) @ (xx_obs - mu_x).T
    kyy = kyy - kyx @ np.linalg.inv(kxx) @ kyx.T
    return mu_y.squeeze(), kyy

class gp_interpolator:
    def __init__(self, P, e0 = 30, thres = 1e-3) -> None:
        # # of stations 
        self.p_ = P
        # see Peng & Albertson 2021 but a vector of initial noise level is used here
        self.sn = e0 * np.ones(self.p_) # initial noise level
        self.sn_thres = thres # a user-specified threshold for the noise level convergence
        self.iter = 0 # iteration counter
        return
    
    def read_rainfall(self, obs, sim):
        # read observational rainfall in a matrix of N by P
        # N = # of months, P = # of stations
        self.rain_obs = obs
        # for QoL, get simulated rainfall time series corresponding to their station-based counterparts
        # the size of this matrix should also be N by P
        self.wrf_sta = sim
        # prepare the prior distribution
        self.mu_x = np.mean(self.wrf_sta, axis = 0)
        self.kxx = np.cov(self.wrf_sta.T, ddof = 1)
        return
    
    def sn_converge(self):
        sn0 = 9999 * np.ones(self.p_)
        xx = self.wrf_sta
        xx_obs = self.rain_obs
        while np.mean((sn0 - self.sn)**2) / np.mean(self.sn**2) > self.sn_thres:
            self.iter += 1
            sn0 = self.sn
            tmp_err = []
            for j in range(self.p_):
                tsn = np.delete(self.sn, obj = j)
                txx = np.delete(xx, obj = j, axis = 1)
                tmu_x = np.mean(txx, axis = 0)
                tkxx = np.cov(txx.T, ddof = 1)
                tkyx = np.cov(xx[:,j].T, txx.T, ddof = 1)[:1, 1:]
                tmu_y = np.mean(xx[:,j])
                tkyy = np.cov(xx[:,j], ddof = 1)
                txx_obs = np.delete(xx_obs, obj = j, axis = 1)
                fit_y, _ = gp_infer(xx_obs = txx_obs, mu_x = tmu_x, kyx = tkyx, kxx = tkxx + np.diag(tsn**2),
                                    mu_y = tmu_y, kyy = tkyy)
                tmp_err.append(np.sqrt(np.mean((xx_obs[:, j] - fit_y) ** 2)))
            self.sn = np.array(tmp_err)
        return
    
    def predict(self, yy):
        kyy = np.cov(yy.T, ddof = 1)
        mu_y = np.mean(yy, axis = 0)
        kyx = np.cov(yy.T, self.wrf_sta.T, ddof = 1)[:yy.shape[1], yy.shape[1]:]
        post_mu_y, post_kyy = gp_infer(xx_obs = self.rain_obs, mu_x = self.mu_x, kyx = kyx, 
                                       kxx = self.kxx + np.diag(self.sn**2), mu_y = mu_y, kyy = kyy)
        return post_mu_y, post_kyy

