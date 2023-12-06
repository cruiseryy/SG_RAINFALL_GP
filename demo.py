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


class gp:
    def __init__(self,
                 sta ='data/sta_monthly.csv',
                 wrf = 'data/wrf_monthly.csv',
                 sta_spec = 'data/sta_lookup_new.csv',
                 wrf_loc = 'data/wrf_loc.csv',
                 geo = 'data/lonlat.txt',
                 target_season = [1, 2]
                 ) -> None:
        
        
        self.ts = target_season
        
        self.rain_sta = np.loadtxt(sta)

        # see Peng & Albertson 2021 but a vector of initial noise level is used here
        self.sn = 30 * np.ones(self.rain_sta.shape[1]) # initial noise level
        
        self.rain_wrf_f = np.loadtxt(wrf)
        self.rain_wrf = self.rain_wrf_f.reshape(self.rain_wrf_f.shape[0], 120, 160)

        self.sta_loc = np.genfromtxt(sta_spec, delimiter=',')[:, 2:]
        self.wrf_idx = np.loadtxt(wrf_loc)[:, :2].astype(int)
        self.wrf_loc = np.loadtxt(wrf_loc)[:, 2:]

        lonlat = np.loadtxt(geo)
        self.lon = lonlat[0, :].reshape([120, 160])
        self.lat = lonlat[1, :].reshape([120, 160])

        self.wrf_sta = np.array([self.rain_wrf[:, i, j] for _, (i, j) in enumerate(self.wrf_idx)]).T

        self.coastline = gpd.read_file('/home/mizu_home/xp53/nas_home/coastlines-split-SGregion/lines.shp')
        self.mask = np.loadtxt('mask.txt')

        # a small tweak to use cmip6 covariance matrix 
        cmip_rain_f = np.loadtxt('data/wrf_monthly_cmip.txt')[-40:, :]
        cmip_rain0 = cmip_rain_f.reshape(cmip_rain_f.shape[0], 12, 120, 160)
        self.cmip_rain = np.zeros(cmip_rain0.shape)
        self.cmip_rain[:, 0:6, :, :] = cmip_rain0[:, 6:12, :, :]
        self.cmip_rain[:, 6:12, :, :] = cmip_rain0[:, 0:6, :, :]
        pause = 1
        return
    
    def kge(self, true_, fit):
        # the Kling-Gupta Efficiency
        r = np.corrcoef(true_, fit)[0, 1]
        alpha = np.std(fit, ddof = 1) / np.std(true_, ddof= 1)
        beta = np.mean(fit) / np.mean(true_)

        return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    def gp_fit(self, rxx, mu_x, kyx, kxx, mu_y, kyy):
        mu_y = mu_y + kyx @ np.linalg.inv(kxx) @ (rxx - mu_x).T
        cov_y = kyy - kyx @ np.linalg.inv(kxx) @ kyx.T
        return mu_y.squeeze(), cov_y
    
    def sn_converge(self):
        xx = np.zeros([self.wrf_sta.shape[0]//12, self.wrf_sta.shape[1]])
        xx_obs = np.zeros([self.rain_sta.shape[0]//12, self.rain_sta.shape[1]])
        for i_ in self.ts:
            xx += self.wrf_sta[(i_-1)::12, :]
            xx_obs += self.rain_sta[(i_-1)::12, :]
        mu_x = np.mean(xx, axis = 0)
        sn0 = 9999 * np.ones(self.rain_sta.shape[1])
        while np.mean((sn0 - self.sn)**2) / np.mean(self.sn**2) > 0.001:
            sn0 = self.sn
            tmp_err = []
            for j in range(xx.shape[1]):
                # mask/delete the selected grid from sim rainfall
                tsn = np.delete(self.sn, obj = j)
                txx = np.delete(xx, obj = j, axis = 1)
                tmu_x = np.mean(txx, axis = 0)
                kxx = np.cov(txx.T, ddof = 1)
                kyx = np.cov(xx[:,j].T, txx.T, ddof = 1)[:1, 1:]
                kyy = np.cov(xx[:,j], ddof = 1)
                # mask/delete the corresponding station from obs rainfall
                txx_obs = np.delete(xx_obs, obj = j, axis = 1)
                fit_y, _ = self.gp_fit(rxx = txx_obs, mu_x = tmu_x, kxx = kxx + np.diag(tsn**2), kyx = kyx, mu_y = mu_x[j], kyy = kyy)
                tmp_err.append(np.sqrt(np.mean((xx_obs[:, j] - fit_y) ** 2))) 
            self.sn = np.array(tmp_err)
            pause = 1
        return

    
    def validation(self):
        xx = np.zeros([self.wrf_sta.shape[0]//12, self.wrf_sta.shape[1]])
        xx_obs = np.zeros([self.rain_sta.shape[0]//12, self.rain_sta.shape[1]])
        for i_ in self.ts:
            xx += self.wrf_sta[(i_-1)::12, :]
            xx_obs += self.rain_sta[(i_-1)::12, :]
        mu_x = np.mean(xx, axis = 0)

        # this chunk of codes is to check the predictions at training points
        # kxx = np.cov(xx.T, ddof = 1)
        # ftr, _ = self.gp_fit(fxx = fxx, mu_fx = mu_x, kxx = kxx, kxy = kxx, mu_fy = mu_x, kyy = kxx) 
        # pause = 1
        # Kxx = np.cov(xx.T, ddof=1)
        # kxx_w = np.cov(xx.T, ddof=1)
        # kxx_s = np.cov(fxx.T, ddof=1)
        # pause = 1
        
        fig, ax = plt.subplots(figsize = [16, 14], nrows = 4, ncols = 4)
        # the usage of y kinda confusing here
        f_est = np.zeros(xx_obs.shape)
        score_kge = []
        score_cc = []
        for j in range(xx.shape[1]):
            # mask/delete the selected grid from sim rainfall
            tsn = np.delete(self.sn, obj = j)
            txx = np.delete(xx, obj = j, axis = 1)
            tmu_x = np.mean(txx, axis = 0)
            kxx = np.cov(txx.T, ddof = 1)
            kyx = np.cov(xx[:,j].T, txx.T, ddof = 1)[:1, 1:]
            kyy = np.cov(xx[:,j], ddof = 1)
            # mask/delete the corresponding station from obs rainfall
            txx_obs = np.delete(xx_obs, obj = j, axis = 1)
            fit_y, _ = self.gp_fit(rxx = txx_obs, mu_x = tmu_x, kxx = kxx + np.diag(tsn**2), kyx = kyx, mu_y = mu_x[j], kyy = kyy)
            KGE = self.kge(true_ = xx_obs[:, j], fit = fit_y)
            KGE2 = self.kge(true_ = xx_obs[:, j], fit = xx[:, j])
            score_kge.append([KGE, KGE2])
            score_cc.append([np.corrcoef(fit_y, xx_obs[:,j])[0, 1], np.corrcoef(xx[:, j], xx_obs[:,j])[0, 1]])
            print('the corrcoef for station {} is {:.2f} ({:.2f}), the KGE is {:.2f} ({:.2f})'.format(j, np.corrcoef(fit_y, xx_obs[:,j])[0, 1], np.corrcoef(xx[:, j], xx_obs[:,j])[0, 1], KGE, KGE2))

            f_est[:, j] = fit_y
            
            ri, ci = j // 4, j % 4
            ax[ri][ci].scatter(xx_obs[:,j], fit_y, color='blue', marker='o', label = 'interpolation')
            ax[ri][ci].scatter(xx_obs[:,j], xx[:, j], color='red', marker='D', label = 'raw simulation')
            tmpmin = min([np.min(xx_obs[:,j]), np.min(fit_y)])
            tmpmax = max([np.max(xx_obs[:,j]), np.max(fit_y)])
            ax[ri][ci].plot([tmpmin, tmpmax], [tmpmin, tmpmax], 'k--')
            ax[ri][ci].set_xlabel('obs rainfall [mm]')
            ax[ri][ci].set_ylabel('est rainfall [mm]')
            ax[ri][ci].set_title('(' + chr(ord('a') + j) + ')' + ' KGE = {:.2f} ({:.2f})'.format(KGE, KGE2))
    
        ax[3][2].axis('off')
        ax[3][3].axis('off')
        plt.tight_layout()
        fig.savefig('figs/0validation' + str(self.ts[0]) + '.pdf')
        plt.close(fig)
        pause = 1
        return np.array(score_kge), np.array(score_cc)
    
    
    def interpolate1(self, plot = 0, write_ = 0):

        xx = np.zeros([self.wrf_sta.shape[0]//12, self.wrf_sta.shape[1]])
        xx_obs = np.zeros([self.rain_sta.shape[0]//12, self.rain_sta.shape[1]])
        for i_ in self.ts:
            xx += self.wrf_sta[(i_-1)::12, :]
            xx_obs += self.rain_sta[(i_-1)::12, :]

        ref = {(i, j):idx for idx, (i, j) in enumerate(self.wrf_idx)}
        fyy = np.zeros([40, 120, 160])
        yvar = np.zeros([120, 160])
        mu_x = np.mean(xx, axis = 0)
        
        kxx = np.cov(xx.T, ddof = 1)
        xvar = np.zeros([120, 160])
        # i could ve just deleted the ref columns in rain_wrf_f but life is short 
        for i in range(120):
            for j in range(160):
                if (i, j) in ref: 
                    fyy[:, i, j] = xx_obs[:, ref[(i, j)]]
                    yvar[i, j] = self.sn[ref[(i, j)]] ** 2
                    continue
                tyy = np.zeros([self.rain_wrf.shape[0]//12, ])
                for i_ in self.ts:
                    tyy += self.rain_wrf[(i_-1)::12, i, j]
                # tyy = self.rain_wrf[0::12, i, j] + self.rain_wrf[1::12, i, j]
                kyx = np.cov(tyy.T, xx.T, ddof = 1)[:1, 1:]
                # if i == 60 and j == 80: 
                #     pause = 1
                kyy = np.cov(tyy.T, ddof = 1)
                mu_y = np.mean(tyy)
                # fit_y, _ = self.gp_fit(rxx = txx_obs, mu_x = tmu_x, kxx = kxx + np.diag(tsn**2), kyx = kyx, mu_y = mu_x[j], kyy = kyy)
                fit_y, fit_yvar = self.gp_fit(rxx = xx_obs, mu_x = mu_x, kxx = kxx + np.diag(self.sn**2), kyx = kyx, mu_y = mu_y, kyy = kyy)
                fyy[:, i, j] = fit_y
                yvar[i, j] = fit_yvar
                xvar[i, j] = kyy

        self.rainfall_interp = fyy
        self.rainfall_var = yvar

        if write_ == 1:
            fyy_flat = fyy.reshape([fyy.shape[0], -1])
            np.savetxt('output/interp' + str(self.ts[0]) +'.csv', fyy_flat)
            # fyy_flat2 = np.loadtxt('interp.csv')
            # fyy2 = fyy_flat2.reshape([fyy_flat2.shape[0], 120, 160])
            pause = 1

        if plot == 1:
            fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize=[12,8], subplot_kw={'projection': crs.PlateCarree()})
            # tmphigh = np.max([np.mean(self.rain_wrf, axis=0), np.mean(fyy, axis=0)])
            tmp_rain_wrf = np.zeros([self.rain_wrf.shape[0]//12, self.rain_wrf.shape[1], self.rain_wrf.shape[2]])
            for i_ in self.ts:
                tmp_rain_wrf += self.rain_wrf[(i_-1)::12, :, :]
            m_rain_wrf = np.multiply(tmp_rain_wrf, self.mask)
            tmpmax1 = np.nanmax(m_rain_wrf)
            m_fyy = np.multiply(fyy, self.mask)
            tmpmax2 = np.nanmax(m_fyy)
            tmpmax = np.max([tmpmax1, tmpmax2])
            self.map_plotter(ax[0,0], data = np.mean(tmp_rain_wrf, axis=0), show_sta = 1, vmax=tmpmax, vmin=0)
            ax[0,0].set_title('(a) simulation')
            self.map_plotter(ax[0,1], data = np.mean(fyy, axis=0), show_sta = 1, vmax=tmpmax, vmin=0)
            ax[0,1].set_title('(b) interpolation')
            self.map_plotter(ax[1,1], data = np.sqrt(yvar), show_sta = 1, vmax=360, vmin=0)
            ax[1,1].set_title('(d) 1 sigma intp')
            self.map_plotter(ax[1,0], data = np.sqrt(xvar), show_sta = 1, vmax=360, vmin=0)
            ax[1,0].set_title('(c) 1 sigma sim')
            plt.tight_layout()
            plt.savefig('figs/comp_' + str(self.ts[0]) + '.pdf')
            plt.close(fig)
        pause = 1

        return
    
    def interpolate2(self, plot_ = 0, write_ = 0):

        xx = np.zeros([self.wrf_sta.shape[0]//12, self.wrf_sta.shape[1]])
        xx_obs = np.zeros([self.rain_sta.shape[0]//12, self.rain_sta.shape[1]])
        
        ref = {(i, j):idx for idx, (i, j) in enumerate(self.wrf_idx)}
        data_array_full = []
        loc = []
        for i in range(120):
            for j in range(160):
                if (i, j) in ref:
                    continue
                data_array_full.append(self.rain_wrf[:, i, j])
                loc.append([i, j])
        data_array_full = np.array(data_array_full).T
        data_array = np.zeros([data_array_full.shape[0]//12, data_array_full.shape[1]])

        for i_ in self.ts:
            xx += self.wrf_sta[(i_-1)::12, :]
            xx_obs += self.rain_sta[(i_-1)::12, :]
            data_array += data_array_full[(i_-1)::12, :]

        kyy = np.cov(data_array.T, ddof = 1)
        kyx = np.cov(data_array.T, xx.T, ddof = 1)[:data_array.shape[1], data_array.shape[1]:]
        kxx = np.cov(xx.T, ddof = 1)
        mu_x = np.mean(xx, axis = 0)
        mu_y = np.mean(data_array, axis = 0)
        
        mu_yn = kyx @ np.linalg.inv(kxx + np.diag(self.sn**2)) @ (xx_obs - mu_x).T + mu_y[:, None]
        kyyn = kyy - kyx @ np.linalg.inv(kxx + np.diag(self.sn**2)) @ kyx.T

        fyy = np.zeros([40, 120, 160])
        yvar = np.zeros([120, 160])
        xvar = np.zeros([120, 160])
        for idx, (i, j) in enumerate(loc):
            fyy[:, i, j] = mu_yn[idx, :]
            yvar[i, j] = kyyn[idx, idx]
            xvar[i, j] = kyy[idx, idx]
        for idx, (i, j) in enumerate(self.wrf_idx):
            fyy[:, i, j] = xx_obs[:, idx]
            yvar[i, j] = self.sn[idx] ** 2
        mu_masked = np.multiply(fyy, self.mask)
        mu_masked[mu_masked < 0] = 0
        mu_avg = np.nanmean(mu_masked, axis = (1, 2))
        np.savetxt('mu_avg' + str(self.ts[0]) + '.csv', mu_avg)
        self.mu_post = mu_yn
        self.cov_post = kyyn
        self.loc = loc

        if write_ == 1:
            fyy_flat = fyy.reshape([fyy.shape[0], -1])
            np.savetxt('output/MC_MU_interp' + str(self.ts[0]) +'.csv', fyy_flat)
            # fyy_flat2 = np.loadtxt('interp.csv')
            # fyy2 = fyy_flat2.reshape([fyy_flat2.shape[0], 120, 160])
            pause = 1

        if plot_ == 1:
            fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize=[12,8], subplot_kw={'projection': crs.PlateCarree()})
            # tmphigh = np.max([np.mean(self.rain_wrf, axis=0), np.mean(fyy, axis=0)])
            tmp_rain_wrf = np.zeros([self.rain_wrf.shape[0]//12, self.rain_wrf.shape[1], self.rain_wrf.shape[2]])
            for i_ in self.ts:
                tmp_rain_wrf += self.rain_wrf[(i_-1)::12, :, :]

            m_rain_wrf = np.multiply(np.mean(tmp_rain_wrf, axis=0), self.mask)
            m_fyy = np.multiply(np.mean(fyy, axis=0), self.mask)
            tmpmax = np.max([np.nanmax(m_rain_wrf), np.nanmax(m_fyy)])
            tmpmin = np.min([np.nanmin(m_rain_wrf), np.nanmin(m_fyy)])

            post_sigma = np.sqrt(yvar)
            m_post_sigma = np.multiply(post_sigma, self.mask)
            sigma_min, sigma_max = np.nanmin(m_post_sigma), np.nanmax(m_post_sigma)
            print('min and max of post sigma are {:.2f} and {:.2f}'.format(sigma_min, sigma_max))

            tmpmax2 = ((tmpmax // 20) + 1) * 20

            self.map_plotter(ax[0,0], data = np.mean(tmp_rain_wrf, axis=0), show_sta = 1, vmax=tmpmax2, vmin=0)
            ax[0,0].set_title('(a) simulation')
            self.map_plotter(ax[0,1], data = np.mean(fyy, axis=0), show_sta = 1, vmax=tmpmax2, vmin=0)
            ax[0,1].set_title('(b) interpolation')
            self.map_plotter(ax[1,1], data = np.sqrt(yvar), show_sta = 1, vmax = 200, vmin = 0)
            ax[1,1].set_title('(d) 1 sigma intp')
            self.map_plotter(ax[1,0], data = np.sqrt(xvar), show_sta = 1, vmax = 200, vmin = 0)
            ax[1,0].set_title('(c) 1 sigma sim')
            plt.tight_layout()
            plt.savefig('figs/MC_comp_' + str(self.ts[0]) + '.pdf')
            plt.close(fig)
        pause = 1
        return
    
    def interpolate3(self, plot_ = 0, write_ = 0):
        
        tt = self.ts[0] - 1

        ref = {(i, j):idx for idx, (i, j) in enumerate(self.wrf_idx)}
        
        data_array = []
        loc = []
        xx = []
        for _, (i, j) in enumerate(self.wrf_idx):
            xx.append(self.cmip_rain[:, tt, i, j])

        for i in range(120):
            for j in range(160):
                if (i, j) in ref:
                    continue
                data_array.append(self.cmip_rain[:, tt, i, j])
                loc.append([i, j])
        xx = np.array(xx).T
        data_array = np.array(data_array).T
        xx_obs = self.rain_sta[tt::12, :]

        kyy = np.cov(data_array.T, ddof = 1)
        kyx = np.cov(data_array.T, xx.T, ddof = 1)[:data_array.shape[1], data_array.shape[1]:]
        kxx = np.cov(xx.T, ddof = 1)
        mu_x = np.mean(xx, axis = 0)
        mu_y = np.mean(data_array, axis = 0)
        
        mu_yn = kyx @ np.linalg.inv(kxx + np.diag(self.sn**2)) @ (xx_obs - mu_x).T + mu_y[:, None]
        kyyn = kyy - kyx @ np.linalg.inv(kxx + np.diag(self.sn**2)) @ kyx.T

        fyy = np.zeros([40, 120, 160])
        yvar = np.zeros([120, 160])
        xvar = np.zeros([120, 160])
        for idx, (i, j) in enumerate(loc):
            fyy[:, i, j] = mu_yn[idx, :]
            yvar[i, j] = kyyn[idx, idx]
            xvar[i, j] = kyy[idx, idx]
        for idx, (i, j) in enumerate(self.wrf_idx):
            fyy[:, i, j] = xx_obs[:, idx]
            yvar[i, j] = self.sn[idx] ** 2
        mu_masked = np.multiply(fyy, self.mask)
        mu_masked[mu_masked < 0] = 0
        mu_avg = np.nanmean(mu_masked, axis = (1, 2))
        np.savetxt('cmip_mu_avg' + str(self.ts[0]) + '.csv', mu_avg)
        self.mu_post = mu_yn
        self.cov_post = kyyn
        self.loc = loc

        if write_ == 1:
            fyy_flat = fyy.reshape([fyy.shape[0], -1])
            np.savetxt('output/cmip_MU_interp' + str(self.ts[0]) +'.csv', fyy_flat)
            # fyy_flat2 = np.loadtxt('interp.csv')
            # fyy2 = fyy_flat2.reshape([fyy_flat2.shape[0], 120, 160])
            pause = 1

        if plot_ == 1:
            fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize=[12,8], subplot_kw={'projection': crs.PlateCarree()})
            # tmphigh = np.max([np.mean(self.rain_wrf, axis=0), np.mean(fyy, axis=0)])
            tmp_rain_wrf = np.zeros([self.rain_wrf.shape[0]//12, self.rain_wrf.shape[1], self.rain_wrf.shape[2]])
            for i_ in self.ts:
                tmp_rain_wrf += self.rain_wrf[(i_-1)::12, :, :]
            self.map_plotter(ax[0,0], data = np.mean(tmp_rain_wrf, axis=0), show_sta = 1, color_high = -1)
            ax[0,0].set_title('(a) simulation')
            self.map_plotter(ax[0,1], data = np.mean(fyy, axis=0), show_sta = 1, color_high = -1)
            ax[0,1].set_title('(b) interpolation')
            self.map_plotter(ax[1,1], data = np.sqrt(yvar), show_sta = 1)
            ax[1,1].set_title('(d) 1 sigma intp')
            self.map_plotter(ax[1,0], data = np.sqrt(xvar), show_sta = 1)
            ax[1,0].set_title('(c) 1 sigma sim')
            plt.tight_layout()
            plt.savefig('figs/cmip_comp_' + str(self.ts[0]) + '.pdf')
            plt.close(fig)
        pause = 1
        return
    
    def MC_generate(self, nn = 1000):

        xx_obs = np.zeros([self.rain_sta.shape[0]//12, self.rain_sta.shape[1]])
        for i_ in self.ts:
            xx_obs += self.rain_sta[(i_-1)::12, :]
        
        sample_all = []
        obs_all = []
        for t in range(self.mu_post.shape[1]):
            mu = self.mu_post[:, t]

            t1 = time.time()
            mu_tf = tf.convert_to_tensor(mu, dtype=tf.float32)
            tmp_cov = self.cov_post + np.eye(self.cov_post.shape[0]) * np.mean(np.diag(self.cov_post)) * 1e-4
            cov_tf = tf.convert_to_tensor(tmp_cov, dtype=tf.float32)
            mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=mu_tf, covariance_matrix=cov_tf)
            data_sample1 = mvn.sample(sample_shape=(nn,))
            print('time used for sampling {} is {:.2f} sec'.format(t, time.time() - t1))

            data_sample = data_sample1.numpy()
            data_sample[data_sample < 0] = 0

            data_map = np.zeros((nn, 120, 160))
            mu_map = np.zeros((120, 160))
            for idx, (i, j) in enumerate(self.loc):
                data_map[:, i, j] = data_sample[:, idx]
                mu_map[i, j] = mu[idx]
            for idx, (i, j) in enumerate(self.wrf_idx):
                data_map[:, i, j] = xx_obs[t, idx]
                mu_map[i, j] = xx_obs[t, idx]
            
            mu_map = np.multiply(mu_map, self.mask)
            data_map = np.multiply(data_map, self.mask)
            r_avg = np.nanmean(data_map, axis = (1, 2))
            print('the median of the avg rainfall is {:.2f} ({:.2f}), obs = {:.2f}'.format(np.median(r_avg), np.nanmean(mu_map), np.mean(xx_obs[t, :])))
            sample_all.append(r_avg)
            obs_all.append(np.mean(xx_obs[t, :]))
        
        return np.array(sample_all), np.array(obs_all)
            
        

    def map_plotter(self, ax, data, show_sta = 1, vmax=-1, vmin=-1):

        ipcc_precip_colors = [(84,48,5), (110,68,15), (137,88,25), (164,108,35), (191,129,44), (200,148,79), (210,169,113), (220,189,147), (229,209,180), (239,228,215), (248,248,247), (216,232,231), (183,216,213), (151,200,195), (118,183,178), (85,167,160), (53,151,143), (39,128,119), (26,105,95), (13,82,71), (0,60,48)]

        c_wet = np.array(ipcc_precip_colors) / 255
        c_dry = np.array(ipcc_precip_colors[::-1]) / 255

        # create a cmap with 10 colors defined by c_wet
        cmap_wet = mpl.colors.ListedColormap(c_wet, name='precip_wet')
        cmap_dry = mpl.colors.ListedColormap(c_dry, name='precip_dry')

        self.coastline.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
        # tmplon = self.slice_(self.lon, buffer = 10)
        # tmplat = self.slice_(self.lat, buffer = 10)
        # tmpdata = self.slice_(data, buffer = 10)
        basemap = ax.contourf(self.lon, self.lat, data, levels=np.linspace(vmin, vmax, 21), transform=crs.PlateCarree(), cmap=cmap_wet, extend = 'max')
        for c in basemap.collections:
            c.set_edgecolor("face")
        if show_sta:
            sta_map = ax.scatter(self.wrf_loc[:,0], self.wrf_loc[:,1], s = 25, facecolors='k', marker='D')
        ax.set_extent([103.58, 104.12, 1.153, 1.502], crs=crs.PlateCarree())
        cbar = plt.colorbar(basemap, ax=ax, orientation='vertical', shrink=0.72)
        cbar.set_label('Rainfall [mm]')
        # cbar.mappable.set_clim(vmin=0, vmax=600) 
        gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator([103.5, 103.6, 103.7, 103.8, 103.9, 104.0, 104.1])
        gl.ylocator = mticker.FixedLocator([1.1, 1.2, 1.3, 1.4, 1.5])
        # ax.set_title(tit)
        pause = 1
        return
    
    def hovmoller_diagram(self, ax1, ax2, data):
        tmplon = self.slice_(self.lon, buffer = 10)
        tmplat = self.slice_(self.lat, buffer = 10)
        tmpdata = self.slice_(data, buffer = 10)
        time = np.array([range(1981, 2021, 1)])
        lon = np.linspace(np.min(tmplon), np.max(tmplon), tmplon.shape[1])
        X_mesh, Y_mesh = np.meshgrid(lon, time)
        Z = np.mean(tmpdata, axis = 1)
        # fig, ax = plt.subplots()
        basemap = ax2.contourf(X_mesh, Y_mesh, Z)
        ax2.set_xlabel('longitude')
        ax2.set_ylabel('time')
        ax2.yaxis.get_ticklocs(minor=True)
        ax2.minorticks_on() 
        cbar = plt.colorbar(basemap, ax=ax2, orientation='horizontal')
        cbar.set_label('rainfall [mm]')
        ax1.plot(lon, np.mean(Z, axis = 0))
        ax1.set_xlim(left = np.min(lon), right = np.max(lon))
        ax1.set_ylabel('rainfall [mm]')
        plt.setp(ax1.get_xticklabels(), visible=False)
        # fig.savefig('hov_diagram.pdf')
        pause = 1

        return


if __name__ == '__main__':
    # tmp = gp(target_season = [2])
    # tmp.sn_converge()
    # tmp.interpolate()
    # tmp.interpolate_un_range()

    # pause = 1
    # tmp = gp()
    # fig, ax = plt.subplots(figsize=[20,15], subplot_kw={'projection': crs.PlateCarree ()})
    # tmp.map_plotter(ax = ax, data = tmp.rain_wrf[0,:,:], show_sta = 1)
    # tmp.validation()
    # tmp.interpolate(write_ = 1)
    # fig, ax = plt.subplots(figsize=(8, 6), nrows=2, ncols=2, height_ratios=[1, 4]) 
    # tmp.hovmoller_diagram(ax1 = ax[0,1], ax2 = ax[1,1], data = tmp.rainfall_interp)
    # ax[0,1].set_title('(b) interpolation')
    # hov_data = np.zeros([40, 120, 160])
    # for i_ in tmp.ts:
    #     hov_data += tmp.rain_wrf[(i_-1)::12, :, :]
    # tmp.hovmoller_diagram(ax1 = ax[0,0], ax2 = ax[1,0], data = hov_data)
    # ax[0,0].set_title('(a) simulation')
    # plt.tight_layout()
    # fig.subplots_adjust(hspace=0)
    # fig.savefig('hov_diag.pdf')

    # kge0 = np.zeros([12, 14, 2])
    # cc0 = np.zeros([12, 14, 2])
    # for i in range(12):
    #     t1 = time.time()
    #     tmp = gp(target_season=[i+1])
    #     tmp.sn_converge()
    #     tkge, tcc = tmp.validation()
    #     kge0[i, :, :] = tkge
    #     cc0[i, :, :] = tcc
    #     tmp.interpolate(write_ = 1, plot = 1)
    #     print('month {} used {:.2f} sec'.format(i + 1, time.time() - t1))
    
    # kge1 = np.zeros([12, 14, 2])
    # cc1 = np.zeros([12, 14, 2])
    # for i in range(12):
    #     t1 = time.time()
    #     tmp = gp(target_season=[i+1])
    #     tmp.sn_converge()
    #     tkge, tcc = tmp.validation()
    #     kge1[i, :, :] = tkge
    #     cc1[i, :, :] = tcc
    #     tmp.interpolate2(write_ = 1, plot_ = 1)
    #     print('month {} used {:.2f} sec'.format(i + 1, time.time() - t1))

    # pause = 1
    # mpl.rcParams.update({'font.size': 14})
    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize = [14, 8])
    # ps = np.arange(0, 14*2, 2)
    # ax[0][0].boxplot(cc1[:,:,0], positions= ps - 0.3, widths = 0.4, patch_artist=True, boxprops=dict(facecolor='lightgreen'), flierprops={'markersize': 4, 'markerfacecolor': 'lightgreen'})
    # ax[0][0].boxplot(cc1[:,:,1], positions= ps + 0.3, widths = 0.4, flierprops={'markersize': 4})
    # ax[0][0].set_xticks(ps)
    # ax[0][0].set_xticklabels(np.arange(1,15))
    # ax[0][0].set_xlabel('Station Idx')
    # ax[0][0].set_ylabel('CC')
    # ax[0][0].set_title('(a)')

    # ax[0][1].boxplot(kge1[:,:,0], positions= ps - 0.3, widths = 0.4, patch_artist=True, boxprops=dict(facecolor='lightgreen'), flierprops={'markersize': 4, 'markerfacecolor': 'lightgreen'})
    # ax[0][1].boxplot(kge1[:,:,1], positions= ps + 0.3, widths = 0.4, flierprops={'markersize': 4})
    # ax[0][1].set_xticks(ps)
    # ax[0][1].set_xticklabels(np.arange(1,15))
    # ax[0][1].set_xlabel('Station Idx')
    # ax[0][1].set_ylabel('KGE')
    # ax[0][1].set_title('(a)')

    # ps = np.arange(0, 12*2, 2)
    # ax[1][0].boxplot(cc1[:,:,0].T, positions= ps - 0.3, widths = 0.4, patch_artist=True, boxprops=dict(facecolor='lightgreen'), flierprops={'markersize': 4, 'markerfacecolor': 'lightgreen'})
    # ax[1][0].boxplot(cc1[:,:,1].T, positions= ps + 0.3, widths = 0.4, flierprops={'markersize': 4})
    # ax[1][0].set_xticks(ps)
    # ax[1][0].set_xticklabels(np.arange(1,13))
    # ax[1][0].set_xlabel('Month')
    # ax[1][0].set_ylabel('CC')
    # ax[1][0].set_title('(c)')

    # ax[1][1].boxplot(kge1[:,:,0].T, positions= ps - 0.3, widths = 0.4, patch_artist=True, boxprops=dict(facecolor='lightgreen'), flierprops={'markersize': 4, 'markerfacecolor': 'lightgreen'})
    # ax[1][1].boxplot(kge1[:,:,1].T, positions= ps + 0.3, widths = 0.4, flierprops={'markersize': 4})
    # ax[1][1].set_xticks(ps)
    # ax[1][1].set_xticklabels(np.arange(1,13))
    # ax[1][1].set_xlabel('Month')
    # ax[1][1].set_ylabel('KGE')
    # ax[1][1].set_title('(b)')
    # fig.tight_layout()
    # fig.savefig('figs/box_test_one_column.pdf')
    # pause = 1

    selected_month = list(range(1, 13))
    for i in selected_month:
        t1 = time.time()
        tmp = gp(target_season=[i])
        # tmp.sn_converge()
        tmp.validation()
        # tmp.interpolate2(write_ = 0, plot_ = 1)
        # print('month {} used {:.2f} sec'.format(i, time.time() - t1))
        # rain_samples, rain_obs = tmp.MC_generate(nn = 1000)
        # np.savetxt('rain_samples' + str(i) + '.csv', rain_samples)
        # np.savetxt('rain_obs' + str(i) + '.csv', rain_obs)
        pause = 1
