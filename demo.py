import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt 

import geopandas as gpd
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
import cartopy.io.shapereader as shpreader
import matplotlib.ticker as mticker


class gp:
    def __init__(self,
                 sta ='data/sta_monthly.csv',
                 wrf = 'data/wrf_monthly.csv',
                 sta_spec = 'data/sta_lookup_new.csv',
                 wrf_loc = 'data/wrf_loc.csv',
                 geo = 'data/lonlat.txt'
                 ) -> None:
        
        self.sn = 0.01 # obs noise level
        
        self.rain_sta = np.loadtxt(sta)
        
        self.rain_wrf_f = np.loadtxt(wrf)
        self.rain_wrf = self.rain_wrf_f.reshape(self.rain_wrf_f.shape[0], 120, 160)

        self.sta_loc = np.genfromtxt(sta_spec, delimiter=',')[:, 2:]
        self.wrf_idx = np.loadtxt(wrf_loc)[:, :2].astype(int)
        self.wrf_loc = np.loadtxt(wrf_loc)[:, 2:]

        lonlat = np.loadtxt(geo)
        self.lon = lonlat[0, :].reshape([120, 160])
        self.lat = lonlat[1, :].reshape([120, 160])

        self.wrf_sta = np.array([self.rain_wrf[:, i, j] for _, (i, j) in enumerate(self.wrf_idx)]).T
        
        pause = 1

        self.coastline = gpd.read_file('/home/climate/xp53/for_plotting/cropped/coastlines.shp')
        return
    
    def kge(self, true_, fit):
        # the Kling-Gupta Efficiency
        r = np.corrcoef(true_, fit)[0, 1]
        alpha = np.std(fit, ddof = 1) / np.std(true_, ddof= 1)
        beta = np.mean(fit) / np.mean(true_)

        return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    def cov_est(self):

        return
    
    def gp_fit(self, fxx, mu_fx, kxy, kxx, mu_fy, kyy):
        tmpy = kxy @ np.linalg.inv(kxx) @ (fxx - mu_fx).T
        tmpy += mu_fy
        cov_y = kyy - kxy @ np.linalg.inv(kxx) @ kxy.T
        return tmpy.squeeze(), cov_y
    
    def validation(self):
        xx = self.wrf_sta[0::12, :] + self.wrf_sta[1::12, :]
        yy = self.rain_sta[0::12, :] + self.rain_sta[1::12, :]
        mu_x = np.mean(xx, axis = 0)
        # Kxx = np.cov(xx.T, ddof=1)
        
        y_est = np.zeros(yy.shape)
        for j in range(xx.shape[1]):
            txx = np.delete(xx, obj = j, axis = 1)
            t_mu = np.delete(mu_x, obj = j)
            tyy = np.delete(yy, obj = j, axis = 1)
            kxx = np.cov(txx.T, ddof = 1)
            kxy = np.cov(xx[:,j].T, txx.T, ddof = 1)[:1, 1:]
            kyy = np.cov(xx[:,j], ddof = 1)
            y, _ = self.gp_fit(fxx = tyy, mu_fx = t_mu, kxx = kxx, kxy = kxy, mu_fy = mu_x[j], kyy = kyy)
            KGE = self.kge(true_ = yy[:, j], fit = y)
            KGE2 = self.kge(true_ = yy[:, j], fit = xx[:, j])

            print('the corrcoef for station {} is {:.2f} ({:.2f}), the KGE is {:.2f} ({:.2f})'.format(j, np.corrcoef(y, yy[:,j])[0, 1], np.corrcoef(xx[:, j], yy[:,j])[0, 1], KGE, KGE2))

            y_est[:, j] = y
        pause = 1
        return
    
    
    def uncertainty(self):
        idx = 0
        mp = np.zeros([2, 120*160])
        yy = np.zeros([40, 120*160])
        xx = self.wrf_sta[0::12, :] + self.wrf_sta[1::12, :]

        ref = {(i, j):idx for idx, (i, j) in enumerate(self.wrf_idx)}
        fyy = np.zeros([40, 120, 160])
        ycov = np.zeros([120, 160])
        mu_x = np.mean(xx, axis = 0)
        fxx = self.rain_sta[0::12, :] + self.rain_sta[1::12, :]
        kxx = np.cov(xx.T, ddof = 1)
        # i could ve just deleted the ref columns in rain_wrf_f but life is short 
        for i in range(120):
            for j in range(160):
                if (i, j) in ref: 
                    fyy[:, i, j] = xx[:, ref[(i, j)]]
                    continue
                tyy = self.rain_wrf[0::12, i, j] + self.rain_wrf[1::12, i, j]
                kxy = np.cov(tyy.T, xx.T, ddof = 1)[:1, 1:]
                kyy = np.cov(tyy.T, ddof = 1)
                mu_y = np.mean(tyy)
                fit_y, fit_ycov = self.gp_fit(fxx = fxx, mu_fx = mu_x, kxx = kxx, kxy = kxy, mu_fy = mu_y, kyy = kyy)
                fyy[:, i, j] = fit_y
                ycov[i, j] = fit_ycov

                yy[:, idx] = tyy 
                mp[:, idx] = [i, j]
                idx += 1
        mp = mp[:, :idx].astype(int)
        yy = yy[:, :idx]

        fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=[24,7], subplot_kw={'projection': crs.PlateCarree()})
        self.map_plotter(ax[0], data = np.mean(self.rain_wrf, axis=0), show_sta = 1)
        self.map_plotter(ax[1], data = np.mean(fyy, axis=0), show_sta = 1)
        self.map_plotter(ax[2], data = np.sqrt(ycov), show_sta = 1)
        plt.tight_layout()
        plt.savefig('comp.pdf')
        pause = 1

        return
    
    def map_plotter(self, ax, data, show_sta = 0):
        self.coastline.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
        basemap = ax.contourf(self.lon, self.lat, data, transform=crs.PlateCarree(), cmap="jet")
        if show_sta:
            sta_map = ax.scatter(self.wrf_loc[:,0], self.wrf_loc[:,1], s = 50, facecolors='k', marker='D')
        ax.set_extent([np.min(self.lon), np.max(self.lon), np.min(self.lat), np.max(self.lat)], crs=crs.PlateCarree())
        cbar = plt.colorbar(basemap, ax=ax, orientation='vertical', shrink=.5)
        cbar.set_label('Rainfall [mm]')
        gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True, linewidth=1.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator([103.5, 103.7, 103.9, 104.1])
        gl.ylocator = mticker.FixedLocator([1.1, 1.3, 1.5])
        pause = 1
        return


if __name__ == '__main__':
    tmp = gp()
    # fig, ax = plt.subplots(figsize=[20,15], subplot_kw={'projection': crs.PlateCarree()})
    # tmp.map_plotter(ax = ax, data = tmp.rain_wrf[0,:,:], show_sta = 1)
    # tmp.validation()
    tmp.uncertainty()