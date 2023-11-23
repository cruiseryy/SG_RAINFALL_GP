import numpy as np

sta ='data/sta_monthly.csv'
rain_sta = np.loadtxt(sta)
rain_sta = rain_sta[0::12, :]
wrf_rain = np.loadtxt('output/interp1.csv')
wrf_rain = wrf_rain.reshape(wrf_rain.shape[0], 120, 160)
mask = np.loadtxt('mask.txt')
wrf_rain = np.multiply(wrf_rain, mask)
wrf_rain_avg = np.nanmean(wrf_rain, axis = (1, 2))
sta_rain_avg = np.mean(rain_sta, axis = 1)
pause = 1