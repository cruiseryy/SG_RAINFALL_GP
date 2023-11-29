import numpy as np
import matplotlib.pyplot as plt

yy = np.loadtxt('data/cmip_yr_record.txt')
rain_wrf_f = np.loadtxt('data/wrf_monthly_cmip.txt')[-40:, :]
rain_wrf = rain_wrf_f.reshape(rain_wrf_f.shape[0], 12, 120, 160)
rain_wrf_b = rain_wrf.copy()
rain_wrf1 = np.zeros(rain_wrf.shape)
rain_wrf1[:, 0:6, :, :] = rain_wrf[:, 6:12, :, :]
rain_wrf1[:, 6:12, :, :] = rain_wrf[:, 0:6, :, :]



rand_test = np.random.randint(low = 0, high = 100, size = (44, 12, 120, 160))
rand_test_flat = rand_test.reshape(rand_test.shape[0], -1)
rand_test_re = rand_test_flat.reshape(rand_test.shape[0], 12, 120, 160)

if (rain_wrf1[:, 11, :, :] != rain_wrf_b[:, 5, :, :]).any():
    print('reshape is wrong')

pause = 1