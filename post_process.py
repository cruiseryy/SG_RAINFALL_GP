import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Myriad Pro'

rain_obs = np.loadtxt('rain_obs12.csv')
mu_obs = np.loadtxt('mu_avg12.csv')
mu_obs2 = np.loadtxt('cmip_mu_avg12.csv') 
rain_samples = np.loadtxt('rain_samples12.csv')
rain_samples2 = np.loadtxt('cmip_rain_samples12.csv')

rr = np.array([218, 35, 94]) / 255
bb = np.array([54, 94, 136]) / 255

low = np.quantile(rain_samples, 0.025, axis=1)
high = np.quantile(rain_samples, 0.975, axis=1)
low2 = np.quantile(rain_samples2, 0.025, axis=1)
high2 = np.quantile(rain_samples2, 0.975, axis=1)
fig = plt.figure(figsize=(5, 4))
plt.scatter(rain_obs, mu_obs, s = 60, marker='>', color=rr, edgecolors='none', label='ERA5', zorder=1)
plt.scatter(rain_obs, mu_obs2, s = 30, marker='o', color=bb, edgecolors='none', label='CMIP6 SSP585', zorder=2)
plt.legend(loc='upper left')
for j in range(40):
    plt.plot([rain_obs[j], rain_obs[j]], [low[j], high[j]], color=rr, linewidth=5, alpha=0.5, zorder=1)
    plt.plot([rain_obs[j], rain_obs[j]], [low2[j], high2[j]], color=bb, linewidth=2, alpha=0.7, zorder=2)
plt.axline((0, 0), (1, 1), color='black', linestyle='--', linewidth=1, zorder=0)
plt.gca().set_xlim((80, 750))
plt.gca().set_ylim((80, 750))
plt.gca().set_aspect('equal', adjustable='box')
plt.title('(b) Dec')
plt.xlabel('Observed Rainfall [mm/day]')
plt.ylabel('Interpolated Rainfall [mm/day]')
fig.savefig('rain_obs_vs_sim12b.pdf')
pause = 1