import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Myriad Pro'

rain_obs = np.loadtxt('rain_obs12.csv')
mu_obs = np.loadtxt('mu_avg12.csv')
mu_obs2 = np.loadtxt('cmip_mu_avg12.csv') 
rain_samples = np.loadtxt('rain_samples12.csv')
rain_samples2 = np.loadtxt('cmip_rain_samples12.csv')

low = np.quantile(rain_samples, 0.025, axis=1)
high = np.quantile(rain_samples, 0.975, axis=1)
low2 = np.quantile(rain_samples2, 0.025, axis=1)
high2 = np.quantile(rain_samples2, 0.975, axis=1)
fig = plt.figure(figsize=(6, 4.5))
plt.scatter(rain_obs, mu_obs, s = 40, marker='>', color='red', edgecolors='none', label='obs', alpha=0.75, zorder=1)
plt.scatter(rain_obs, mu_obs2, s = 20, marker='o', color='blue', edgecolors='none', label='cmip', zorder=2)
for j in range(40):
    plt.plot([rain_obs[j], rain_obs[j]], [low[j], high[j]], color='red', linewidth=3, alpha=0.75, zorder=1)
    plt.plot([rain_obs[j], rain_obs[j]], [low2[j], high2[j]], color='blue', linewidth=1, zorder=2)
plt.axline((0, 0), (1, 1), color='black', linestyle='--', linewidth=1, zorder=0)
plt.gca().set_xlim(left=0)
plt.gca().set_ylim(bottom=0)
plt.title('(b) Dec')
plt.xlabel('Observed Rainfall [mm/day]')
plt.ylabel('Interpolated Rainfall [mm/day]')
fig.savefig('rain_obs_vs_sim12b.pdf')
pause = 1