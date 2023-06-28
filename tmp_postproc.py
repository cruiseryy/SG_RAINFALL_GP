import numpy as np

file = 'interp'
data = np.zeros([480, 120*160])
for i in range(1, 13):
    tmpdata = np.loadtxt(file + str(i) + '.csv')
    data[(i-1)::12, :] = tmpdata
    pause = 1
np.savetxt('gp_interp.csv', data)