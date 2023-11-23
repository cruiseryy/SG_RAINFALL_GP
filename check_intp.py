import numpy as np

pre1 = 'output/interp'
pre2 = 'output/MU_interp'

suf = '.csv'

for i in range(1, 13):
    data0 = np.loadtxt(pre1 + str(i) + suf)
    data1 = np.loadtxt(pre2 + str(i) + suf)

    df = data0 - data1

    print(i, np.max(np.abs(df)))

pause = 1