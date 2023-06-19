import numpy as np


class gp_interp:
    def __init__(self,
                 file0 ='data/sta_monthly.csv',
                 filew = '/Users/cruiseryy/Documents/WORK/temp_data/nea_daily/wrf_sta_prcp.txt'
                 ) -> None:
        self.sn = 0.01 # obs noise level
        
        self.r0 = np.loadtxt(file0)
        
        self.rw = np.loadtxt (filew)

        pass


if __name__ == '__main__':
    tmp = gp_interp()