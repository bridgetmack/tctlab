import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import moyal

import process, functs, plotting, spatres

plt.rcParams['figure.dpi'] = 150
import mplhep as hep
hep.style.use("LHCb2")

def cfd(datalocation, date):
    coords = np.loadtxt("{0}/scposition{1}.txt".format(datalocation, date))
    xx = coords[:,0]
    yy = coords[:,1]
  
    for i in range(len(coords)):
        if i == 1:
            channel, x, y, wf_t, wf_v, wf_dev = functs.avg_waveform(datalocation, date, 2, int(xx[i]), int(yy[i]))

            wf_v = list(wf_v)

            ampl = np.min(wf_v)
            mindex = wf_v.index(ampl)
            #print(mindex)
            
            t_ampl = wf_t[mindex]
            ampl_dev = wf_dev[mindex]

            icmin = mindex - 100
            icmax = mindex + 100
 
            cut_t = wf_t[icmin:icmax]
            cut_v = wf_v[icmin:icmax]
            cut_dev = wf_dev[icmin:icmax]

            p0 = [

            params, cov = curve_fit(functs.land_func, cut_t, cut_v, sigma=cut_dev)

            tvals = np.linspace(min(cut_t), max(cut_t), 1000)

            plt.errorbar(wf_t[(mindex-50):(mindex+50)], wf_v[(mindex-50):(mindex+50)], yerr=wf_dev[(mindex-50):(mindex+50)], color='purple', ecolor='plum', capsize=0)
            plt.plot(cut_t, functs.land_func(cut_t, *params), 'b')
            plt.show()
            
            
