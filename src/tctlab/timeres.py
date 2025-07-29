import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from scipy.stats import moyal

import process, functs, plotting, spatres

plt.rcParams['figure.dpi'] = 150
import mplhep as hep
hep.style.use("LHCb2")

def cfd(datalocation, date, channel):
    coords = np.loadtxt("{0}/scposition{1}.txt".format(datalocation, date))
    xx = coords[:,0]
    yy = coords[:,1]
 
    t = []
 
    for i in range(len(coords)):
        channel, x, y, wf_t, wf_v, wf_dev = functs.avg_waveform(datalocation, date, channel, int(xx[i]), int(yy[i]))
        wf_v = wf_v - np.mean(wf_v[1000:])
        wf_v = list(wf_v)

        if channel == 1: 
            ampl = np.max(wf_v)
            mindex = wf_v.index(ampl)
        else:
            ampl = np.min(wf_v)
            mindex = wf_v.index(ampl)
        #print(ampl, mindex)
            
        t_ampl = wf_t[mindex]
        ampl_dev = wf_dev[mindex]
        
        if channel == 1: 
            icmin = mindex - 100
            icmax = mindex + 100
        else:
            icmin = mindex - 15
            icmax = mindex + 15
 
        cut_t = wf_t[icmin:icmax]
        cut_v = wf_v[icmin:icmax]
        cut_dev = wf_dev[icmin:icmax]

        guess = [-10, 20, 0.5]

        params, cov = curve_fit(functs.land_func, cut_t, cut_v, sigma=cut_dev, maxfev=8000, p0=guess)
        #print(np.sqrt(np.diag(cov)))            

        tvals = np.linspace(15, wf_t[icmax], 1000)

        if i == 100:
            plt.errorbar(wf_t, wf_v, yerr=wf_dev, color='purple', ecolor='plum', capsize=0, marker=',')
            plt.plot(tvals, functs.land_func(tvals, *params), 'b')
            plt.show()
            
        threshold = 0.5 * ampl

        def function(x):
            return functs.land_func(x, *params) + ampl

        init_guess = 20
        t.append(float(fsolve(function, init_guess)))
    t = list(t)
    
    np.savetxt("{0}/times-ch{1}.txt".format(datalocation, channel), t)
    
    bb = np.linspace(16, 20, 40)
    plt.hist(t, color='green', edgecolor='black', bins=bb)
    plt.ylabel('ns')
    plt.show()

def laser(datalocation, date):
    return 0
