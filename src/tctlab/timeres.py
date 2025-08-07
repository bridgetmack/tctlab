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

        ampl = np.min(wf_v)
        mindex = wf_v.index(ampl)
        #print(ampl, mindex)
            
        t_ampl = wf_t[mindex]
        ampl_dev = wf_dev[mindex]
        
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

def t_reco(c1, c2, datalocation, date, xmin, ymin):
    coords = np.loadtxt("{0}/scposition{1}.txt".format(datalocation, date))
    xx = coords[:,0]
    yy = coords[:,1]

    converted_x, converted_y = [], []
    for i in range(len(xx)):
        converted_x.append((xx[i] - xmin)*2.5)
        converted_y.append((yy[i] - ymin)*2.5)

    t1 = np.loadtxt("{0}/times-ch{1}.txt".format(datalocation, c1))
    t2 = np.loadtxt("{0}/times-ch{1}.txt".format(datalocation, c2))
    t1_dev = 0
    t2_dev = 0 ## actually figure out what to do for this

    ampl1 = np.loadtxt("{0}/amplitude_ch{1}.txt".format(datalocation, c1))
    ampl2 = np.loadtxt("{0}/amplitude_ch{1}.txt".format(datalocation, c2))
    ampl1_dev = np.loadtxt("{0}/amplitude_dev_ch{1}.txt".format(datalocation, c1))
    ampl2_dev = np.loadtxt("{0}/amplitude_dev_ch{1}.txt".format(datalocation, c2))

    treco = []
    cx, cy = [], []
    cuts = []

    for i in range(len(t1)): 
        tt = (ampl1[i]**2 * t1[i] + ampl2[i]**2 * t2[i]) / (ampl1[i]**2 + ampl2[i]**2)
        if tt < 100: 
            treco.append(tt)
            cy.append(converted_y[i])
            cx.append(converted_x[i])
        else: 
            cuts.append(i)

    #return converted_x, converted_y, treco

    plt.plot(cy, treco, '.')
    plt.ylim(bottom=0, top=40)
    plt.ylabel("Reconstructed time (ns)")
    plt.xlabel("Position Y (microns)")
    plt.axvspan(0, 105, color='grey', alpha=0.3)
    plt.axvspan(395, 500, color='grey', alpha=0.3)
    plt.show()
    plt.clf()

    plt.plot(cx, treco, '.')
    plt.ylim(bottom=0, top=40)
    plt.show()
    plt.clf()

    bb = np.linspace(15, 20, 50)
    plt.hist(treco, bins=bb)
    plt.show()
    
    print(np.mean(treco))
    print(np.std(treco))

    return cx, cy, treco, cuts

def laser(datalocation, date):
    coords = np.loadtxt("{0}/scposition{1}.txt".format(datalocation, date))
    xx = coords[:,0]
    yy = coords[:,1]

    t = []

    for i in range(len(coords)):
        channel, x, y, wf_t, wf_v, wf_dev = functs.avg_waveform(datalocation, date, 1, int(xx[i]), int(yy[i]))
        wf_v = wf_v - np.mean(wf_v[1000:])
        wf_v = list(wf_v)

        ampl = np.max(wf_v)
        mindex = wf_v.index(ampl)

        t_ampl = wf_t[mindex]
        ampl_dev = wf_dev[mindex]

        icmin = mindex - 50
        icmax = mindex

        #icmin= 0
        #icmax=-1

        cut_t = wf_t[icmin:icmax]
        cut_v = wf_v[icmin:icmax]
        cut_dev = wf_dev[icmin:icmax]

        guess = [1600, 15, 1]
        params, cov = curve_fit(functs.poly, cut_t, cut_v, sigma=cut_dev, maxfev=8000)

        tvals = np.linspace(16, wf_t[icmax], 1000)
 
        if i == 1: 
            plt.errorbar(wf_t, wf_v, yerr=wf_dev, color='purple', ecolor='plum', capsize=0, marker=',')
            plt.plot(tvals, functs.poly(tvals, *params), 'b')
            plt.show()

        threshold = 0.5*ampl

        def function(x):
            return functs.poly(x, *params) - ampl

        init_guess = 15
        t.append(float(fsolve(function, init_guess)))
    t = list(t)

    bb = np.linspace(14, 16, 100)
    plt.hist(t, bins=bb)
    plt.show()

    print(np.mean(t))

    return t

def difference(c1, c2, datalocation, date, xmin, ymin):
    t = laser(datalocation, date)
    cx, cy, treco, cuts = t_reco(c1, c2, datalocation, date, xmin, ymin)

    coords = np.loadtxt("{0}/scposition{1}.txt".format(datalocation, date))
    xx = coords[:,0]
    yy = coords[:,1]

    cut_x, cut_y = [], []
    cut_t = [] 
    
    for i in range(len(treco)):
        if i != cuts:
            cut_t.append(t[i])
    
    plt.plot(cut_t, treco, '.')
    plt.show()
    plt.clf()

    cut_t = np.array(cut_t)
    treco = np.array(treco)

    bb = np.linspace(0, 4, 40)
    dff = treco - cut_t
    plt.hist(dff, bins=bb)
    plt.show()
    plt.clf()

    plt.plot(cy, dff, 'm.')
    plt.axvspan(0, 105, color='grey', alpha=0.3)
    plt.axvspan(395, 500, color='grey', alpha=0.3)
    plt.show()
    plt.clf() 
