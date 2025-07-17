import numpy as np
import matplotlib.pyplot as plt
import itertools, os
import functs, plotting, process

plt.rcParams['figure.dpi'] = 150
import mplhep as hep
hep.style.use("LHCb2")

from scipy.optimize import curve_fit
from cycler import cycler
from scipy.special import erf

def dev_frac(a1, a2, d1, d2):
    df1= (a2 / (a1 + a2)**2) * d1
    df2= (-1 * a1 / (a1 + a2)**2) * d2

    return np.sqrt(df1**2 + df2**2)

def y_fit(c1, c2, order, correction, datalocation, date, ymin):
    coords= np.loadtxt("{0}/scposition{1}.txt".format(datalocation, date))
    xx= coords[:,0]
    yy= coords[:,1]

    ampl1= np.loadtxt("{0}/amplitude_ch{1}.txt".format(datalocation, c1))
    dev1= np.loadtxt("{0}/amplitude_dev_ch{1}.txt".format(datalocation, c1))

    ampl2= np.loadtxt("{0}/amplitude_ch{1}.txt".format(datalocation, c2))
    dev2= np.loadtxt("{0}/amplitude_dev_ch{1}.txt".format(datalocation, c2))

    ampl1 = np.abs(ampl1)
    ampl2 = np.abs(ampl2)

    sig_frac, sig_dev, sig_y = [], [], []
    converted_y = []

    for i in range(len(ampl1)):
        converted_y.append((yy[i] - ymin)*2.5 - correction)

        s= ampl1[i] + ampl2[i]
        sig_frac.append(ampl1[i] / s)
        sig_dev.append(dev_frac(ampl1[i], ampl2[i], dev1[i], dev2[i]))

    ## cut to only include where we are not metalized; numbers assume we currect to start at the center of one pad and end at the center of the other
    cut_y, cut_frac, cut_dev = [], [], []
    for i in range(len(converted_y)):
        if converted_y[i] >= 135 and converted_y[i] <= 395:
            cut_y.append(converted_y[i])
            cut_frac.append(sig_frac[i])
            cut_dev.append(sig_dev[i])

    cut_frac= np.array(cut_frac)
    cut_y= np.array(cut_y)
    cut_dev= np.array(cut_dev)

    if order == 1:
        yparams, ycov = curve_fit(functs.line, cut_frac, cut_y, sigma=cut_dev)
    elif order == 2:
        yparams, ycov = curve_fit(functs.quad, cut_frac, cut_y, sigma=cut_dev)
    elif order == 3:
        yparams, ycov = curve_fit(functs.tri, cut_frac, cut_y, sigma=cut_dev)
    elif order == 4:
        yparams, ycov = curve_fit(functs.quart, cut_frac, cut_y, sigma=cut_dev)
    elif order == 5:
        yparams, ycov = curve_fit(functs.poly, cut_frac, cut_y, sigma=cut_dev)

    recon, redev = [], []

    for i in range(len(cut_y)):
        if order == 1:
            recon.append(functs.line(cut_frac[i], *yparams))
        elif order == 2:
            recon.append(functs.quad(cut_frac[i], *yparams))
        elif order == 3:
            recon.append(functs.tri(cut_frac[i], *yparams))
        elif order == 4:
            recon.append(functs.quart(cut_frac[i], *yparams))
        elif order == 5:
            recon.append(functs.poly(cut_frac[i], *yparams))
    
    dify = []
    for i in range(len(cut_y)):
        dify.append((recon[i]-cut_y[i]))
    dify = np.array(dify)

    return yparams, ycov, converted_y, cut_y, cut_frac, cut_dev, dify

def n_fit_y(c1, c2, order, correction, datalocation, date, ymin):
    coords = np.loadtxt("{0}/scposition{1}.txt".format(datalocation, date))
    xx = coords[:,0]
    yy = coords[:,1]

    ampl1 = np.load("{0}/scan_amplitude_{1}.npy".format(datalocation, channel)
