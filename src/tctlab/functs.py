import numpy as np
import matplotlib.pyplot as plt
import itertools, os

plt.rcParams['figure.dpi']= 150
import mplhep as hep
hep.style.use("LHCb2")

def make_float(list):
    for i in range(len(list)):
        list[i]= float(list[i])
    return np.array(list)

def find_fwhm(x_data, y_data, polarity):
    v_trunkated, t_trunkated= [], []
    y_data= y_data - np.mean(y_data[-100:])

    ## you are going to need to fit this instead; this method will not work
    return 0

def integrate_waveform(t, v):
    R= 50
    units= 1e-12
    ctoe= 6.25e18

    N= len(t)
    h= (t[-1] - t[0]) / N
    s= 0.5 * v[0] + 0.5 * v[-1]
    for k in range(1, N):
        s += v[k]
    return h*s*units*ctoe/R

def integrate(t, v):
    N= len(t)
    h= (t[-1] - t[0]) / N
    s= 0.5 * v[0] + 0.5 * v[-1]
    for k in range(1, N):
        s += v[k]
    return h*s

def import_waveform(datalocation, date, channel, x, y):
    wf_t= np.loadtxt("{0}/chan{1}t{2}-x{3}-y{4}.txt".format(datalocation, channel, date, x, y), float)
    wf_vs= np.loadtxt("{0}/chan{1}v{2}-x{3}-y{4}.txt".format(datalocation, channel, date, x, y), float)

    return channel, x, y, wf_t, wf_vs

def amplitude(datalocation, date, channel, p):
    coords= np.loadtxt("{0}/scposition{1}.txt".format(datalocation, date))
    xx= coords[:,0]
    yy= coords[:,1]

    avg= []
    stdev= []

    for i in range(len(coords)):
        wfms= import_waveform(channel, int(xx[i]), int(yy[i]))[4]
        wfms= np.array(wfms)

        if p == -1 and channel != 1:
            ampl= np.min(wfms, axis=0)

            avg.append(np.mean(ampl))
            stdev.append(np.std(ampl))

        elif p == 1 or channel == 1:
            ampl= np.max(wfms, axis=0)

            avg.append(np.mean(ampl))
            stdev.append(np.std(ampl))

    np.savetxt("{0}/amplitude_ch{1}.txt".format(datalocation, channel), avg)
    np.savetxt("{0}/amplitude_dev_ch{1}.txt".format(datalocation, channel), stdev)

    return avg, stdev

def channel_number(channel, channel_tags, ch):
    if len(channel_tags) == 1:
        return ch[0]
    else:
        if channel == 2:
            return ch[0]
        elif channel == 3:
            return ch[1]
        elif channel == 4:
            return ch[2]