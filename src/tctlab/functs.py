import numpy as np
import matplotlib.pyplot as plt
import itertools, os

from scipy.special import erf
from scipy.stats import landau

plt.rcParams['figure.dpi']= 150
import mplhep as hep
hep.style.use("LHCb2")

def make_float(list):
    for i in range(len(list)):
        list[i]= float(list[i])
    return np.array(list)

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

def avg_waveform(datalocation, date, channel, x, y, nn):
    channel, x, y, wf_t, wf_vs = import_waveform(datalocation, date, channel, x, y)

    wf_v = np.mean(wf_vs, axis=1)

    wf_v = wf_v - np.mean(wf_v[1000:])
    wf_v = list(wf_v)

    wf_stdev = np.std(wf_vs, axis=1) / np.sqrt(nn)

    return channel, x, y, wf_t, wf_v, wf_stdev

def amplitude(datalocation, date, channel, p, nn):
    coords= np.loadtxt("{0}/scposition{1}.txt".format(datalocation, date))
    xx= coords[:,0]
    yy= coords[:,1]

    noise = -4

    avg= []
    stdev= []

    for i in range(len(coords)):
        wfms= import_waveform(datalocation, date, channel, int(xx[i]), int(yy[i]))[4]
        wfms= np.array(wfms)

        if p == -1 and channel != 1:
            ampl= np.min(wfms, axis=0) - np.mean(wfms[1000:], axis=0)

            if np.mean(ampl) <= noise:
                avg.append(np.mean(ampl))
                stdev.append(np.std(ampl) / np.sqrt(nn))
            else: 
                avg.append(0)
                stdev.append(np.std(ampl) / np.sqrt(nn))

        elif p == 1 or channel == 1:
            ampl= np.max(wfms, axis=0) - np.mean(wfms[1000:], axis=0)

            avg.append(np.mean(ampl))
            stdev.append(np.std(ampl) / np.sqrt(nn))

    #plt.plot(yy, avg, '.')
    #plt.show()

    np.save("{0}/amplitudes_ch{1}.npy".format(datalocation, channel), ampl)
    np.savetxt("{0}/amplitude_ch{1}.txt".format(datalocation, channel), avg)
    np.savetxt("{0}/amplitude_dev_ch{1}.txt".format(datalocation, channel), stdev)

    return avg, stdev

def find_fwhm(x_data, y_data, polarity):
    v_trunkated, t_trunkated= [], []
    y_data= y_data - np.mean(y_data[-100:])

    ## you are going to need to fit this instead; this method will not work
    return 0

# def amplitude2(datalocation, date, channel):
#     coords = np.loadtxt("{0}/scposition{1}.txt".format(datalocation, date))
#     xx = coords[:,0]
#     yy = coords[:,1]

#     #wfms1 = np.load("{0}/scan_wfms{1}.npy".format(datalocation, c1))

#     ampl1 = []

#     for i in range(len(coords)):
#         wfms1 = import_waveform(datalocation, date, channel, int(xx[i]), int(yy[i]))[4]
#         ampl1.append(np.min(wfms1, axis=0))

#     np.save("{0}/scan_amplitudes_{1}.npy".format(datalocation, channel), ampl1)

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
        
def channel_center(channel, channel_tags, ch):
    cc = channel_number(channel, channel_tags, ch)

    if cc == 4:
        return [1648, 28962]
    elif cc == 3: 
        return [1648, 29162]
    elif cc == 12: 
        return [1848, 29162]

def poly(x, a, b, c, d, f, g):
    return (a + b*x + c*x**2 + d*x**3 + f*x**4 + g*x**5)

def line(x, a, b):
    return a + b*x

def quad(x, a, b, c):
    return a + b*x + c*x**2

def tri(x, a, b, c, d):
    return (a + b*x + c*x**2 + d*x**3)

def quart(x, a, b, c, d, f):
    return (a + b*x + c*x**2 + d*x**3 + f*x**4)

def erf_func(x, a, b, c, d):
    return a*erf(b*(x+c))+d

def land_func(x, a, mpv, wid):
    return a * landau.pdf(x, loc=mpv, scale=wid)

def find_land(x, ampl, a, mpv, wid):
    return a * landau.pdf(x, loc=mpv, scale=wid) + ampl

def gaus_func(x, a, b, c):
    return a * np.exp(-1*(x-b)*2 / c**2)
