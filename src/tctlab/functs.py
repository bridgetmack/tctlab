import numpy as np
import matplotlib.pyplot as plt
import itertools, os

from scipy.special import erf
#from scipy.stats import landau

plt.rcParams['figure.dpi']= 150
import mplhep as hep
hep.style.use("LHCb2")

def make_float(list):
    for i in range(len(list)):
        list[i]= float(list[i])
    return np.array(list)

def integrate(t, v):
    N= len(t)
    h= (t[-1] - t[0]) / N
    s= 0.5 * v[0] + 0.5 * v[-1]
    for k in range(1, N):
        s += v[k]
    return h*s

class waveforms:
    def import_waveform(datalocation, date, channel, x, y):
        '''For input pointer returns the waveform and its matrix'''
        wf_t= np.loadtxt(f"{datalocation}/chan{channel}t{date}-x{x}-y{y}.txt", float)
        wf_vs= np.loadtxt(f"{datalocation}/chan{channel}v{date}-x{x}-y{y}.txt", float)

        return channel, x, y, wf_t, wf_vs

    def avg_waveform(datalocation, date, channel, x, y, nn):
        '''For input pointer, returns the average waveform and its features'''
        channel, x, y, wf_t, wf_vs = waveforms.import_waveform(datalocation, date, channel, x, y)
        print(len(wf_vs))
        print(len(wf_vs[1]))

        wf_v = np.mean(wf_vs, axis=1)
        wf_v = wf_v - np.mean(wf_v[1000:])
        wf_v = list(wf_v)
       
        wf_stdev = np.std(wf_vs, axis=1) / np.sqrt(nn)

        return channel, x, y, wf_t, wf_v, wf_stdev
        
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

    def amplitude(datalocation, date, channel, p, nn):
        coords= np.loadtxt(f"{datalocation}/scposition{date}.txt")
        xx= coords[:,0]
        yy= coords[:,1]

        avg= []
        stdev= []

        for i in range(len(coords)):
            wfms= waveforms.import_waveform(datalocation, date, channel, int(xx[i]), int(yy[i]))[4]
            wfms= np.array(wfms) - np.mean(wfms[1000:], axis=0)

            if p == -1 and channel != 1:
                ampl= np.abs(np.min(wfms, axis=0))
                avg.append(np.mean(ampl))
                stdev.append(np.std(ampl) / np.sqrt(nn))

                # need to add histograms for each point: will definitelty need to do something about the bins
                
            elif p == 1 or channel == 1:
                ampl= np.abs(np.max(wfms, axis=0))
                avg.append(np.mean(ampl))
                stdev.append(np.std(ampl) / np.sqrt(nn))
           
            np.save(f"{datalocation}/amplitudes_ch{channel}-x{int(xx[i])}-y{int(yy[i])}.npy", ampl)
        np.savetxt(f"{datalocation}/amplitude_ch{channel}.txt", avg)
        np.savetxt(f"{datalocation}/amplitude_dev_ch{channel}.txt", stdev)

        return avg, stdev

class bnl:
    def geometry_matrix():
        mmm = np.zeros([4,4], int)

        mmm[0, :] = [15, 14, 1, 16]
        mmm[1, :] = [13, 12, 3, 2]
        mmm[2, :] = [11, 10, 4, 5]
        mmm[3, :] = [9, 8, 7, 6]

        return mmm

    def convert_coords(datalocation, date):
        coords = np.loadtxt(f"{datalocation}/scposition{date}.txt")
        xx = coords[:,0]
        yy = coords[:,1]

        xx = ( np.array(xx) - 1370 ) * 2.5
        yy = ( np.array(yy) - 28730-4) * 2.5

        return xx, yy

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
        cc = bnl.channel_number(channel, channel_tags, ch)

        if cc == 1:
            return [750, 1750]
        elif cc == 2:
            return [250, 1250]
        elif cc == 3:
            return [750, 1250]
        elif cc == 4:
            return [750, 750]
        elif cc == 5:
            return [250, 750]
        elif cc == 6:
            return [250, 250]
        elif cc == 7:
            return [750, 250]
        elif cc == 8:
            return [1250, 250]
        elif cc == 9:
            return [1750, 250]
        elif cc == 10:
            return [1250, 750]
        elif cc == 11:
            return [1750, 750]
        elif cc == 12:
            return [1250, 1250]
        elif cc == 13:
            return [1750, 1250]
        elif cc == 14:
            return [1250, 1750]
        elif cc == 15:
            return [1750, 1750]
        elif cc == 16:
            return [0, 0]

class fits:
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
