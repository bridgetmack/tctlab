'''These functions are used to analyze raw data from TCT output. An input datalocation is needed to run'''

import numpy as np
import matplotlib.pyplot as plt
import functs

plt.rcParams['figure.dpi'] = 150
import mplhep as hep
hep.style.use("LHCb2")

## normal run: do either no-pos or regular depending on if the scan saved the position information

import os

def recover_pos(datalocation, date):
    os.system(f"ls {datalocation}/chan1t*.txt > {datalocation}/files.txt")
    f = np.loadtxt(f"{datalocation}/files.txt", dtype=str)

    pos = []
    for i in range(len(f)):
        filename= f[i].split("-")
        try:
            #print(filename)
            xi = filename[5]
            yi = filename[6]

            xi = xi.replace(xi[0], "", 1)
            yi = yi.replace(yi[0], "", 1) 
            yi = yi.replace(yi[-1], "", 1)
            yi = yi.replace(yi[-1], "", 1)
            yi = yi.replace(yi[-1], "", 1)
            yi = yi.replace(yi[-1], "", 1)
        
            xi = int(xi)
            yi = int(yi)

            print(xi, yi)
            pos.append([xi, yi])
        except:
            print(filename)

    np.savetxt(f"{datalocation}/scposition{date}.txt", pos)

def matrices(datalocation, date, channel, nn):
    ww, pp = [], []
    coords= np.loadtxt(f"{datalocation}/scposition{date}.txt", float)
    print(len(coords))
    xx= coords[:,0]
    yy= coords[:,1]
    for i in range(len(coords)):
        pp.append([int(xx[i]), int(yy[i])])
        ww.append([functs.waveforms.avg_waveform(datalocation, date, channel, int(xx[i]), int(yy[i]), nn)[3], functs.waveforms.avg_waveform(datalocation, date, channel, int(xx[i]), int(yy[i]), nn)[4], functs.waveforms.avg_waveform(datalocation, date, channel, int(xx[i]), int(yy[i]), nn)[5]])
        print(i)
    print(ww)
    ww= np.array(ww)
    
    np.save(f"{datalocation}/scan_wfms{channel}.npy", ww)

def single_pt(datalocation, date, channel, nn):
    ww, pp = [], []
    coords= np.loadtxt(f"{datalocation}/scposition{date}.txt", float)
    xx = int(coords[0])
    yy = int(coords[1])
    
    ww.append([functs.waveforms.avg_waveform(datalocation, date, channel, xx, yy, nn)[3], functs.waveforms.avg_waveform(datalocation, date, channel, xx, yy, nn)[4], functs.waveforms.avg_waveform(datalocation, date, channel, xx, yy, nn)[5]])
    
    #ww= np.array(ww)
    np.save(f"{datalocation}/scan_wfms{channel}.npy", ww)
