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
    pos2 = np.unique(pos)

    np.savetxt(f"{datalocation}/scposition{date}.txt", pos)

def no_pos_matrices(datalocation, xmin, xmax, ymin, ymax, channel, date):
    xrang= np.linspace(xmin, xmax+1, int(xmax-xmin), dtype=int)
    yrang= np.linspace(ymin, ymax+1, int(ymax-ymin), dtype=int)

    wfms= []
    coords= []


    for i in range(len(xrang)):
        for j in range(len(yrang)):
            try:
                coords.append([int(functs.import_waveform(channel, xrang[i], yrang[j])[1]), int(functs.import_waveform(channel, xrang[i], yrang[j])[2])])
                wfms.append([functs.import_waveform(channel, xrang[i], yrang[j])[3], functs.avg_waveform(channel, xrang[i], yrang[j])[0], functs.avg_waveform(channel, xrang[i], yrang[j])[1]])
                    # print(len(wfms))
            except:
                continue

    np.savetxt("{0}/scposition{1}.txt".format(datalocation, date), coords)
    np.save("{0}/scan_wfms{1}.npy".format(datalocation, channel), wfms)

def matrices(datalocation, date, channel, nn):
    ww, pp = [], []
    coords= np.loadtxt("{0}/scposition{1}.txt".format(datalocation, date), float)
    print(len(coords))
    xx= coords[:,0]
    yy= coords[:,1]
    for i in range(len(coords)):
        pp.append([int(xx[i]), int(yy[i])])
        ww.append([functs.avg_waveform(datalocation, date, channel, int(xx[i]), int(yy[i]), nn)[3], functs.avg_waveform(datalocation, date, channel, int(xx[i]), int(yy[i]), nn)[4], functs.avg_waveform(datalocation, date, channel, int(xx[i]), int(yy[i]), nn)[5]])
        #progress_bar.next()
        print(i)

    ww= np.array(ww)
    np.save("{0}/scan_wfms{1}.npy".format(datalocation, channel), ww)


