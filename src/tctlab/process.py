'''These functions are used to analyze raw data from TCT output. An input datalocation is needed to run'''

import numpy as np
import matplotlib.pyplot as plt
import functs

plt.rcParams['figure.dpi'] = 150
import mplhep as hep
hep.style.use("LHCb")

## normal run: do either no-pos or regular depending on if the scan saved the position information

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

def matrices(datalocation, date, channel):
    ww, pp = [], []
    coords= np.loadtxt("{0}/scposition{1}.txt".format(datalocation, date), float)
    print(len(coords))
    xx= coords[:,0]
    yy= coords[:,1]
    for i in range(len(coords)):
        pp.append([int(xx[i]), int(yy[i])])
        ww.append([functs.import_waveform(channel, int(xx[i]), int(yy[i]))[3], functs.avg_waveform(channel, int(xx[i]), int(yy[i]))[0], functs.avg_waveform(channel, int(xx[i]), int(yy[i]))[1]])
        #progress_bar.next()
        print(i)

    ww= np.array(ww)
    np.save("{0}/scan_wfms{1}.npy".format(datalocation, channel), ww)


