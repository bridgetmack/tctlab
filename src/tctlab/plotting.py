import numpy as np
import matplotlib.pyplot as plt
import itertools

import functs, move, process

plt.rcParams['figure.dpi'] = 150
import mplhep as hep
hep.style("LHCb2")

def map_amplitude_2d(channel, xx, yy, datalocation):
    x1 = np.unique(xx, axis=0)
    x2 = np.unique(yy, axis=0)

    ar = np.zeros([len(x2), len(x1)], dtype=int)
    cc = []

    for i in range(len(xx)):
        cc.append([xx[i], yy[i]])

    for i in range(len(yy)):
        for j in range(len(xx)):
            for k in range(len(cc)):
                if xx[j] == cc[k][0] and yy[i] == cc[k][1]:
                    x = np.where(x1 == xx[j])
                    y = np.where(x2 == yy[i])

                    ar[y,x] = k

    ampl = np.loadtxt("{0}/amplitude_ch{1}.txt".format(datalocation, channel))
    mamp = np.zeros([len(x2), len(x1)])

    for i in range(len(x2)):
        for j in range(len(x1)):
            n = ar[i,j]

            mamp[i,j] = ampl[n]

    plt.imshow(mamp, origin='lower')
    plt.title("Amplitude Map; Channel {}".format(channel))
    plt.colorbar(label="mV")
    plt.savefig("{0}/plots/map_amp_ch{1}.pdf".format(datalocation, channel))
    plt.clf()