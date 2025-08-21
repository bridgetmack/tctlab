import numpy as np
import matplotlib.pyplot as plt
import itertools

import functs, process, spatres

plt.rcParams['figure.dpi'] = 150
import mplhep as hep
hep.style.use("LHCb2")

def plot_all_wfms(channel, datalocation, date, channel_tags, ch):
    wfms= np.load("{0}/scan_wfms{1}.npy".format(datalocation, channel))
    coords= np.loadtxt("{0}/scposition{1}.txt".format(datalocation, date))

    for i in range(len(coords)):
        plt.plot(wfms[i][0], wfms[i][1], label="({0}, {1})".format(int(coords[:,0][i]), int(coords[:,1][i])))
    plt.legend(loc='right', fontsize='xx-small')
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (mV)')
    plt.title('Scan Waveforms; Channel {}'.format(functs.channel_number(channel, channel_tags, ch)))
    plt.savefig("{0}/plots/all_wfm_c{1}".format(datalocation, channel))
    #plt.show()

    for i in range(len(coords)):
        plt.plot(wfms[i][0][:500], wfms[i][1][:500], label="({0}, {1})".format(int(coords[:,0][i]), int(coords[:,1][i])))
    plt.legend(loc='right', fontsize='xx-small')
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (mV)')
    plt.title('Scan Waveforms; Channel {}'.format(functs.channel_number(channel), channel_tags, ch))
    plt.savefig("{0}/plots/all_wfm_c{1}_zoom".format(datalocation, channel))
    #plt.show()

def plot_sep_wfms(channel, datalocation, date, channel_tags, ch):
    wfms= np.load("{0}/scan_wfms{1}.npy".format(datalocation, channel))
    coords= np.loadtxt("{0}/scposition{1}.txt".format(datalocation, date))

    for i in range(len(coords)):
        if channel == 2:
            plt.errorbar(wfms[i][0], wfms[i][1], yerr=wfms[i][2], color="purple",  ecolor='plum', capsize=0, label="({0}, {1})".format(int(coords[:,0][i]), int(coords[:,1][i])))
        elif channel == 3:
            plt.errorbar(wfms[i][0], wfms[i][1], yerr=wfms[i][2], color="teal",  ecolor='paleturquoise', capsize=0, label="({0}, {1})".format(int(coords[:,0][i]), int(coords[:,1][i])))
        elif channel == 4:
            plt.errorbar(wfms[i][0], wfms[i][1], yerr=wfms[i][2], color="green",  ecolor='palegreen', capsize=0, label="({0}, {1})".format(int(coords[:,0][i]), int(coords[:,1][i])))
        plt.legend()
        plt.title("Average Waveform; Channel {}".format(functs.channel_number(channel, channel_tags, ch)))
        plt.xlabel('Time (ns)')
        plt.ylabel('Voltage (mV)')
        plt.savefig("{0}/plots/avg_wfm_ch{1}-x{2}-y{3}.pdf".format(datalocation, channel, int(coords[:,0][i]), int(coords[:,1][i])))

        plt.clf()

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



def plot_y_fit(c1, c2, order, correction, datalocation, date, ymin, channel_tags, ch):
    yparams, ycov, sig_frac, sig_dev, converted_y, cut_y, cut_frac, cut_dev, dify = spatres.y_fit(c1, c2, order, correction, datalocation, date, ymin)

    yfrac = np.linspace(min(cut_frac), max(cut_frac), 1000)
    
    plt.errorbar(converted_y, sig_frac, yerr=sig_dev, linestyle='none', marker='.', ecolor='plum', color='purple')
    if order == 1:
        plt.plot(functs.line(yfrac, *yparams), yfrac, '-', label="polynomial order: {}".format(order))
    elif order == 2:
        plt.plot(functs.quad(yfrac, *yparams), yfrac, '-', label="polynomial order: {}".format(order))
    elif order == 3:
        plt.plot(functs.tri(yfrac, *yparams), yfrac, '-', label="polynomial order: {}".format(order))
    elif order == 4:
        plt.plot(functs.quart(yfrac, *yparams), yfrac, '-', label="polynomial order: {}".format(order))
    elif order == 5:
        plt.plot(functs.poly(yfrac, *yparams), yfrac, '-', label="polynomial order: {}".format(order))
    plt.axvspan(0, 105, color='grey', alpha=0.3)
    plt.axvspan(395, 500, color='grey', alpha=0.3)
    plt.xlabel("Y position (microns)")
    plt.ylabel("Ampliude Fraction")
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.legend()
    plt.title("Amplitude Fraction vs Y; Ch {0} against Ch {1}".format(functs.channel_number(c1, channel_tags, ch), functs.channel_number(c2, channel_tags, ch)))
    plt.savefig("{0}/plots/ampl-frac-y2-ch{1}-ch{2}-order{3}".format(datalocation, c1, c2, order))
    plt.show()
    plt.clf()

    plt.plot(cut_y, dify, '.', label='polynomial order {}'.format(order))
    plt.ylabel("Spatial Resolution (microns)")
    plt.xlabel("True Y (microns)")
    plt.ylim(bottom=-5, top=5)
    plt.axvspan(0, 105, color='grey', alpha=0.3)
    plt.axvspan(395, 500, color='grey', alpha=0.3)
    #plt.legend()
    plt.show()
    plt.clf()

    plt.plot(cut_frac, dify, '.', label='polynomial order {}'.format(order))
    plt.ylabel("Spatial Resolution (microns)")
    plt.xlabel("Signal Fraction")
    plt.ylim(bottom=-5, top=5)
    #plt.legend()
    plt.show()
    plt.clf()

    bb = np.linspace(-5, 5, 100) ##100 nm bins
    plt.hist(dify, color='purple', edgecolor='black', bins=bb, label='mean = {}\n$\sigma$ = {}'.format(round(np.mean(dify), 3), round(np.std(dify),3)))
    plt.legend()
    plt.show()
    plt.clf()

def plot_cfd(datalocation, channel):
    t = np.loadtxt("{0}/times-ch{1}.txt".format(datalocation, channel), float)

    if channel == 1:
        bb = np.linspace(18, 22, 400)
    else:
        bb = np.linspace(16, 20, 40)
    plt.hist(t, color='lightblue', edgecolor='steelblue', bins=bb)
    plt.xlabel('ns')
    plt.show()

def plot_treco():
    return 0
