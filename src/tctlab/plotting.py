import numpy as np
import matplotlib.pyplot as plt
import itertools

import functs, process, spatres

plt.rcParams['figure.dpi'] = 150
import mplhep as hep
hep.style.use("LHCb2")

from scipy.stats import norm
import matplotlib.mlab as mlab

class plot_wfm:
    def plot_all_individual(channel, datalocation, plotlocation, date, x, y):
        channel, x, y, t, vs = functs.waveforms.import_waveform(datalocation, date, channel, x, y)
        
        vv = np.transpose(vs)
        for i in range(len(vv)):
            plt.plot(t, vv[i])
        
        
        plt.savefig(f"{plotlocation}/point-plot-{x}-{y}.pdf")
        plt.clf()

    def plot_individual(channel, event, datalocation, plotlocation, date, channel_tags, ch, ped):
        coords= np.loadtxt(f"{datalocation}/scposition{date}.txt")
        xx, yy = coords[:,0], coords[:,1]
        
        for i in range(len(xx)):
            x, y = int(xx[i]), int(yy[i])
            
            t = functs.waveforms.import_waveform(datalocation, date, channel, x, y)[3]
            
            if ped == True:
                vs = functs.waveforms.apply_dynamic_pedestal(datalocation, date, channel, x, y)
            else:
                vs = functs.waveforms.remove_baseline(datalocation, date, channel, x, y)
            
            vv = np.transpose(vs)
            plt.plot(t, vv[event])
            plt.title(f"Waveform for Channel {functs.bnl.channel_number(channel, channel_tags, ch)}; Event {event}; Position ({x}, {y}")    
            plt.savefig(f"{plotlocation}/point-plot{event}-ch{channel}-{x}-{y}.pdf")
            plt.clf()

    def plot_all_wfms(channel, datalocation, plotlocation, date, channel_tags, ch):
        '''Plots all waveforms for a specific channel on the same graph'''
        wfms= np.load(f"{datalocation}/scan_wfms{channel}.npy")
        print(wfms)
        coords= np.loadtxt(f"{datalocation}/scposition{date}.txt")

        for i in range(len(coords)):
            plt.plot(wfms[i][0], wfms[i][1], label=f"({int(coords[:,0][i])}, {int(coords[:,1][i])})")
        plt.xlabel('Time (ns)')
        plt.ylabel('Voltage (mV)')
        plt.title(f'Scan Waveforms; Channel {functs.bnl.channel_number(channel, channel_tags, ch)}')
        plt.savefig(f"{plotlocation}/all_wfm_c{channel}.pdf")
        plt.clf()

        ## zoomed in version:
        for i in range(len(coords)):
            plt.plot(wfms[i][0][:500], wfms[i][1][:500], label=f"({int(coords[:,0][i])}, {int(coords[:,1][i])})")
        plt.xlabel('Time (ns)')
        plt.ylabel('Voltage (mV)')
        plt.title(f'Scan Waveforms; Channel {functs.bnl.channel_number(channel, channel_tags, ch)}')
        plt.savefig(f"{plotlocation}/all_wfm_c{channel}_zoom.pdf")
        plt.clf()

    def plot_sep_wfms(channel, datalocation, plotlocation, date, channel_tags, ch):
        '''Plots all waveforms on separate axes'''
        wfms= np.load(f"{datalocation}/scan_wfms{channel}.npy")
        coords= np.loadtxt(f"{datalocation}/scposition{date}.txt")

        for i in range(len(coords)):
            #if channel == 2:
            plt.errorbar(wfms[i][0], wfms[i][1], yerr=wfms[i][2], color="purple",  ecolor='plum', capsize=0, label=f"({int(coords[:,0][i])}, {int(coords[:,1][i])})")
            #elif channel == 3:
                #plt.errorbar(wfms[i][0], wfms[i][1], yerr=wfms[i][2], color="teal",  ecolor='paleturquoise', capsize=0, label=f"({int(coords[:,0][i])}, {int(coords[:,1][i])})")
            #elif channel == 4:
                #plt.errorbar(wfms[i][0], wfms[i][1], yerr=wfms[i][2], color="green",  ecolor='palegreen', capsize=0, label=f"({int(coords[:,0][i])}, {int(coords[:,1][i])})")
            plt.legend()
            plt.title(f"Average Waveform; Channel {functs.bnl.channel_number(channel, channel_tags, ch)}")
            plt.xlabel('Time (ns)')
            plt.ylabel('Voltage (mV)')
            plt.savefig(f"{plotlocation}/avg_wfm_ch{channel}-x{int(coords[:,0][i])}-y{int(coords[:,1][i])}.pdf")
            plt.clf()

class mapping:
    def map_matrix(datalocation, date):
        '''returns matrix to fill to make maps'''
        coords = np.loadtxt(f"{datalocation}/scposition{date}.txt")
        xx, yy = coords[:,0], coords[:,1]
        x1, x2 = np.unique(xx, axis=0), np.unique(yy, axis=0)
        
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
        return ar, x1, x2
    
    def map_amplitude_2d(channel, datalocation, plotlocation, date, channel_tags, ch):
        '''Plots in 2d the map of average amplitudes -- need to check geometry'''
        # coords = np.loadtxt(f"{datalocation}/scposition{date}.txt")
        # xx = coords[:,0]
        # yy = coords[:,1]

        # x1 = np.unique(xx, axis=0)
        # x2 = np.unique(yy, axis=0)

        # ar = np.zeros([len(x2), len(x1)], dtype=int)
        # cc = []

        # for i in range(len(xx)):
            # cc.append([xx[i], yy[i]])

        # for i in range(len(yy)):
            # for j in range(len(xx)):
                # for k in range(len(cc)):
                    # if xx[j] == cc[k][0] and yy[i] == cc[k][1]:
                        # x = np.where(x1 == xx[j])
                        # y = np.where(x2 == yy[i])

                        # ar[y,x] = k
        ar, x1, x2 = mapping.map_matrix(datalocation, date)
        ampl = np.loadtxt(f"{datalocation}/amplitude_ch{channel}.txt")
        mamp = np.zeros([len(x2), len(x1)])

        for i in range(len(x2)):
            for j in range(len(x1)):
                n = ar[i,j]

                mamp[i,j] = ampl[n]

        plt.imshow(mamp, origin='lower')
        plt.title(f"Amplitude Map; Channel {functs.bnl.channel_number(channel, channel_tags, ch)}")
        plt.colorbar(label="mV")
        plt.savefig(f"{plotlocation}/map_amp_ch{channel}.pdf")
        plt.clf()
        
    def map_spatres(datalocation, plotlocation, date, nn,  channel_tags, ch):
        ar, x1, x2 = mapping.map_matrix(datalocation, date)
        mux, muy, sig_x, sig_y, sig_r = spatres.weighted_average(datalocation, date, nn, channel_tags, ch)
        xmap = np.zeros([len(x2), len(x1)])
        ymap = np.zeros([len(x2), len(x1)])
        rmap = np.zeros([len(x2), len(x1)])
        
        for i in range(len(x2)):
            for j in range(len(x1)):
                n = ar[i,j]
                
                xmap[i,j] = sig_x[n]
                ymap[i,j] = sig_y[n]
                rmap[i,j] = sig_r[n]
        
        plt.imshow(xmap, origin='lower')
        plt.title(f"Sig X")
        plt.colorbar(label="um")
        plt.savefig(f"{plotlocation}/map_sig_x.pdf")
        plt.clf()
        
        plt.imshow(ymap, origin='lower')
        plt.title(f"Sig Y")
        plt.colorbar(label="um")
        plt.savefig(f"{plotlocation}/map_sig_y.pdf")
        plt.clf()
        
        plt.imshow(rmap, origin='lower')
        plt.title(f"Sig R")
        plt.colorbar(label="um")
        plt.savefig(f"{plotlocation}/map_sig_r.pdf")
        plt.clf()

    def spat_diff_map(datalocation, plotlocation, date, nn, channel_tags, ch):
        ar, x1, x2 = mapping.map_matrix(datalocation, date)
        mux, muy, sig_x, sig_y, sig_r = spatres.weighted_average(datalocation, date, nn, channel_tags, ch)
        
        event = 10
        
        for i in range(len(xx)):
            wax = np.loadtxt(f"{datalocation}/wax-x{int(xx[i])}-y{int(yy[i])}-board0.txt")
            way = np.loadtxt(f"{datalocation}/way-x{int(xx[i])}-y{int(yy[i])}-board0.txt")

class plot_apml:            
    def plot_avg_ampl(channel, datalocation, plotlocation, date, channel_tags, ch):
        ampl = np.loadtxt(f"{datalocation}/amplitude_ch{channel}.txt")
        dev = np.loadtxt(f"{datalocation}/amplitude_dev_ch{channel}.txt")
      
        coords = np.loadtxt(f"{datalocation}/scposition{date}.txt")
        xx = coords[:,0]
        yy = coords[:,1]

        cx, cy = functs.bnl.convert_coords(datalocation, date)
        
        plt.errorbar(cx, ampl, yerr=dev, linestyle='none', marker='.', color='purple', ecolor='plum', label=f"Channel {functs.bnl.channel_number(channel, channel_tags, ch)}")
        plt.legend()
        plt.title('Amplitude vs X')
        plt.xlabel('X Position (microns)')
        plt.ylabel('Amplitude (mV)')
        plt.axvspan(145, 355, color='grey', alpha=0.3)
        plt.axvspan(645, 855, color='grey', alpha=0.3)
        plt.axvspan(1145, 1355, color='grey', alpha=0.3)
        plt.axvspan(1645, 1845, color='grey', alpha=0.3)
        plt.savefig(f"{plotlocation}/ampl-x-{channel}.pdf")
        plt.clf()

        plt.errorbar(cy, ampl, yerr=dev, label=f"Channel {functs.bnl.channel_number(channel, channel_tags, ch)}", linestyle='none', marker='.', color='purple', ecolor='plum')
        plt.legend()
        plt.title('Amplitude vs Y')
        plt.xlabel('Y Position (microns)')
        plt.ylabel('Amplitude (mV)')
        plt.axvspan(145, 355, color='grey', alpha=0.3)
        plt.axvspan(645, 855, color='grey', alpha=0.3)
        plt.axvspan(1145, 1355, color='grey', alpha=0.3)
        plt.axvspan(1645, 1855, color='grey', alpha=0.3)
        plt.savefig(f"{plotlocation}/ampl-y-{channel}.pdf")
        plt.clf()

    def plot_all_avg_ampl(datalocation, plotlocation, date, channel_tags, ch):
        coords = np.loadtxt(f"{datalocation}/scposition{date}.txt")
        xx = coords[:,0]
        yy = coords[:,1]

        cx, cy = functs.bnl.convert_coords(datalocation, date)
        
        totx, toty = np.zeros(len(xx)), np.zeros(len(yy))
        
        for i in range(len(channel_tags)):
            ampl = np.loadtxt(f"{datalocation}/amplitude_ch{channel_tags[i]}.txt")
            dev = np.loadtxt(f"{datalocation}/amplitude_dev_ch{channel_tags[i]}.txt")
            
            totx += ampl
            
            if channel_tags[i] == 2:
                plt.errorbar(cx, ampl, yerr=dev, linestyle='none', marker='.', color='purple', ecolor='plum', label=f"Channel {functs.bnl.channel_number(2, channel_tags, ch)}")
            elif channel_tags[i] == 3:
                plt.errorbar(cx, ampl, yerr=dev, linestyle='none', marker='.', color='teal', ecolor='paleturquoise', label=f"Channel {functs.bnl.channel_number(3, channel_tags, ch)}")
            elif channel_tags[i] == 4:
                plt.errorbar(cx, ampl, yerr=dev, linestyle='none', marker='.', color='green', ecolor='palegreen', label=f"Channel {functs.bnl.channel_number(4, channel_tags, ch)}")
        plt.plot(cx, totx, 'm.', label="Total Amplitude")
        plt.legend()
        plt.title('Amplitude vs X')
        plt.xlabel('X Position (microns)')
        plt.ylabel('Amplitude (mV)')
        plt.axvspan(145, 355, color='grey', alpha=0.3)
        plt.axvspan(645, 855, color='grey', alpha=0.3)
        plt.axvspan(1145, 1355, color='grey', alpha=0.3)
        plt.axvspan(1645, 1845, color='grey', alpha=0.3)
        plt.savefig(f"{plotlocation}/ampl-all-x.pdf")
        plt.clf()

        for i in range(len(channel_tags)):
            ampl = np.loadtxt(f"{datalocation}/amplitude_ch{channel_tags[i]}.txt")
            dev = np.loadtxt(f"{datalocation}/amplitude_dev_ch{channel_tags[i]}.txt")
            
            toty += ampl
            
            if channel_tags[i] == 2:
                plt.errorbar(cy, ampl, yerr=dev, linestyle='none', marker='.', color='purple', ecolor='plum', label=f"Channel {functs.bnl.channel_number(2, channel_tags, ch)}")
            elif channel_tags[i] == 3:
                plt.errorbar(cy, ampl, yerr=dev, linestyle='none', marker='.', color='teal', ecolor='paleturquoise', label=f"Channel {functs.bnl.channel_number(3, channel_tags, ch)}")
            elif channel_tags[i] == 4:
                plt.errorbar(cy, ampl, yerr=dev, linestyle='none', marker='.', color='green', ecolor='palegreen', label=f"Channel {functs.bnl.channel_number(4, channel_tags, ch)}")
        plt.plot(cy, toty, 'm.', label='Total Amplitude')
        plt.legend()
        plt.title('Amplitude vs Y')
        plt.xlabel('Y Position (microns)')
        plt.ylabel('Amplitude (mV)')
        plt.axvspan(145, 355, color='grey', alpha=0.3)
        plt.axvspan(645, 855, color='grey', alpha=0.3)
        plt.axvspan(1145, 1355, color='grey', alpha=0.3)
        plt.axvspan(1645, 1855, color='grey', alpha=0.3)
        plt.savefig(f"{plotlocation}/ampl-all-y.pdf")
        plt.clf()

        ## do from center of pad:
        for i in range(len(channel_tags)):
            xcen, ycen = functs.bnl.channel_center(channel_tags[i], channel_tags, ch)
            R = []
            for j in range(len(xx)):
                R.append(np.sqrt( (cx[j] - xcen)**2 + (cy[j] - ycen)**2 ))
            
            ampl = np.loadtxt(f"{datalocation}/amplitude_ch{channel_tags[i]}.txt")
            dev = np.loadtxt(f"{datalocation}/amplitude_dev_ch{channel_tags[i]}.txt")
            
            if channel_tags[i] == 2:
                plt.errorbar(R, ampl, yerr=dev, linestyle='none', marker='.', color='purple', ecolor='plum', label=f"Channel {functs.bnl.channel_number(2, channel_tags, ch)}")
            elif channel_tags[i] == 3:
                plt.errorbar(R, ampl, yerr=dev, linestyle='none', marker='.', color='teal', ecolor='paleturquoise', label=f"Channel {functs.bnl.channel_number(3, channel_tags, ch)}")
            elif channel_tags[i] == 4:
                plt.errorbar(R, ampl, yerr=dev, linestyle='none', marker='.', color='green', ecolor='palegreen', label=f"Channel {functs.bnl.channel_number(4, channel_tags, ch)}")
        plt.legend()
        plt.title('Amplitude vs R')
        plt.xlabel('R from Center of Pad (microns)')
        plt.ylabel('Amplitude (mV)')
        plt.savefig(f"{plotlocation}/ampl-all-r.pdf")
        plt.clf()

def ampl_hist(channel, datalocation, plotlocation, date, channel_tags, ch):
    coords= np.loadtxt(f"{datalocation}/scposition{date}.txt")
    xx, yy = coords[:,0], coords[:,1]
    
    for i in range(len(coords)):
        ampl = np.load(f"{datalocation}/amplitudes_ch{channel}-x{int(xx[i])}-y{int(yy[i])}.npy")
    
        bb = np.linspace(min(ampl), max(ampl), 100)
        plt.hist(ampl, color='purple', edgecolor='black', bins=bb, label='mean = {} mV\n$\sigma$ = {} mV'.format(round(np.mean(ampl), 3), round(np.std(ampl),3)))
        plt.legend()
        plt.title(f"Amplitude; Channel {functs.bnl.channel_number(channel, channel_tags, ch)}; Position ({int(coords[:,0][i])}, {int(coords[:,1][i])})")
        plt.xlabel("Amplitude (mV)")
        plt.savefig(f"{plotlocation}/hist-ampl-ch{channel}-x{int(xx[i])}-y{int(yy[i])}.pdf")
        plt.clf()

def weighted_avg_hist(datalocation, plotlocation, date):
    coords = np.loadtxt(f"{datalocation}/scposition{date}.txt")
    xx, yy = coords[:,0], coords[:,1]
    ux, uy = functs.bnl.convert_coords(datalocation, date)
    
    for i in range(len(xx)):
        wax = np.loadtxt(f"{datalocation}/wax-x{int(xx[i])}-y{int(yy[i])}-board0.txt")
        way = np.loadtxt(f"{datalocation}/way-x{int(xx[i])}-y{int(yy[i])}-board0.txt")
                
        # bbx = np.linspace(min(wax) - 10, max(wax) + 10, 100)
        # bby = np.linspace(min(way) - 10, max(way) + 10, 100)
        
        (xmu, xsigma) = norm.fit(wax)
        (ymu, ysigma) = norm.fit(way)
        
        
        
        h, xbins, xpatches = plt.hist(wax, bins=100, color='purple', edgecolor='black', label=f'mean = {round(np.mean(wax), 3)} \n$\sigma$ = {round(np.std(wax), 3)}')
        xplt = norm.pdf(xbins, xmu, xsigma)
        plt.plot(xbins, xplt, 'r--', linewidth=2, label=f"mu={xmu}, sigma={xsigma}")
        plt.legend()
        plt.title(f'Reconstructed X; True x = {int(ux[i])}')
        plt.xlabel('Reconstructed X (weighted average)')
        plt.savefig(f'{plotlocation}/hist-wax-x{int(xx[i])}-y{int(yy[i])}.pdf')
        plt.clf()
        
        hy, ybins, ypatches = plt.hist(way, bins= 100, color='purple', edgecolor='black', label=f'mean = {round(np.mean(way), 3)} \n$\sigma$ = {round(np.std(way), 3)}')
        yplt = norm.pdf(ybins, ymu, ysigma)
        plt.plot(ybins, yplt, 'b--', linewidth=2, label=f"mu={ymu}, sigma={ysigma}")
        plt.legend()
        plt.title(f'Reconstructed Y; True y = {int(uy[i])}')
        plt.xlabel('Reconstructed Y (weighted average)')
        plt.savefig(f'{plotlocation}/hist-way-x{int(xx[i])}-y{int(yy[i])}.pdf')
        plt.clf()
        
def spat_diff(datalocation, plotlocation, date):
    coords = np.loadtxt(f"{datalocation}/scposition{date}.txt")
    xx, yy = coords[:,0], coords[:,1]
    ux, uy = functs.bnl.convert_coords(datalocation, date)
    
    trux, truy, recox, recoy = spatres.diffs(datalocation, date, nn, channel_tags, ch)
    
    diffx = recox - trux
    diffy = recoy - truy
    
    plt.plot(trux, recox, '.')
    plt.xlabel("True x position")
    plt.ylabel("Reconstructed x position")
    plt.savefig(f"{plotlocation}/tru-vs-reco-x.pdf")
    plt.clf()
    
    plt.plot(truy, recoy, '.')
    plt.xlabel("True y position")
    plt.ylabel("Reconstructed y position")
    plt.savefig(f"{plotlocation}/tru-vs-reco-y.pdf")
    plt.clf()
    
    plt.hist(diffx, color='purple', edgecolor='black', label=f'mean = {round(np.mean(diffx), 3)} \n$\sigma$ = {round(np.std(diffx), 3)}')
    plt.legend()
    plt.title(f'Reconstructed - True')
    plt.xlabel('Reco - True (microns) (weighted average)')
    plt.savefig(f'{plotlocation}/hist-diffx.pdf')
    plt.clf()
    
    plt.hist(diffy, color='purple', edgecolor='black', label=f'mean = {round(np.mean(diffy), 3)} \n$\sigma$ = {round(np.std(diffy), 3)}')
    plt.legend()
    plt.title(f'Reconstructed - True')
    plt.xlabel('Reco - True (microns) (weighted average)')
    plt.savefig(f'{plotlocation}/hist-diffy.pdf')
    plt.clf()
 

        
        
 
#########
     
def plot_all_ampl(datalocation, date, xcorr, ycorr, xmin, ymin, channel_tags, ch):
    for i in range(len(ch)):
        channel = channel_tags[i]
        ampl = np.loadtxt("{0}/amplitude_ch{1}.txt".format(datalocation, channel))
        dev = np.loadtxt("{0}/amplitude_dev_ch{1}.txt".format(datalocation, channel))
    
        coords = np.loadtxt("{0}/scposition{1}.txt".format(datalocation, date))
        xx = coords[:,0]
        yy = coords[:,1]

        cx, cy = [], []
        for i in range(len(xx)):
            cx.append( (xx[i] - xmin)*2.5 - xcorr )
            cy.append( (yy[i] - ymin)*2.5 - ycorr )
        
        # plt.errorbar(cx, ampl, yerr=dev, linestyle='none', marker='.', color='purple', ecolor='plum', label="Channel {0}".format(functs.channel_number(channel, channel_tags, ch)))
        # plt.legend()
        # plt.title('Amplitude vs X')
        # plt.xlabel('X Position (microns)')
        # plt.ylabel('Amplitude (mV)')
        # plt.axvspan(0, 105, color='grey', alpha=0.3)
        # plt.axvspan(395, 500, color='grey', alpha=0.3)
        # plt.show()
        # plt.clf()

        # plt.errorbar(cy, ampl, yerr=dev, label="Channel {0}".format(functs.channel_number(channel, channel_tags, ch)), linestyle='none', marker='.', color='purple', ecolor='plum')
        # plt.legend()
        # plt.title('Amplitude vs Y')
        # plt.xlabel('Y Position (microns)')
        # plt.ylabel('Amplitude (mV)')
        # plt.axvspan(0, 105, color='grey', alpha=0.3)
        # plt.axvspan(395, 500, color='grey', alpha=0.3)
        # plt.show()
        # plt.clf()

        ## do from center of pad:
        xcen, ycen = functs.channel_center(channel, channel_tags, ch)
        R = []
        for i in range(len(xx)):
            R.append(np.sqrt( (xx[i] - xcen)**2 + (yy[i] - ycen)**2 ) * 2.5)

        plt.errorbar(R, ampl, yerr=dev, linestyle='none', marker='.', label="Channel {0}".format(functs.channel_number(channel, channel_tags, ch) ))
        plt.legend()
        plt.title('Amplitude vs R')
        plt.xlabel('R from Center of Pad (microns)')
        plt.ylabel('Amplitude (mV)')
    plt.show()
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
