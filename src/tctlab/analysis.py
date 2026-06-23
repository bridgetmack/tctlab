import numpy as np
import matplotlib.pyplot as plt

import functs, plotting, process, spatres

plt.rcParams['figure.dpi'] = 150
import mplhep as hep
hep.style.use("LHCb2")

## for a brand new data set:
def full_run(channel, datalocation, plotlocation, date, p, nn, channel_tags, ch, xmin, xmax, ymin, ymax):

    for i in range(8):
        process.matrices(datalocation, date, i, nn)
        functs.waveforms.amplitude(datalocation, date, i, 1, nn)
        print("amplitudes updated")

        plotting.plot_all_wfms(i, datalocation, plotlocation, date, channel_tags, ch)

        plotting.plot_sep_wfms(i, datalocation, plotlocation, date, channel_tags, ch)
        #plotting.map_amplitude_2d(i, datalocation, plotlocation, date)
    
        plotting.plot_avg_ampl(i, datalocation, plotlocation, date, channel_tags, ch)

    plotting.plot_all_avg_ampl(datalocation, plotlocation, date, channel_tags, ch)
    print("all waveforms plotted")


    #process.matrices(datalocation, date, 2, nn)
    #process.matrices(datalocation, date, 3, nn)
    #process.matrices(datalocation, date, 4, nn)
    #process.matrices(datalocation, date, 5, nn)
    #process.matrices(datalocation, date, 6, nn)
    #process.matrices(datalocation, date, 7, nn)

    #functs.waveforms.amplitude(datalocation, date, 1, 1, nn)
    #functs.waveforms.amplitude(datalocation, date, 2, 1, nn)
    #functs.waveforms.amplitude(datalocation, date, 3, 1, nn)
    #functs.waveforms.amplitude(datalocation, date, 4, 1, nn)
    #functs.waveforms.amplitude(datalocation, date, 5, 1, nn)
    #functs.waveforms.amplitude(datalocation, date, 6, 1, nn)
    #functs.waveforms.amplitude(datalocation, date, 7, 1, nn)
    #print("amplitudes updated")

    #plotting.plot_all_wfms(2, datalocation, plotlocation, date, channel_tags, ch)
    #plotting.plot_all_wfms(3, datalocation, plotlocation, date, channel_tags, ch)
    #plotting.plot_all_wfms(4, datalocation, plotlocation, date, channel_tags, ch)

    #plotting.plot_sep_wfms(2, datalocation, plotlocation, date, channel_tags, ch)
    #plotting.plot_sep_wfms(3, datalocation, plotlocation, date, channel_tags, ch)
    #plotting.plot_sep_wfms(4, datalocation, plotlocation, date, channel_tags, ch)

    #plotting.map_amplitude_2d(2, datalocation, plotlocation, date)
    #plotting.map_amplitude_2d(3, datalocation, plotlocation, date)
    #plotting.map_amplitude_2d(4, datalocation, plotlocation, date)
    
    #plotting.plot_avg_ampl(2, datalocation, plotlocation, date, channel_tags, ch)
    #plotting.plot_avg_ampl(3, datalocation, plotlocation, date, channel_tags, ch)
    #plotting.plot_avg_ampl(4, datalocation, plotlocation, date, channel_tags, ch)
    
    #plotting.plot_all_avg_ampl(datalocation, plotlocation, date, channel_tags, ch)
    #print("all waveforms plotted")

    ## need to look at laser spot size
    ## add spatial, time resolution stuff.

def testing(channel, datalocation, plotlocation, date, p, nn, channel_tags, ch, xmin, xmax, ymin, ymax):
    process.matrices(datalocation, date, channel, nn)
    functs.waveforms.amplitude(datalocation, date, channel, 1, nn)
    print("amplitudes updated")
    
    plotting.plot_individual(channel, 0, datalocation, plotlocation, date, xmin, ymin)
    #plotting.plot_all_wfms(channel, datalocation, plotlocation, date, channel_tags, ch)
    #plotting.plot_sep_wfms(channel, datalocation, plotlocation, date, channel_tags, ch)
    plotting.map_amplitude_2d(channel, datalocation, plotlocation, date, channel_tags, ch)
    plotting.plot_avg_ampl(channel, datalocation, plotlocation, date, channel_tags, ch)
    plotting.ampl_hist(channel, datalocation, plotlocation, date, channel_tags, ch)
    print("waveform plots updated")
    
    spatres.weighted_average(datalocation, date, nn, channel_tags, ch)
    spatres.diffs(datalocation, date, nn, channel_tags, ch)
    
    plotting.weighted_avg_hist(datalocation, plotlocation, date)
    plotting.spat_diff_hist(datalocation, plotlocation, date)
    print("spatial resolutions updated")