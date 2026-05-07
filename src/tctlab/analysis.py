import numpy as np
import matplotlib.pyplot as plt

import functs, plotting, process, spatres

plt.rcParams['figure.dpi'] = 150
import mplhep as hep
hep.style.use("LHCb2")

## for a brand new data set:
def full_run(datalocation, plotlocation, date, p, nn, channel_tags, ch, xmin, xmax, ymin, ymax):

    process.matrices(datalocation, date, 2, nn)
    process.matrices(datalocation, date, 3, nn)
    process.matrices(datalocation, date, 4, nn)

    functs.waveforms.amplitude(datalocation, date, 2, -1, nn)
    functs.waveforms.amplitude(datalocation, date, 3, -1, nn)
    functs.waveforms.amplitude(datalocation, date, 4, -1, nn)
    print("amplitudes updated")

    plotting.plot_all_wfms(2, datalocation, plotlocation, date, channel_tags, ch)
    plotting.plot_all_wfms(3, datalocation, plotlocation, date, channel_tags, ch)
    plotting.plot_all_wfms(4, datalocation, plotlocation, date, channel_tags, ch)

    plotting.plot_sep_wfms(2, datalocation, plotlocation, date, channel_tags, ch)
    plotting.plot_sep_wfms(3, datalocation, plotlocation, date, channel_tags, ch)
    plotting.plot_sep_wfms(4, datalocation, plotlocation, date, channel_tags, ch)

    plotting.map_amplitude_2d(2, datalocation, plotlocation, date)
    plotting.map_amplitude_2d(3, datalocation, plotlocation, date)
    plotting.map_amplitude_2d(4, datalocation, plotlocation, date)
    
    plotting.plot_avg_ampl(2, datalocation, plotlocation, date, channel_tags, ch)
    plotting.plot_avg_ampl(3, datalocation, plotlocation, date, channel_tags, ch)
    plotting.plot_avg_ampl(4, datalocation, plotlocation, date, channel_tags, ch)
    
    plotting.plot_all_avg_ampl(datalocation, plotlocation, date, channel_tags, ch)
    print("all waveforms plotted")

    
    ## add spatial, time resolution stuff.
