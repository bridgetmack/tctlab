import numpy as np
import matplotlib.pyplot as plt

import functs, plotting, process, spatres

plt.rcParams['figure.dpi'] = 150
import mplhep as hep
hep.style.use("LHCb2")

## for a brand new data set:
def full_run(datalocation, date, p, nn, channel_tags, ch, xmin, xmax, ymin, ymax):

    try: 
        process.matrices(datalocation, date, 2, nn)
        process.matrices(datalocation, date, 3, nn)
        process.matrices(datalocation, date, 4, nn)
    except: 
        process.no_pos_matrices(datalocation, xmin, xmax, ymin, ymax, 2, date)
        process.no_pos_matrices(datalocation, xmin, xmax, ymin, ymax, 3, date)
        process.no_pos_matrices(datalocation, xmin, xmax, ymin, ymax, 4, date)

        process.matrices(datalocation, date, 2, nn)
        process.matrices(datalocation, date, 3, nn)
        process.matrices(datalocation, date, 4, nn)

    functs.amplitude(datalocation, date, 2, p, nn)
    functs.amplitude(datalocation, date, 3, p, nn)
    functs.amplitude(datalocation, date, 4, p, nn)
    print("amplitudes updated")

    plotting.plot_all_wfms(2, datalocation, date, channel_tags, ch)
    plotting.plot_all_wfms(3, datalocation, date, channel_tags, ch)
    plotting.plot_all_wfms(4, datalocation, date, channel_tags, ch)

    plotting.plot_sep_wfms(2, datalocation, date, channel_tags, ch)
    plotting.plot_sep_wfms(3, datalocation, date, channel_tags, ch)
    plotting.plot_sep_wfms(4, datalocation, date, channel_tags, ch)
    print("all waveforms plotted")

    ## add spatial, time resolution stuff.
