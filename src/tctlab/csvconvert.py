import numpy as np
import matplotlib.pyplot as plt

import functs, plotting, process, spatres

plt.rcParams['figure.dpi'] = 150
import mplhep as hep
hep.style.use("LHCb2")

# need to be able to look at files directly from csv, and not rely on scpositions

import sys, os

# os.system(f"rm {datalocation}/csv/files.txt")
# os.system(f"ls {datalocation}csv/*.csv > {datalocation}/csv/files.txt")

def make_txt(datalocation, date, channel_tags):

    files = np.loadtxt(f"{datalocation}/csv/files.txt", dtype=str)
    pos = []
    
    for f in range(len(files)):
        filename = files[f].split("-")
        if "waveforms" in filename:
            xx, yy = filename[1], filename[2]
        else:
            xx, yy = filename[0], filename[1]
        xx = int(xx.replace(xx[0], "", 1))
        yy = int(yy.replace(yy[0], "", 1))
        pos.append([xx, yy])
        
        csv_file = np.loadtxt(f"{datalocation}/csv/{files[f]}", delimiter=",", skiprows=1)
        
        events = csv_file[:,0]
        samples = csv_file[:,1]
        nsamples = int(max(samples)+1)
        
        start_window = csv_file[:,3]
        sw = []
        for e in range(len(events)):
            sw.append([events[e], start_window[e]])
        # sw = np.array(sw)
        sw = np.unique(np.array(sw), axis=0)
        
        for channel in range(len(channel_tags)):
            if "board1" in filename:
                channel -= 8
            
            ch_list = csv_file[:,channel+4]
            
            c0 = np.zeros((int(max(events)+1), nsamples))
            cc = [ch_list[i:i + nsamples] for i in range(0, len(ch_list), nsamples)]
            for j in range(len(cc)):
                try: 
                    c0[j,:] = cc[j]
                except:
                    print(j)
            
            wfms = np.transpose(cc)
            t = np.linspace(0, nsamples*10, nsamples)
            t = t/1000 #picoseconds?
            
            # remove baseline
            wfms = np.array(wfms) - np.mean(wfms[:100], axis=0)
        
            if ped == True:
                data = np.zeros([len(wfms), len(wfms[0])])
                starts = sw[:,1]
                npts = len(wfms)
                
                ped_file = np.genfromtxt(f"{datalocation}/dynamicPedestals.csv", delimiter=",", names=True)
                slopes = f"ch{channel}_slopeDevs"
                yiters = f"ch{channel}_yIters"
                ADC = 3.1422522482546027
                
                for i in range(len(wfms[0])):
                    slope_seg = ped_file[slopes][int(starts[i]*64):int(starts[i]*64+npts)]
                    
                    data[:,i] = wfms[:,i] - (slope_seg*wfms[:,i]/ADC)
                
                wfms = data
            
            if "board1" in filename:
                channel += 8
            np.savetxt(f"{datalocation}/chan{channel}v{date}-x{int(xx)}-y{int(yy)}.txt", wfms)
            np.savetxt(f"{datalocation}/chan{channel}t{date}-x{int(xx)}-y{int(yy)}.txt", t)
              
    np.savetxt(f"{datalocation}/scposition{date}.txt", pos)
            