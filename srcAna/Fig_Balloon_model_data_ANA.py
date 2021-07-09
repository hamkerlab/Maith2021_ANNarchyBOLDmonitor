import numpy as np
import pylab as plt
from extras import set_size

### LOAD DATA
recordingsB = np.load('../dataRaw/Fig_Balloon_model_data_recordingsB.npy', allow_pickle=True).item()
simParams   = np.load('../dataRaw/Fig_Balloon_model_data_simParams.npy', allow_pickle=True).item()

times=np.arange(simParams['dt'],simParams['sim_dur1']+simParams['sim_dur2']+simParams['sim_dur3']+simParams['dt'],simParams['dt'])

ylim_dict={'f_in':     [1, 1.7],
           'v':        [1, 1.7],
           'f_out':    [1, 1.7],
           'BOLD':     [1, 0.015],
           'r':        [1, 1.7],
           'I_CBF':    [1, 0.23],
           'I_CMRO2':  [1, 0.23],
           'q':        [2, 0.75],
           'E':        [2, 0.22],
           's':        [2,-0.2],
           's_CBF':    [2,-0.2],
           's_CMRO2':  [2,-0.2]}

### FIGURE
for key in recordingsB.keys():
    plt.figure(dpi=500)
    plt.axhline(recordingsB[key][0,0], color='grey')
    plt.plot(times, recordingsB[key][:,0], color='k')
    plt.xlim(times[0],times[-1])
    if ylim_dict[key.split(';')[1]][0] == 1:
        ## initial value is at 20%
        y100 = ylim_dict[key.split(';')[1]][1]
        y20  = recordingsB[key][0,0]
        y0   = y20 + (y20 - y100) * (1/4.)
        plt.ylim(y0, y100)
    else:
        ## initial value is at 50%
        y50  = recordingsB[key][0,0]
        y0   = ylim_dict[key.split(';')[1]][1]
        y100 = y50 + (y50-y0)
        plt.ylim(y0,y100)
    plt.title(key)
    plt.subplots_adjust(left=0.3, top=0.7)
    set_size(2.03/2.54,1.35/2.54)
    plt.savefig('../results/Fig_Balloon_model_data/'+key.replace(';','_')+'.svg')
    

### EXTRA FIGURE
plt.figure(figsize=(16,9), dpi=500)
plt.axhline(1, color='grey')
plt.plot(times, recordingsB['2;f_in'], label='CBF')
plt.plot(times, recordingsB['2;r'], label='CMRO2')
## AX LIMITS
plt.xlim(times[0],times[-1])
y100 = ylim_dict['f_in'][1]
y20  = 1
y0   = y20 + (y20 - y100) * (1/4.)
plt.ylim(y0, y100)
plt.legend()
## SAVE
plt.savefig('../results/Fig_Balloon_model_data/extra_fig.png')




