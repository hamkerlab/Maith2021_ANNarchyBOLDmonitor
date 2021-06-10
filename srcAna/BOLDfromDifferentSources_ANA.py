import numpy as np
import pylab as plt
from ANNarchy import raster_plot
from initialTestofBOLD_ANA import get_pop_rate

def spikeActivityPlot(title='', pop='', recordings={}, simParams={}, times=[]):
    try:
        plt.title(title)
        # raster plot
        t,n = raster_plot(recordings[pop+';spike'])
        t*=simParams['dt']
        plt.plot(t,n,'k.',markersize=0.3)
        plt.xlim(times[0],times[-1])
        # population rate
        ax2=ax.twinx()
        ax2.plot(times,get_pop_rate(recordings[pop+';spike'],simParams,simParams['simDur'],t_smooth_ms=-1))
        return 1
    except:
        return 0
        


### LOAD DATA
recordings  = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordings.npy', allow_pickle=True).item()
recordingsB = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordingsB.npy', allow_pickle=True).item()
simParams   = np.load('../dataRaw/simulations_BOLDfromDifferentSources_simParams.npy', allow_pickle=True).item()

times=np.arange(simParams['rampUp']+simParams['dt'],simParams['rampUp']+simParams['simDur']+simParams['dt'],simParams['dt'])



### CREATE FIGURE with standard monitors
plt.figure(figsize=(16,9),dpi=500)

## SUBPLOT INPUT ACTIVITY
ax = plt.subplot(2,4,1)
spikeActivityPlot(title='input activity', recordings=recordings, pop='inputPop', simParams=simParams, times=times)


## SUBPLOT corE ACTIVITY
ax = plt.subplot(2,4,2)
spikeActivityPlot(title='corE activity', pop='corEL1', recordings=recordings, simParams=simParams, times=times)


plt.tight_layout()
plt.savefig('BOLDfromDifferentSources_ANA.png')
