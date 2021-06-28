import numpy as np
import pylab as plt
from ANNarchy import raster_plot
from initialTestofBOLD_ANA import get_pop_rate

def spikeActivityPlot(title='', spikes=None, simParams={}, times=[], ax=None):
    try:
        plt.title(title)
        # raster plot
        t,n = raster_plot(spikes)
        t*=simParams['dt']
        plt.plot(t,n,'k.',markersize=0.3)
        plt.xlim(times[0],times[-1])
        # population rate
        ax2=ax.twinx()
        ax2.plot(times,get_pop_rate(spikes,simParams,simParams['simDur'],t_smooth_ms=-1))
        return 1
    except:
        return 0
        
def splotAverageOfNeuronVariable(title='', variable=[], times=[]):
    plt.title(title)
    plt.plot(times, np.mean(variable,1), color='k')
    plt.xlim(times[0],times[-1])
        


### LOAD DATA
recordings  = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordings.npy', allow_pickle=True).item()
recordingsB = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordingsB.npy', allow_pickle=True).item()
simParams   = np.load('../dataRaw/simulations_BOLDfromDifferentSources_simParams.npy', allow_pickle=True).item()

times=np.arange(simParams['rampUp']+simParams['dt'],simParams['rampUp']+simParams['simDur']+simParams['dt'],simParams['dt'])



### CREATE FIGURE with standard monitors
plt.figure(figsize=(16,9),dpi=500)

## SUBPLOT INPUT ACTIVITY
ax = plt.subplot(2,4,1)
spikeActivityPlot(title='input activity', spikes=recordings['inputPop;spike'], simParams=simParams, times=times, ax=ax)

## SUBPLOT corE ACTIVITY
ax = plt.subplot(2,4,2)
spikeActivityPlot(title='corE activity', spikes=recordings['corEL1;spike'], simParams=simParams, times=times, ax=ax)

## SUBPLOT corI ACTIVITY
ax = plt.subplot(2,4,6)
spikeActivityPlot(title='corI activity', spikes=recordings['corIL1;spike'], simParams=simParams, times=times, ax=ax)

## SUBPLOT corE SYN VARIABLE
plt.subplot(2,4,3)
splotAverageOfNeuronVariable(title='corE syn', variable=recordings['corEL1;syn'], times=times)

## SUBPLOT corE SYN VARIABLE
plt.subplot(2,4,7)
splotAverageOfNeuronVariable(title='corI syn', variable=recordings['corIL1;syn'], times=times)

## SUBPLOT corI r VARIABLE
plt.subplot(2,4,4)
splotAverageOfNeuronVariable(title='corI rToCMRO2', variable=recordings['corIL1;rToCMRO2'], times=times)

## SUBPLOT corI var_r AND var_ra
plt.subplot(2,4,8)
plt.title('corI var_r')
plt.plot(times,np.mean(recordings['corIL1;var_r'],1), label='var_r', alpha=0.5)
plt.plot(times,np.mean(recordings['corIL1;var_ra'],1), label='var_ra', alpha=0.5)
plt.xlim(times[0],times[-1])
plt.legend()

plt.tight_layout()
plt.savefig('BOLDfromDifferentSources_ANA.png')
