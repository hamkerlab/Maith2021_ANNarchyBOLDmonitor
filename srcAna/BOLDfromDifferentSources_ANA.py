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
        
def plotAverageOfNeuronVariables(title='', variables=[], labels=[], times=[], ax=None):
    if isinstance(variables, list):
        if len(variables)==1:
            plt.title(title)
            plt.plot(times, np.mean(variables[0],1), color='k')
            plt.xlim(times[0],times[-1])
            return 1
        elif len(variables)==2:
            # first averaged data
            plt.title(title)
            lns1 = plt.plot(times, np.mean(variables[0],1), color='C0', alpha=0.5, label=labels[0])
            plt.xlim(times[0],times[-1])
            # second averaged data
            ax2=ax.twinx()
            lns2 = ax2.plot(times, np.mean(variables[1],1), color='C1', alpha=0.5, label=labels[1])
            # legend
            lns = lns1+lns2
            ax.legend(lns, labels)
            return 1
        else:
            return 0
    else:
        return 0
        


### LOAD DATA
recordings  = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordings.npy', allow_pickle=True).item()
recordingsB = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordingsB.npy', allow_pickle=True).item()
simParams   = np.load('../dataRaw/simulations_BOLDfromDifferentSources_simParams.npy', allow_pickle=True).item()

times=np.arange(simParams['rampUp']+simParams['dt'],simParams['rampUp']+simParams['simDur']+simParams['dt'],simParams['dt'])



### CREATE FIGURE WITH STANDARD MONITORS
plt.figure(figsize=(16,9),dpi=500)

## FIRST COLUMN
## SUBPLOT INPUT ACTIVITY
ax = plt.subplot(2,5,1)
spikeActivityPlot(title='input activity', spikes=recordings['inputPop;spike'], simParams=simParams, times=times, ax=ax)

## SECOND COLUMN
## SUBPLOT corE ACTIVITY
ax = plt.subplot(2,5,2)
spikeActivityPlot(title='corE activity', spikes=recordings['corEL1;spike'], simParams=simParams, times=times, ax=ax)
## SUBPLOT corI ACTIVITY
ax = plt.subplot(2,5,7)
spikeActivityPlot(title='corI activity', spikes=recordings['corIL1;spike'], simParams=simParams, times=times, ax=ax)

## THIRD COLUMN
## SUBPLOT corE RECORDED RATE
ax = plt.subplot(2,5,3)
plotAverageOfNeuronVariables(title='corE recorded rate', variables=[recordings['corEL1;r']], times=times)
## SUBPLOT corI RECORDED RATE
ax = plt.subplot(2,5,8)
plotAverageOfNeuronVariables(title='corI recorded rate', variables=[recordings['corIL1;r']], times=times)

## FOURTH COLUMN
## SUBPLOT corE SYN VARIABLE
plt.subplot(2,5,4)
plotAverageOfNeuronVariables(title='corE syn', variables=[recordings['corEL1;syn']], times=times)
## SUBPLOT corE SYN VARIABLE
plt.subplot(2,5,9)
plotAverageOfNeuronVariables(title='corI syn', variables=[recordings['corIL1;syn']], times=times)

## FIFTH COLUMN
## SUBPLOT corE var_r AND var_ra
ax = plt.subplot(2,5,5)
plotAverageOfNeuronVariables(title='corE var_r', variables=[recordings['corEL1;var_r'],recordings['corEL1;var_ra']], labels=['var_r','var_ra'], times=times, ax=ax)
## SUBPLOT corI var_r AND var_ra
ax = plt.subplot(2,5,10)
plotAverageOfNeuronVariables(title='corI var_r', variables=[recordings['corIL1;var_r'],recordings['corIL1;var_ra']], labels=['var_r','var_ra'], times=times, ax=ax)

plt.tight_layout()
plt.savefig('BOLDfromDifferentSources_ANA_standard.png')


### NEXT FIGURE WITH BOLD MONITORS
plt.figure(figsize=(16,9),dpi=500)

## FIRST COLUMN
## BOLD1
plt.subplot(5,5,1)
plt.title('E raw input')
plt.plot(times, recordingsB['1Eraw;r'][:,0])
plt.subplot(5,5,6)
plt.title('I raw input')
plt.plot(times, recordingsB['1Iraw;r'][:,0])
plt.subplot(5,5,11)
plt.title('actual input'+str(round(np.mean(recordingsB['1;r'][:,0]),3)))
plt.plot(times, recordingsB['1;r'][:,0])
plt.subplot(5,5,16)
plt.subplot(5,5,21)
plt.title('BOLD')
plt.plot(times, recordingsB['1;BOLD'][:,0])

## SECOND COLUMN
plt.subplot(5,5,2)
plt.title('E raw input')
plt.plot(times, recordingsB['2Eraw;r'][:,0])
plt.subplot(5,5,7)
plt.title('I raw input')
plt.plot(times, recordingsB['2Iraw;r'][:,0])
plt.subplot(5,5,12)
plt.title('actual input'+str(round(np.mean(recordingsB['2;r'][:,0]),3)))
plt.plot(times, recordingsB['2;r'][:,0])
plt.subplot(5,5,17)
plt.subplot(5,5,22)
plt.title('BOLD')
plt.plot(times, recordingsB['2;BOLD'][:,0])

## THIRD COLUMN
plt.subplot(5,5,3)
plt.title('E raw input')
plt.plot(times, recordingsB['3Eraw;r'][:,0])
plt.subplot(5,5,8)
plt.title('I raw input')
plt.plot(times, recordingsB['3Iraw;r'][:,0])
plt.subplot(5,5,13)
plt.title('actual input'+str(round(np.mean(recordingsB['3;r'][:,0]),3)))
plt.plot(times, recordingsB['3;r'][:,0])
plt.subplot(5,5,18)
plt.subplot(5,5,23)
plt.title('BOLD')
plt.plot(times, recordingsB['3;BOLD'][:,0])

## FOURTH COLUMN
plt.subplot(5,5,4)
plt.title('E raw input')
plt.plot(times, recordingsB['4Eraw;I_CBF'][:,0], label='I_f', alpha=0.5)
plt.plot(times, recordingsB['4Eraw;I_CMRO2'][:,0], label='I_r', alpha=0.5)
plt.legend()
plt.subplot(5,5,9)
plt.title('I raw input')
plt.plot(times, recordingsB['4Iraw;I_CBF'][:,0], label='I_f', alpha=0.5)
plt.plot(times, recordingsB['4Iraw;I_CMRO2'][:,0], label='I_r', alpha=0.5)
plt.legend()
plt.subplot(5,5,14)
plt.title('actual input')
plt.plot(times, recordingsB['4;I_CBF'][:,0], label='I_CBF', alpha=0.5)
plt.plot(times, recordingsB['4;I_CMRO2'][:,0], label='I_CMRO2', alpha=0.5)
plt.legend()
plt.subplot(5,5,19)
plt.title('flow and oxygen')
plt.plot(times, recordingsB['4;CBF'][:,0], label='CBF', alpha=0.5)
plt.plot(times, recordingsB['4;CMRO2'][:,0], label='CMRO2', alpha=0.5)
plt.legend()
plt.subplot(5,5,24)
plt.title('BOLD')
plt.plot(times, recordingsB['4;BOLD'][:,0])

## FITH COLUMN
plt.subplot(5,5,5)
plt.title('E raw input')
plt.plot(times, recordingsB['5Eraw;I_CBF'][:,0], label='I_f', alpha=0.5)
plt.plot(times, recordingsB['5Eraw;I_CMRO2'][:,0], label='I_r', alpha=0.5)
plt.legend()
plt.subplot(5,5,10)
plt.title('I raw input')
plt.plot(times, recordingsB['5Iraw;I_CBF'][:,0], label='I_f', alpha=0.5)
plt.plot(times, recordingsB['5Iraw;I_CMRO2'][:,0], label='I_r', alpha=0.5)
plt.legend()
plt.subplot(5,5,15)
plt.title('actual input')
plt.plot(times, recordingsB['5;I_CBF'][:,0], label='I_CBF', alpha=0.5)
plt.plot(times, recordingsB['5;I_CMRO2'][:,0], label='I_CMRO2', alpha=0.5)
plt.legend()
plt.subplot(5,5,20)
plt.title('flow and oxygen')
plt.plot(times, recordingsB['5;CBF'][:,0], label='CBF', alpha=0.5)
plt.plot(times, recordingsB['5;CMRO2'][:,0], label='CMRO2', alpha=0.5)
plt.legend()
plt.subplot(5,5,25)
plt.title('BOLD')
plt.plot(times, recordingsB['5;BOLD'][:,0])

plt.tight_layout()
plt.savefig('BOLDfromDifferentSources_ANA_BOLD.png')




















