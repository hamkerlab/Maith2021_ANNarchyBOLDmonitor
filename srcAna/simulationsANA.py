import numpy as np
import pylab as plt
from ANNarchy import raster_plot

### TODO: move this to srcSim/extras
### maybe makes error with few/ no spikes... check this
def get_pop_rate(spikes,params,maxDur,t_smooth_ms=-1):
    """
        spikes: spikes dictionary from ANNarchy
        maxDur: duration of period after rampUp from which rate is calculated in ms
        t_smooth_ms: time window size for rate calculation in ms, optional, standard = -1 which means automatic window size
        
        returns smoothed population rate from period after rampUp period until maxDur
    """
    tempmaxDur=maxDur+params['rampUp']
    t,n = raster_plot(spikes)
    if len(t)>1:#check if there are spikes in population at all
        if t_smooth_ms==-1:
            ISIs = []
            minTime=np.inf
            maxDur=0
            for idx,key in enumerate(spikes.keys()):
                times = np.array(spikes[key]).astype(int)
                if len(times)>1:#check if there are spikes in neuron
                    ISIs += (np.diff(times)*params['dt']).tolist()#ms
                    minTime=np.min([minTime,times.min()])
                    maxDur=np.max([maxDur,times.max()])
                else:# if there is only 1 spike set ISI to 10ms
                    ISIs+=[10]
            t_smooth_ms = np.min([(maxDur-minTime)/2.*params['dt'],np.mean(np.array(ISIs))*10+10])

        rate=np.zeros((len(list(spikes.keys())),int(tempmaxDur/params['dt'])))
        rate[:]=np.NaN
        binSize=int(t_smooth_ms/params['dt'])
        bins=np.arange(0,int(tempmaxDur/params['dt'])+binSize,binSize)
        binsCenters=bins[:-1]+binSize//2
        for idx,key in enumerate(spikes.keys()):
            times = np.array(spikes[key]).astype(int)
            for timeshift in np.arange(-binSize//2,binSize//2+binSize//10,binSize//10).astype(int):
                hist,edges=np.histogram(times,bins+timeshift)
                rate[idx,np.clip(binsCenters+timeshift,0,rate.shape[1]-1)]=hist/(t_smooth_ms/1000.)

        poprate=np.nanmean(rate,0)
        timesteps=np.arange(0,int(tempmaxDur/params['dt']),1).astype(int)
        time=timesteps[np.logical_not(np.isnan(poprate))]
        poprate=poprate[np.logical_not(np.isnan(poprate))]
        poprate=np.interp(timesteps, time, poprate)

        ret = poprate[int(params['rampUp']/params['dt']):]
    else:
        ret = np.zeros(int(maxDur/params['dt']))

    return ret


### LOAD DATA
recordings  = np.load('../dataRaw/simulations_initialTestofBOLD_recordings.npy', allow_pickle=True).item()
recordingsB = np.load('../dataRaw/simulations_initialTestofBOLD_recordingsB.npy', allow_pickle=True).item()
simParams   = np.load('../dataRaw/simulations_initialTestofBOLD_simParams.npy', allow_pickle=True).item()

times=np.arange(simParams['rampUp']+simParams['dt'],simParams['rampUp']+simParams['simDur']+simParams['dt'],simParams['dt'])



### CREATE FIGURE
rows=10
plt.figure(figsize=(3*rows,9),dpi=500)

## ROW 1: Input & population activities
row=1
ax=plt.subplot(3,rows,rows*0+row)
plt.title('inputPop')
if simParams['input']=='Current':
    for neuron in range(recordings['inputPop;r'].shape[1]):
        plt.plot(times,recordings['inputPop;r'][:,neuron], color='k')
    plt.xlim(times[0],times[-1])
elif simParams['input']=='Poisson':
    # raster plot
    t,n = raster_plot(recordings['inputPop;spike'])
    t*=simParams['dt']
    plt.plot(t,n,'k.',markersize=0.3)
    plt.xlim(times[0],times[-1])
    # population rate
    ax2=ax.twinx()
    ax2.plot(times,get_pop_rate(recordings['inputPop;spike'],simParams,simParams['simDur'],t_smooth_ms=-1))
    
ax=plt.subplot(3,rows,rows*1+row)
plt.title('corE')
# raster plot
t,n = raster_plot(recordings['corEL1;spike'])
print('corE:',len(t)/((simParams['simDur']/1000)*len(list(recordings['corEL1;spike'].keys()))))
t*=simParams['dt']
plt.plot(t,n,'k.',markersize=0.3)
plt.xlim(times[0],times[-1])
# population rate
ax2=ax.twinx()
ax2.plot(times,get_pop_rate(recordings['corEL1;spike'],simParams,simParams['simDur'],t_smooth_ms=-1))
    
ax=plt.subplot(3,rows,rows*2+row)
plt.title('corI')
# raster plot
t,n = raster_plot(recordings['corIL1;spike'])
print('corI:',len(t)/((simParams['simDur']/1000)*len(list(recordings['corIL1;spike'].keys()))))
t*=simParams['dt']
plt.plot(t,n,'k.',markersize=0.3)
plt.xlim(times[0],times[-1])
# population rate
ax2=ax.twinx()
ax2.plot(times,get_pop_rate(recordings['corIL1;spike'],simParams,simParams['simDur'],t_smooth_ms=-1))

## ROW 2: BOLD 1
row=2
plt.subplot(3,rows,rows*0+row)
plt.title('Standard BOLD')
plt.plot(times,recordingsB['1;BOLD'])
plt.ylim(0,0.025)
plt.xlim(times[0],times[-1])

## ROW 3: BOLD 2
row=3
plt.subplot(3,rows,rows*0+row)
plt.title('BOLD with input r')
plt.plot(times,recordingsB['2;BOLD'])
plt.ylim(0,0.025)
plt.xlim(times[0],times[-1])

plt.subplot(3,rows,rows*1+row)
plt.title('r')
plt.plot(times,recordingsB['2;r'])
plt.ylim(0,0.6)
plt.xlim(times[0],times[-1])

plt.subplot(3,rows,rows*2+row)
plt.title('theoretical population z-transform')
rawInput = recordingsB['2;r']
baseline = np.mean(rawInput[:int(2000/simParams['dt'])])
scaling  = np.std(rawInput[:int(2000/simParams['dt'])])*10
print(baseline)
normalizedInput = (rawInput-baseline)/scaling
plt.plot(times,normalizedInput)
plt.ylim(-1.05,1.05)
plt.xlim(times[0],times[-1])

## ROW 4: BOLD 3
row=4
plt.subplot(3,rows,rows*0+row)
plt.title('Scaled population signals')
plt.plot(times,recordingsB['3;BOLD'])
plt.ylim(0,0.025)
plt.xlim(times[0],times[-1])

plt.subplot(3,rows,rows*1+row)
plt.title('r')
plt.plot(times,recordingsB['3;r'])
plt.ylim(0,0.6)
plt.xlim(times[0],times[-1])

## ROW 5: BOLD 4
row=5
plt.subplot(3,rows,rows*0+row)
plt.title('Baseline over 2000 ms')
plt.plot(times,recordingsB['4;BOLD'])
plt.ylim(0,0.025)
plt.xlim(times[0],times[-1])

plt.subplot(3,rows,rows*1+row)
plt.title('CBF')
plt.plot(times,recordingsB['4;f_in'])
plt.ylim(0.5,1.8)
plt.xlim(times[0],times[-1])

plt.subplot(3,rows,rows*2+row)
plt.title('r')
plt.plot(times,recordingsB['4;r'])
plt.ylim(-1.05,1.05)
plt.xlim(times[0],times[-1])

## ROW 6: BOLD 5
row=6
plt.subplot(3,rows,rows*0+row)
plt.title('Self-defined model')
plt.plot(times,recordingsB['5;BOLD'])
#plt.ylim(0,0.025)
plt.xlim(times[0],times[-1])

plt.subplot(3,rows,rows*1+row)
plt.title('CBF & CMRO2')
plt.plot(times,recordingsB['5;CBF'],label='CBF')
plt.plot(times,recordingsB['5;CMRO2'],label='CMRO2')
#plt.ylim(0.5,1.8)
plt.xlim(times[0],times[-1])
plt.legend()

plt.subplot(3,rows,rows*2+row)
plt.title('I_CBF & I_CMRO2')
plt.plot(times,recordingsB['5;I_CBF'],label='I_CBF '+str(round(np.mean(recordingsB['5;I_CBF']),2)))
plt.plot(times,recordingsB['5;I_CMRO2'],label='I_CMRO2 '+str(round(np.mean(recordingsB['5;I_CMRO2']),2)))
#plt.ylim(-1.05,1.05)
plt.xlim(times[0],times[-1])
plt.legend()

## ROW 7: BOLD 6
row=7
plt.subplot(3,rows,rows*0+row)
plt.title('Self-defined model only corE')
plt.plot(times,recordingsB['6;BOLD'])
#plt.ylim(0,0.025)
plt.xlim(times[0],times[-1])

plt.subplot(3,rows,rows*1+row)
plt.title('CBF & CMRO2')
plt.plot(times,recordingsB['6;CBF'],label='CBF')
plt.plot(times,recordingsB['6;CMRO2'],label='CMRO2')
#plt.ylim(0.5,1.8)
plt.xlim(times[0],times[-1])
plt.legend()

plt.subplot(3,rows,rows*2+row)
plt.title('I_CBF & I_CMRO2')
plt.plot(times,recordingsB['6;I_CBF'],label='I_CBF '+str(round(np.mean(recordingsB['6;I_CBF']),2)))
plt.plot(times,recordingsB['6;I_CMRO2'],label='I_CMRO2 '+str(round(np.mean(recordingsB['6;I_CMRO2']),2)))
#plt.ylim(-1.05,1.05)
plt.xlim(times[0],times[-1])
plt.legend()

## ROW 8: BOLD 7
row=8
plt.subplot(3,rows,rows*0+row)
plt.title('Self-defined model only corI')
plt.plot(times,recordingsB['7;BOLD'])
#plt.ylim(0,0.025)
plt.xlim(times[0],times[-1])

plt.subplot(3,rows,rows*1+row)
plt.title('CBF & CMRO2')
plt.plot(times,recordingsB['7;CBF'],label='CBF')
plt.plot(times,recordingsB['7;CMRO2'],label='CMRO2')
#plt.ylim(0.5,1.8)
plt.xlim(times[0],times[-1])
plt.legend()

plt.subplot(3,rows,rows*2+row)
plt.title('I_CBF & I_CMRO2')
plt.plot(times,recordingsB['7;I_CBF'],label='I_CBF '+str(round(np.mean(recordingsB['7;I_CBF']),2)))
plt.plot(times,recordingsB['7;I_CMRO2'],label='I_CMRO2 '+str(round(np.mean(recordingsB['7;I_CMRO2']),2)))
#plt.ylim(-1.05,1.05)
plt.xlim(times[0],times[-1])
plt.legend()

## ROW 9: BOLD 8
row=9
plt.subplot(3,rows,rows*0+row)
plt.title('Self-defined model only corI')
plt.plot(times,recordingsB['8;BOLD'])
#plt.ylim(0,0.025)
plt.xlim(times[0],times[-1])

plt.subplot(3,rows,rows*1+row)
plt.title('CBF & CMRO2')
plt.plot(times,recordingsB['8;CBF'],label='CBF')
plt.plot(times,recordingsB['8;CMRO2'],label='CMRO2')
#plt.ylim(0.5,1.8)
plt.xlim(times[0],times[-1])
plt.legend()

plt.subplot(3,rows,rows*2+row)
plt.title('I_CBF & I_CMRO2')
plt.plot(times,recordingsB['8;I_CBF'],label='I_CBF '+str(round(np.mean(recordingsB['8;I_CBF']),2)))
plt.plot(times,recordingsB['8;I_CMRO2'],label='I_CMRO2 '+str(round(np.mean(recordingsB['8;I_CMRO2']),2)))
#plt.ylim(-1.05,1.05)
plt.xlim(times[0],times[-1])
plt.legend()

## ROW 10: BOLD 9
row=10
plt.subplot(3,rows,rows*0+row)
plt.title('Self-defined model only corI')
plt.plot(times,recordingsB['9;BOLD'])
#plt.ylim(0,0.025)
plt.xlim(times[0],times[-1])

plt.subplot(3,rows,rows*1+row)
plt.title('CBF & CMRO2')
plt.plot(times,recordingsB['9;CBF'],label='CBF')
plt.plot(times,recordingsB['9;CMRO2'],label='CMRO2')
#plt.ylim(0.5,1.8)
plt.xlim(times[0],times[-1])
plt.legend()

plt.subplot(3,rows,rows*2+row)
plt.title('I_CBF & I_CMRO2')
plt.plot(times,recordingsB['9;I_CBF'],label='I_CBF '+str(round(np.mean(recordingsB['9;I_CBF']),2)))
plt.plot(times,recordingsB['9;I_CMRO2'],label='I_CMRO2 '+str(round(np.mean(recordingsB['9;I_CMRO2']),2)))
#plt.ylim(-1.05,1.05)
plt.xlim(times[0],times[-1])
plt.legend()

plt.savefig('simulationsANA_test.png')











