import numpy as np
import pylab as plt
from ANNarchy import raster_plot

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    
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
