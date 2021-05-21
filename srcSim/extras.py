import numpy as np
import pylab as plt
from ANNarchy import raster_plot
from scipy.stats import lognorm

def lognormalPDF(x, mu=-0.702, sigma=0.9355, shift=0):
    """
        standard values from lognormal distribution of PSPs from Song et al. (2005)
    """
    x=x-shift
    if x.min()<=0:
        return np.concatenate([ np.zeros(x[x<=0].size) , np.exp(-(np.log(x[x>0]) - mu)**2 / (2 * sigma**2)) / (x[x>0] * sigma * np.sqrt(2 * np.pi)) ])
    else:
        return np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))

    

def getFiringRateDist(spikes, dur):
    """
        spikes: spike trains of population of time period
        dur: length of time period in ms
        
        return firing rate distribution of population
    """
    rate = []
    for neuron, train in spikes.items():
        rate.append(len(train)/(dur/1000.))
    return np.array(rate)



def plot_input_and_raster(recordings, name):
    plt.figure(figsize=(18,12))
    plt.subplot(221)
    plt.title('Input current for CorE')
    for idx in range(recordings['inputPop;r'].shape[1]):
        plt.plot(recordings['inputPop;r'][:,idx])

    plt.subplot(223)
    plt.title('CorE')
    t,n = raster_plot(recordings['corEL1;spike'])
    plt.plot(t,n,'k.',markersize=0.2)

    plt.subplot(224)
    plt.title('CorI')
    t,n = raster_plot(recordings['corIL1;spike'])
    plt.plot(t,n,'k.',markersize=0.2)
    plt.savefig('../results/'+name)
            
def generateInputs(a,b,c,d,rng):
    """
        draws values from logNorm distribution
        redrwas values above the 99% percentile value
        a: shift of values
        b: mean of lognorm dist
        c: sigma of lognorm dist
        d: number of values
        rng: numpy random number generator
        
        returns: [values array, threshold]
    """


    ### get maxVal from CDF
    maxVal = lognorm.ppf(0.99,s=c,loc=0,scale=np.exp(b))
    
    keptVals=[]
    while len(keptVals)!=d:
        ### draw from disrtibution
        vals=rng.lognormal(mean=b, sigma=c, size=d-len(keptVals))
        
        ### only keep Vals below maxVal
        keptVals=keptVals+list(vals[vals<=maxVal])

    return {'values':np.array(keptVals)+a, 'threshold':maxVal+a}
    
    
