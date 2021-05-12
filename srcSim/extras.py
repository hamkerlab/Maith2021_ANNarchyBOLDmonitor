import numpy as np

def lognormalPDF(x, mu=-0.702, sigma=0.9355):
    """
        standard values from lognormal distribution of PSPs from Song et al. (2005)
    """
    return (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))

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
