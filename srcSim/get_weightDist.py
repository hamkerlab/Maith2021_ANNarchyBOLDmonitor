from ANNarchy import *
import pylab as plt
from scipy import signal, stats
from model_neuronmodels import params, rng, Izhikevich2007RS, Izhikevich2007FS
from extras import lognormalPDF, get_log_normal_fit, set_size


### create 1000 neurons of each neurontype
corE = Population(1000, neuron=Izhikevich2007RS, name='CorE')
corI = Population(1000, neuron=Izhikevich2007FS, name='CorI')


### create spiketimearray input with one spike
spike_times = [
  [10] for i in range(1)
]
inputPop = SpikeSourceArray(spike_times=spike_times)


### direct this spike to both populations
inputPopcorE = Projection( 
    pre    = inputPop,
    post   = corE, 
    target = 'ampa',
    name = 'inputPopcorE'
).connect_all_to_all(weights = params['weightDist'](rng))#weights have specific log normal dist to obtain PSP dist

inputPopcorI = Projection( 
    pre    = inputPop,
    post   = corI, 
    target = 'ampa',
    name = 'inputPopcorI'
).connect_all_to_all(weights = params['weightDist'](rng))#weights have specific log normal dist to obtain PSP dist


"""
### Define weights so that the PSP distribution looks like the target PSP population (see extras.lognormalPDF)

#how new mu, sigma were obtained:
#inputPopcorE --> weights * 0.5
test=0.3*rng.lognormal(-0.7,0.93,10000)
shape,loc,scale = stats.lognorm.fit(test)
sigma=shape
mu=np.log(scale)
print(mu,sigma)
quit()
#inputPopcorE --> weights * 0.3-0.4
... both about mu=1.5"""



### compile
compile()



### compare weight distribution with target EPSP distribution
plt.figure()
x=np.arange(0.01,7.5,0.01)
plt.hist(np.array(inputPopcorE.w).flatten(), 100, density=True, align='mid', label='weights')
plt.plot(x, lognormalPDF(x), label='EPSPs Song et al. (2005)')
plt.xlim(0,10)
plt.legend()
plt.savefig('../results/get_weightDist/get_weightDist_weightsE_dist.svg')


### record membranepotential of both neurons
mon_v_corE = Monitor(corE, 'v')
mon_v_corI = Monitor(corI, 'v')


### simulate 100 ms
simulate(100)


### plot membrane potentials
v_corE = mon_v_corE.get('v')
v_corI = mon_v_corI.get('v')


### get peak values from membrane potentials
maxPosE=np.zeros(1000)*np.nan
maxPosI=np.zeros(1000)*np.nan
maxValE=np.zeros(1000)*np.nan
maxValI=np.zeros(1000)*np.nan
for idx in range(1000):
    if len(signal.argrelmax(v_corE[:,idx])[0])>0:
        maxPosE[idx]=signal.argrelmax(v_corE[:,idx])[0][0]
        maxValE[idx]=v_corE[int(maxPosE[idx]),idx]
    if len(signal.argrelmax(v_corI[:,idx])[0])>0:
        maxPosI[idx]=signal.argrelmax(v_corI[:,idx])[0][0]
        maxValI[idx]=v_corI[int(maxPosI[idx]),idx]



plt.figure()
plt.subplot(211)
plt.plot(v_corE[:,0],color='C0')
plt.plot(v_corE[:,1],color='C1')
plt.plot(v_corE[:,999],color='C2')
plt.axvline(maxPosE[0],color='C0')
plt.axvline(maxPosE[1],color='C1')
plt.axvline(maxPosE[999],color='C2')
plt.axhline(maxValE[0],color='C0')
plt.axhline(maxValE[1],color='C1')
plt.axhline(maxValE[999],color='C2')
plt.subplot(212)
plt.plot(v_corI[:,0],color='C0')
plt.plot(v_corI[:,1],color='C1')
plt.plot(v_corI[:,999],color='C2')
plt.axvline(maxPosI[0],color='C0')
plt.axvline(maxPosI[1],color='C1')
plt.axvline(maxPosI[999],color='C2')
plt.axhline(maxValI[0],color='C0')
plt.axhline(maxValI[1],color='C1')
plt.axhline(maxValI[999],color='C2')
plt.savefig('../results/get_weightDist/get_weightDist_PSPexamples.svg')



### calculate PSP distribution from membrane potentials

## difference between resting potential and peaks below 0
PSPsE=maxValE[maxValE<0]-params['RS_v_r']
PSPsI=maxValI[maxValE<0]-params['FS_v_r']

plt.figure()
plt.subplot(211)
plt.hist(PSPsE[PSPsE<=10], 100, density=True, align='mid', label='EPSPs in CorE')
plt.plot(x,lognormalPDF(x), label='EPSPs Song et al. (2005)')
plt.xlim(0,10)
plt.legend()
plt.subplot(212)
plt.hist(PSPsI[PSPsI<=10], 100, density=True, align='mid', label='EPSPs in CorI')
plt.plot(x,lognormalPDF(x), label='EPSPs Song et al. (2005)')
plt.xlim(0,10)
plt.legend()
plt.savefig('../results/get_weightDist/get_weightDist_PSP_dist.svg')



### COMPARISON COMBINED PSP DISTRIBUTION SIMULATION (E+I) VS EXPERIMENTAL

## COMBINE PSP DISTRIBUTIONS OF E AND I
PSP_dist = np.concatenate([PSPsE, PSPsI])

### OBTAIN FITTED LOG NORMAL DISTRIBUTION
fit = get_log_normal_fit(PSP_dist)

## PLOT
plt.figure()
plt.hist(PSP_dist[PSP_dist<=10], 100, density=True, align='mid', color='grey', alpha=0.5)
plt.plot(x,lognormalPDF(x), label='Song et al. (2005)', color='k')
plt.plot(x, lognormalPDF(x, mu=fit[0], sigma=fit[1], shift=fit[2]), label='Model', color='red', ls='dashed')
plt.legend(fontsize=8)
set_size(5.93/2.54,2.44/2.54)
plt.savefig('../results/get_weightDist/get_weightDist_combined_PSP_dist.svg')














