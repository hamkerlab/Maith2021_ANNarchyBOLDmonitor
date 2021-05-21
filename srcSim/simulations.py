from ANNarchy import *
import pylab as plt
from model import params, rng
from extras import getFiringRateDist, lognormalPDF, plot_input_and_raster

monDict={'pop;inputPop':['r'],
         'pop;corEL1':['v', 'spike'],
         'pop;corIL1':['v', 'spike']}

mon={}
for key, val in monDict.items():
    compartmentType, compartment = key.split(';')
    if compartmentType=='pop':
        mon[compartment] = Monitor(get_population(compartment),val)
        

compile()

simulate(20000)

recordings={}
for key, val in monDict.items():
    compartmentType, compartment = key.split(';')
    for val_val in val:
        recordings[compartment+';'+val_val] = mon[compartment].get(val_val)

### plot 1
plot_input_and_raster(recordings, 'simulations_input_and_raster.png')


### plot 2
for pop in ['corEL1', 'corIL1']:
    recordings[pop+';rateDist']=getFiringRateDist(recordings[pop+';spike'], 10000)

plt.figure()
x=np.logspace(-3,2,100)

plt.subplot(211)
plt.title('E')
plt.hist(recordings['corEL1;rateDist'], 100, density=True, align='mid', label='simulatedDist')
plt.plot(x, lognormalPDF(x, mu=1.2, sigma=1.1), label='targetDist (Buzsaki & Mizuseki, 2014)')
#plt.xlim(10**-3,30)

plt.subplot(212)
plt.title('I')
plt.hist(recordings['corIL1;rateDist'], 100, density=True, align='mid', label='simulatedDist')
plt.plot(x, lognormalPDF(x, mu=1.2, sigma=1.1), label='targetDist (Buzsaki & Mizuseki, 2014)')
#plt.xlim(10**-3,30)

plt.savefig('../results/simulations_rateDists.svg')


