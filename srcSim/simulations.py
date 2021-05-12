from ANNarchy import *
import pylab as plt
from model import params, rng
from extras import getFiringRateDist
from extras import lognormalPDF

monDict={'pop;inputPop':['r'],
         'pop;corEL1':['v', 'spike'],
         'pop;corEL2':['v', 'spike'],
         'pop;corIL1':['v', 'spike'],
         'pop;corIL2':['v', 'spike']}

mon={}
for key, val in monDict.items():
    compartmentType, compartment = key.split(';')
    if compartmentType=='pop':
        mon[compartment] = Monitor(get_population(compartment),val)
        

compile()
#get_projection('corEL1_corIL1').w=rng.lognormal(-1.5,0.93,(params['corE_popsize']//4,params['numInputs']))
#get_projection('corIL1_corEL1').w=rng.lognormal(-1.5,0.93,(params['corE_popsize'],params['numInputs']))
#get_projection('corIL1_corIL1').w=rng.lognormal(-1.5,0.93,(params['corE_popsize']//4,params['numInputs']))

#get_projection('corEL1_corIL1').w=np.zeros((params['corE_popsize']//4,params['corE_popsize']))
get_projection('corIL1_corEL1').w=np.zeros((params['corE_popsize'],params['corE_popsize']//4))
#get_projection('corIL1_corIL1').w=np.zeros((params['corE_popsize']//4,params['corE_popsize']//4-1))

simulate(20000)

recordings={}
for key, val in monDict.items():
    compartmentType, compartment = key.split(';')
    for val_val in val:
        recordings[compartment+';'+val_val] = mon[compartment].get(val_val)


plt.figure(figsize=(18,12))
plt.subplot(321)
for idx in range(recordings['inputPop;r'].shape[1]):
    plt.plot(recordings['inputPop;r'][:,idx])

plt.subplot(323)
t,n = raster_plot(recordings['corEL1;spike'])
plt.plot(t,n,'k.',markersize=0.2)

plt.subplot(324)
t,n = raster_plot(recordings['corIL1;spike'])
plt.plot(t,n,'k.',markersize=0.2)

plt.subplot(325)
t,n = raster_plot(recordings['corEL2;spike'])
plt.plot(t,n,'k.',markersize=0.2)

plt.subplot(326)
t,n = raster_plot(recordings['corIL2;spike'])
plt.plot(t,n,'k.',markersize=0.2)
plt.savefig('test1.png')



for pop in ['corEL1', 'corEL2', 'corIL1', 'corIL2']:
    recordings[pop+';rateDist']=getFiringRateDist(recordings[pop+';spike'], 10000)

plt.figure()
x=np.logspace(-3,2,100)

plt.subplot(221)
plt.title('E')
plt.hist(recordings['corEL1;rateDist'], 100, density=True, align='mid')
plt.plot(x, lognormalPDF(x, mu=1.2, sigma=1.1))
#plt.xlim(10**-3,30)

plt.subplot(222)
plt.title('I')
plt.hist(recordings['corIL1;rateDist'], 100, density=True, align='mid')
plt.plot(x, lognormalPDF(x, mu=1.2, sigma=1.1))
#plt.xlim(10**-3,30)

plt.subplot(223)
plt.hist(recordings['corEL2;rateDist'], 100, density=True, align='mid')
plt.plot(x, lognormalPDF(x, mu=1.2, sigma=1.1))
#plt.xlim(10**-3,30)

plt.subplot(224)
plt.hist(recordings['corIL2;rateDist'], 100, density=True, align='mid')
plt.plot(x, lognormalPDF(x, mu=1.2, sigma=1.1))
#plt.xlim(10**-3,30)
plt.savefig('test2.svg')


