from ANNarchy import *
import pylab as plt
from model import params, rng

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

simulate(10000)

recordings={}
for key, val in monDict.items():
    compartmentType, compartment = key.split(';')
    for val_val in val:
        recordings[compartment+';'+val_val] = mon[compartment].get(val_val)


plt.figure()
plt.subplot(321)
for idx in range(recordings['inputPop;r'].shape[1]):
    plt.plot(recordings['inputPop;r'][:,idx])

plt.subplot(323)
t,n = raster_plot(recordings['corEL1;spike'])
plt.plot(t,n,'k.')

plt.subplot(324)
t,n = raster_plot(recordings['corIL1;spike'])
plt.plot(t,n,'k.')

plt.subplot(325)
t,n = raster_plot(recordings['corEL2;spike'])
plt.plot(t,n,'k.')

plt.subplot(326)
t,n = raster_plot(recordings['corIL2;spike'])
plt.plot(t,n,'k.')
plt.savefig('test.png')


