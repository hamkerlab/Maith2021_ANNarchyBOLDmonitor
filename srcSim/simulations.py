from ANNarchy import *
import pylab as plt
from model import params, rng
from extras import getFiringRateDist, lognormalPDF, plot_input_and_raster, addMonitors, startMonitors, getMonitors



###################################################   MONITORS   ####################################################
monDict={'pop;inputPop':['r'],
         'pop;corEL1':['syn', 'spike'],
         'pop;corIL1':['syn', 'spike']}
mon={}
mon=addMonitors(monDict,mon)



#################################################   BOLDMONITORS   ##################################################
monB={}
### STANDARD BOLDMONITOR WITHOUT ANY OPTIONALS --> JUST DEFINE POPULATIONS AND INPUT VARIABLE
monB['1'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corEL1')],
                        input_variables="syn")
                            
### ALSO RECORD INPUT (r) OF BOLDNEURON
monB['2'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corEL1')],
                        input_variables="syn",
                        recorded_variables=["BOLD", "r"])
                        
### SCALE THE POPULATION SIGNALS EQUALLY
monB['3'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corEL1')],
                        scale_factor=[1,1],
                        input_variables="syn",
                        recorded_variables=["BOLD", "r"])
                        
### NORMALIZE THE POPULATION SIGNALS WITH BASELINE OVER 1000 ms
monB['4'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corEL1')],
                        normalize_input=[1000,1000],
                        input_variables="syn",
                        recorded_variables=["BOLD", "r"])
                        
### USE SELF DEFINED POPULATION SIGNALS (input_variables + BOLD_MODEL (output_variables + bold_model)
monB['5'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corEL1')],
                        normalize_input=[1000,1000],
                        input_variables=["var_f","var_r"],
                        output_variables=["I_f","I_r"],
                        bold_model=BoldNeuron_new,
                        recorded_variables=["I_CBF","I_CMRO2","CBF","CMRO2","BOLD_Balloon","BOLD_Davis"])



####################################################   COMPILE   ####################################################
compile()

### INITIALIZE PARAMETERS OF OWN BOLD MODEL
kCBF = 1/2.46
kCMRO2 = 2*kCBF
net.get(m_bold_Neuron_NEW).k_CBF=kCBF
net.get(m_bold_Neuron_NEW).k_CMRO2=kCMRO2
net.get(m_bold_Neuron_NEW).c_CBF=0.6*np.sqrt(4*kCBF)
net.get(m_bold_Neuron_NEW).c_CMRO2=np.sqrt(4*kCMRO2)



##################################################   SIMULATION   ###################################################

### RAMP_UP FOR MODEL
simulate(1000)

### START MONITORS
startMonitors(monDict,mon)### TODO


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


