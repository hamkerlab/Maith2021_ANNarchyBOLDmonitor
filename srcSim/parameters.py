from ANNarchy import LogNormal, Uniform
import numpy as np

rng = np.random.default_rng()

params={}
### general ANNarchy params
params['dt'] = 0.1
params['num_threads'] = 1

### Neuron models
## conductance based synapses
params['tau_ampa'] = 10
params['tau_gaba'] = 10
params['E_ampa']   = 0
params['E_gaba']   = -90
## Izhikevich RS neuron
params['RS_C']      = 100
params['RS_k']      = 0.7
params['RS_v_r']    = -60
params['RS_v_t']    = -40
params['RS_a']      = 0.03
params['RS_b']      = -2
params['RS_c']      = -50
params['RS_d']      = 100
params['RS_v_peak'] = 35
## Izhikevich RS neuron
params['FS_C']      = 20
params['FS_k']      = 1
params['FS_v_r']    = -55
params['FS_v_t']    = -40
params['FS_v_b']    = -55
params['FS_a']      = 0.2
params['FS_b']      = 0.025
params['FS_c']      = -45
params['FS_d']      = 0
params['FS_v_peak'] = 25
## Input neurons
params['input_tau'] = 10000#how many miliseconds to increase input current by the value offset

### Populations
params['corE_popsize'] = 200
params['inputPop_init_offsetVal'] = 60+rng.lognormal(mean=1.2, sigma=1.1, size=params['corE_popsize'])# lognormal dist with peak at 1 with height 0.2
#params['inputPop_init_offsetVal'] = rng.normal(loc=60, scale=15, size=params['corE_popsize'])
params['inputPop_init_increaseVal'] = 0#params['inputPop_init_offsetVal']

### Projections
params['weightDist'] = LogNormal(mu=-1.5, sigma=0.93, max=2)
params['numInputs']  = 20

