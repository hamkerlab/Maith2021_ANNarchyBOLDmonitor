from ANNarchy import LogNormal, Uniform
import numpy as np
from extras import generateInputs, scaledANNarchyLogNormal


params={}
### random number generator
params['seed'] = None# int or None
if params['seed']!=None:
    rng = np.random.default_rng(params['seed'])
else:
    rng = np.random.default_rng()
    params['seed'] = rng.integers(0,2**32 - 1)
    rng = np.random.default_rng(params['seed'])

### general ANNarchy params
params['dt'] = 0.1
params['num_threads'] = 1
params['optimizeRates'] = ['v1','v1post','v2','v2post'][3]# v1 = optimized input current distribution and number-pre-fix, v2 = optimize weight scalings, post = not fitting but use fitted values
params['increasingInputs'] = False

if 'v1' in params['optimizeRates']:
    ## define if Poisson or Current input, current input is fitted in v1
    params['input']= ['Current','Poisson'][0]
elif 'v2' in params['optimizeRates']:
    ## in v2 input is Poisson population
    params['input'] = 'Poisson'

if params['optimizeRates']=='v1':
    ## just use some values for the parameters... they will be overwritten during optimization
    params['fittedParams']= {'shift':60, 'mean':1.2, 'sigma':1.1, 'number synapses':20}
elif params['optimizeRates']=='v1post':
    ## load fitted parameters
    params['useFit'] = 13
    params['fittedParams'] = np.load('../dataRaw/optimize_rates_obtainedParams'+str(params['useFit'])+'.npy', allow_pickle=True).item()
elif params['optimizeRates']=='v2':
    ## just use some values for the parameters... they will be overwritten during optimization
    params['fittedParams']= {'S_INP':1, 'S_EI':1, 'S_IE':1, 'S_II':1}
elif params['optimizeRates']=='v2post':
    """
        v2 --> weight scalings were fitted
        1-10:  optimized for: params['numInputs'] = params['corE_popsize']//4 - 1
        11-20: optimized for: params['numInputs'] = 10
    """
    
    #for i in range(11,21):
    #    print(i,np.load('../dataRaw/optimize_ratesv2_obtainedParams'+str(i)+'.npy', allow_pickle=True).item())
    
    params['useFit'] = 15#4
    params['fittedParams'] = np.load('../dataRaw/optimize_ratesv2_obtainedParams'+str(params['useFit'])+'.npy', allow_pickle=True).item()
    print(params['fittedParams'])

### Neuron models
## conductance based synapses
params['tau_ampa'] = 10
params['tau_gaba'] = 10
params['E_ampa']   = 0
params['E_gaba']   = -90
params['tau_syn']  = 10
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
## Izhikevich FS neuron
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
if params['input']=='Current' and ('v1' in params['optimizeRates']):
    # input current distribution
    params['inputPop_init_offsetVal'] = generateInputs(params['fittedParams']['shift'],params['fittedParams']['mean'],params['fittedParams']['sigma'],params['corE_popsize'],rng)['values']
elif params['input']=='Poisson' and ('v1' in params['optimizeRates']):
    # firing rates of all poisson neurons, in v1 all poisson input neurons have the same rate
    params['inputPop_init_offsetVal'] = 500
elif 'v2' in params['optimizeRates']:
    # v2 --> input = poisson with lognormal distributed firing rates, distribution with mean 1.2 and sigma 1.1, similar to firing rate distributions shown in Buzsaki & Mizuseki 2014
    params['inputPop_init_offsetVal'] = generateInputs(0,1.2,1.1,params['corE_popsize'],rng)['values']

if params['increasingInputs']:
    # the input does linearly increase with time t
    params['inputPop_init_increaseVal'] = params['inputPop_init_offsetVal']
else:
    params['inputPop_init_increaseVal'] = 0


### Projections
params['weightDist'] = scaledANNarchyLogNormal

if 'v1' in params['optimizeRates']:
    params['numInputs']  = params['fittedParams']['number synapses']
elif 'v2' in params['optimizeRates']:
    # if model v2 --> use max possible num-pre-fix for all projections, or 10
    params['numInputs']  = 10#params['corE_popsize']//4 - 1


