from ANNarchy import *
from parameters import params, rng


setup(dt=params['dt'])
setup(num_threads=params['num_threads'])
Constant('RS_v_r',params['RS_v_r'])
Constant('FS_v_r',params['FS_v_r'])



####################################################################################################################################
##################################################   NEURON MODELS   ###############################################################
####################################################################################################################################

Izhikevich2007RS = Neuron(
    parameters="""
        C        = 'RS_C'      : population
        k        = 'RS_k'      : population
        v_r      = 'RS_v_r'    : population
        v_t      = 'RS_v_t'    : population
        a        = 'RS_a'      : population
        b        = 'RS_b'      : population
        c        = 'RS_c'      : population
        d        = 'RS_d'      : population
        v_peak   = 'RS_v_peak' : population
        tau_ampa = 'tau_ampa'  : population
        tau_gaba = 'tau_gaba'  : population
        E_ampa   = 'E_ampa'    : population
        E_gaba   = 'E_ampa'    : population
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa : init = 0
        dg_gaba/dt = -g_gaba/tau_gaba : init = 0
        I          = g_exc - g_ampa*(v - E_ampa) - g_gaba*(v - E_gaba)
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I : init = RS_v_r
        du/dt      = a*(b*(v - v_r) - u) : init = 0
    """,
    spike = "v >= v_peak",
    reset = """
        v = c
        u = u + d
    """,
    name = "Izhikevich2007RS",
    description = "RS cortical neuron model from Izhikevich (2007) with additional conductance based synapses.",
    extra_values=params
)

Izhikevich2007FS = Neuron(
    parameters="""
        C        = 'FS_C'      : population
        k        = 'FS_k'      : population
        v_r      = 'FS_v_r'    : population
        v_t      = 'FS_v_t'    : population
        v_b      = 'FS_v_b'    : population
        a        = 'FS_a'      : population
        b        = 'FS_b'      : population
        c        = 'FS_c'      : population
        d        = 'FS_d'      : population
        v_peak   = 'FS_v_peak' : population
        tau_ampa = 'tau_ampa'  : population
        tau_gaba = 'tau_gaba'  : population
        E_ampa   = 'E_ampa'    : population
        E_gaba   = 'E_ampa'    : population
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa : init = 0
        dg_gaba/dt = -g_gaba/tau_gaba : init = 0
        I          = g_exc - g_ampa*(v - E_ampa) - g_gaba*(v - E_gaba)
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I : init = FS_v_r
        U_v        = if v<v_b: 0 else: b*(v - v_b)**3
        du/dt      = a*(U_v - u) : init = 0
    """,
    spike = "v >= v_peak",
    reset = """
        v = c
        u = u + d
    """,
    name = "Izhikevich2007FS",
    description = "FS cortical interneuron model from Izhikevich (2007) with additional conductance based synapses.",
    extra_values=params
)

InputNeuron = Neuron(
    parameters="""
        tau         = 'input_tau' : population
        offsetVal   = 0
        increaseVal = 0
    """,
    equations="""
        r=increaseVal*t/tau+offsetVal
    """,
    name = "InputNeuron",
    description = "Rate of neuron increases linearly, proportionally to offset.",
    extra_values=params
)



####################################################################################################################################
####################################################   POPULATIONS   ###############################################################
####################################################################################################################################

corEL1   = Population(params['corE_popsize'], neuron=Izhikevich2007RS, name='corEL1')
corIL1   = Population(params['corE_popsize']//4, neuron=Izhikevich2007FS, name='corIL1')
corEL2   = Population(params['corE_popsize'], neuron=Izhikevich2007RS, name='corEL2')
corIL2   = Population(params['corE_popsize']//4, neuron=Izhikevich2007FS, name='corIL2')
inputPop = Population(params['corE_popsize'], neuron=InputNeuron, name='inputPop')
inputPop.offsetVal=params['inputPop_init_offsetVal']
inputPop.increaseVal=params['inputPop_init_increaseVal']
### TODO inputs should have offset


####################################################################################################################################
####################################################   PROJECTIONS   ###############################################################
####################################################################################################################################

### Input
inputPop_corEL1 = CurrentInjection(inputPop, corEL1, 'exc', name='inputPop_corEL1')
inputPop_corEL1.connect_current()

### Exc transmission from L1 to L2
corEL1_corEL2 = Projection(pre=corEL1, post=corEL2, target='ampa', name='corEL1_corEL2')
corEL1_corEL2.connect_all_to_all(weights = params['weightDist'])

### Inh loop in L1
"""corEL1_corIL1 = Projection(pre=corEL1, post=corIL1, target='ampa', name='corEL1_corIL1')
corEL1_corIL1.connect_all_to_all(weights = params['weightDist'])

corIL1_corEL1 = Projection(pre=corIL1, post=corEL1, target='gaba', name='corIL1_corEL1')
corIL1_corEL1.connect_all_to_all(weights = params['weightDist'])

corIL1_corIL1 = Projection(pre=corIL1, post=corIL1, target='gaba', name='corIL1_corIL1')
corIL1_corIL1.connect_all_to_all(weights = params['weightDist'])

### Inh loop in L2
corEL2_corIL2 = Projection(pre=corEL2, post=corIL2, target='ampa', name='corEL2_corIL2')
corEL2_corIL2.connect_all_to_all(weights = params['weightDist'])

corIL2_corEL2 = Projection(pre=corIL2, post=corEL2, target='gaba', name='corIL2_corEL2')
corIL2_corEL2.connect_all_to_all(weights = params['weightDist'])

corIL2_corIL2 = Projection(pre=corIL2, post=corIL2, target='gaba', name='corIL2_corIL2')
corIL2_corIL2.connect_all_to_all(weights = params['weightDist'])"""















