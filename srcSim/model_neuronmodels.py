from ANNarchy import *
from parameters import params, rng

setup(dt=params['dt'])
setup(num_threads=params['num_threads'])
setup(seed=params['seed'])
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
        E_gaba   = 'E_gaba'    : population
        
        tau_syn  = 'tau_syn'   : population
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa : init = 0
        dg_gaba/dt = -g_gaba/tau_gaba : init = 0
        I_ampa     = -g_ampa*(v - E_ampa)
        I_gaba     = -g_gaba*(v - E_gaba)
        I          = g_exc + I_ampa + I_gaba
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I : init = RS_v_r
        du/dt      = a*(b*(v - v_r) - u) : init = 0
        
        tau_syn*dsyn/dt = -syn
        var_f  = pos(I_ampa) + 1.5*neg(I_gaba)
        var_fa  = pos(I_ampa) + 1.5*neg(I_gaba)**2
        var_r  = pos(I_ampa) 
        var_ra = pos(I_ampa) 
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
        E_gaba   = 'E_gaba'    : population
        
        tau_syn  = 'tau_syn'   : population
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa : init = 0
        dg_gaba/dt = -g_gaba/tau_gaba : init = 0
        I_ampa     = -g_ampa*(v - E_ampa)
        I_gaba     = -g_gaba*(v - E_gaba)
        I          = g_exc + I_ampa + I_gaba
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I : init = FS_v_r
        U_v        = if v<v_b: 0 else: b*(v - v_b)**3
        du/dt      = a*(U_v - u) : init = 0
        
        tau_syn*dsyn/dt = -syn
        var_f  = pos(I_ampa) + 1.5*neg(I_gaba)
        var_fa  = pos(I_ampa) + 1.5*neg(I_gaba)**2
        var_r  = pos(I_ampa)
        var_ra = r
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

InputPoissonNeuron = Neuron(
    parameters="""
        tau         = 'input_tau' : population
        offsetVal   = 0
        increaseVal = 0
    """,
    equations = """
        p = Uniform(0.0, 1.0) * 1000.0 / dt
        rate=increaseVal*t/tau+offsetVal
    """,
    spike = """
        p <= rate
        """,
    reset = """
       p=0.0
    """,
    name = "InputPoissonNeuron",
    description = "Spiking rate of Poisson neuron increases linearly, proportionally to offset.",
    extra_values=params
)

#new standard model
# damped harmonic oscillators, k->timeconstant, c->damping
# CBF --> try k from Friston
# CMRO2 --> faster --> k=k_CBF*2 (therefore scaling of I_CMRO2 by (k_CMRO2 / k_CBF) --> if same input (I_CBF==I_CMRO2) CMRO2 and CBF same steady-state)
# critical c --> c**2-4k = 0 --> c=sqrt(4k)
# CBF underdamped for undershoot --> c = 0.4*sqrt(4k)
# CMRO2 critical --> c = sqrt(4k)
# after CBF and CMRO2 standard balloon model with revised coefficients, parameter values = Friston et al. (2000)
newBoldNeuron = Neuron(
parameters = """
    c_CBF       = 1
    k_CBF       = 1
    c_CMRO2     = 1
    k_CMRO2     = 1
    ea          = 1.0
    E_0         = 0.34
    tau_0       = 0.98
    alpha       = 0.33
    V_0         = 0.02
    v_0         = 40.3
    TE          = 40/1000.
    epsilon     = 1
    r_0         = 25
""",
equations = """
    I_CBF           = sum(I_f)                                                                    : init=0
    I_CMRO2         = sum(I_r)                                                                    : init=0
    1000*dsCBF/dt   = ea * I_CBF - c_CBF * sCBF - k_CBF * (CBF - 1)                               : init=0
    1000*dCBF/dt    = sCBF                                                                        : init=1, min=0
    1000*dsCMRO2/dt = ea * I_CMRO2 * (k_CMRO2 / k_CBF) - c_CMRO2 * sCMRO2 - k_CMRO2 * (CMRO2 - 1) : init=0
    1000*dCMRO2/dt  = sCMRO2                                                                      : init=1, min=0

    1000*dq/dt      = 1 / tau_0 * (CMRO2 - (q / v) * f_out)                                       : init=1
    1000*dv/dt      = 1 / tau_0 * (CBF - f_out)                                                   : init=1
    f_out           = v**(1 / alpha)                                                              : init=1

    k_1             = 4.3 * v_0 * E_0 * TE
    k_2             = epsilon * r_0 * E_0 * TE
    k_3             = 1 - epsilon
    BOLD            = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v))                 : init=0
    r=0
""",
    name = "-",
    description = "-"
)


### BOLD neuron only for single input recording
BoldNeuron_r = Neuron(
parameters = """
""",
equations = """
    r             = sum(exc)                                                    : init=0
    I_CBF         = sum(I_f)                                                    : init=0
    I_CMRO2       = sum(I_r)                                                    : init=0
"""
)
