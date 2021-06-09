from ANNarchy import *
from model_neuronmodels import params, rng, Izhikevich2007RS, Izhikevich2007FS, InputNeuron, newBoldNeuron, InputPoissonNeuron, BoldNeuron_r
from ANNarchy.extensions.bold import *

####################################################################################################################################
####################################################   POPULATIONS   ###############################################################
####################################################################################################################################

corEL1   = Population(params['corE_popsize'], neuron=Izhikevich2007RS, name='corEL1')
corIL1   = Population(params['corE_popsize']//4, neuron=Izhikevich2007FS, name='corIL1')
if params['input']=='Current':
    inputPop = Population(params['corE_popsize'], neuron=InputNeuron, name='inputPop')
elif params['input']=='Poisson':
    inputPop = Population(params['corE_popsize'], neuron=InputPoissonNeuron, name='inputPop')
inputPop.offsetVal=params['inputPop_init_offsetVal']
inputPop.increaseVal=params['inputPop_init_increaseVal']


####################################################################################################################################
####################################################   PROJECTIONS   ###############################################################
####################################################################################################################################

if 'v1' in params['optimizeRates']:



    ### Input
    if params['input']=='Current':
        # current injection to corE
        inputPop_corEL1 = CurrentInjection(inputPop, corEL1, 'exc', name='inputPop_corEL1')
        inputPop_corEL1.connect_current()
    elif params['input']=='Poisson':
        # one to one ampa poisson input to corE
        inputPop_corEL1 = NormProjection(pre=inputPop, post=corEL1, target='ampa', variable = 'syn', name='inputPop_corEL1')
        inputPop_corEL1.connect_one_to_one(weights=params['weightDist'](rng))    



    ### Inh loop in L1
    corEL1_corIL1 = NormProjection(pre=corEL1, post=corIL1, target='ampa', variable = 'syn', name='corEL1_corIL1')
    corIL1_corEL1 = NormProjection(pre=corIL1, post=corEL1, target='gaba', variable = 'syn', name='corIL1_corEL1')
    corIL1_corIL1 = NormProjection(pre=corIL1, post=corIL1, target='gaba', variable = 'syn', name='corIL1_corIL1')
    if params['optimizeRates'] == 'v1':
        # connect_all_to_all so that the weights can be manipulated (set to zero for varying pre synapse numbers)
        corEL1_corIL1.connect_all_to_all(weights=params['weightDist'](rng))
        corIL1_corEL1.connect_all_to_all(weights=params['weightDist'](rng))
        corIL1_corIL1.connect_all_to_all(weights=params['weightDist'](rng))
    elif params['optimizeRates'] == 'v1post':
        # connect fixed number pre with optimized number
        corEL1_corIL1.connect_fixed_number_pre(number = params['numInputs'], weights=params['weightDist'](rng))
        corIL1_corEL1.connect_fixed_number_pre(number = params['numInputs'], weights=params['weightDist'](rng))
        corIL1_corIL1.connect_fixed_number_pre(number = params['numInputs'], weights=params['weightDist'](rng))



### Projections for v2
def add_scaled_projections(S_INP, S_EI, S_IE, S_II, rng):
    """
        generating the scaled projections of model v2
        
        run this before between population definition and BOLD monitor definition
        
        input arguments: scale factors for weights
    """
    if ('v2' in params['optimizeRates']) == False:
        print('model v2 necessary to add scaled projections')
        quit()
    else:
        ### Input
        ### fixed-num-pre ampa poisson input to corE and corI
        inputPop_corEL1 = NormProjection(pre=get_population('inputPop'), post=get_population('corEL1'), target='ampa', variable = 'syn', name='inputPop_corEL1')
        inputPop_corIL1 = NormProjection(pre=get_population('inputPop'), post=get_population('corIL1'), target='ampa', variable = 'syn', name='inputPop_corIL1')
        inputPop_corEL1.connect_fixed_number_pre(number = params['numInputs'], weights=params['weightDist'](rng, S_INP))
        inputPop_corIL1.connect_fixed_number_pre(number = params['numInputs'], weights=params['weightDist'](rng, S_INP))
        
        ### Inh loop in L1
        corEL1_corIL1 = NormProjection(pre=get_population('corEL1'), post=get_population('corIL1'), target='ampa', variable = 'syn', name='corEL1_corIL1')
        corIL1_corEL1 = NormProjection(pre=get_population('corIL1'), post=get_population('corEL1'), target='gaba', variable = 'syn', name='corIL1_corEL1')
        corIL1_corIL1 = NormProjection(pre=get_population('corIL1'), post=get_population('corIL1'), target='gaba', variable = 'syn', name='corIL1_corIL1')
        corEL1_corIL1.connect_fixed_number_pre(number = params['numInputs'], weights=params['weightDist'](rng, S_EI))
        corIL1_corEL1.connect_fixed_number_pre(number = params['numInputs'], weights=params['weightDist'](rng, S_IE))
        corIL1_corIL1.connect_fixed_number_pre(number = params['numInputs'], weights=params['weightDist'](rng, S_II))











