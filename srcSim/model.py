from ANNarchy import *
from model_neuronmodels import params, rng, Izhikevich2007RS, Izhikevich2007FS, InputNeuron

####################################################################################################################################
####################################################   POPULATIONS   ###############################################################
####################################################################################################################################

corEL1   = Population(params['corE_popsize'], neuron=Izhikevich2007RS, name='corEL1')
corIL1   = Population(params['corE_popsize']//4, neuron=Izhikevich2007FS, name='corIL1')
inputPop = Population(params['corE_popsize'], neuron=InputNeuron, name='inputPop')
inputPop.offsetVal=params['inputPop_init_offsetVal']
inputPop.increaseVal=params['inputPop_init_increaseVal']


####################################################################################################################################
####################################################   PROJECTIONS   ###############################################################
####################################################################################################################################

### Input
inputPop_corEL1 = CurrentInjection(inputPop, corEL1, 'exc', name='inputPop_corEL1')
inputPop_corEL1.connect_current()

### Inh loop in L1
corEL1_corIL1 = Projection(pre=corEL1, post=corIL1, target='ampa', name='corEL1_corIL1')
corIL1_corEL1 = Projection(pre=corIL1, post=corEL1, target='gaba', name='corIL1_corEL1')
corIL1_corIL1 = Projection(pre=corIL1, post=corIL1, target='gaba', name='corIL1_corIL1')
if params['optimizeRates']:
    ### connect_all_to_all so that the weights can be manipulated (set to zero for varying pre synapse numbers)
    corEL1_corIL1.connect_all_to_all(weights=params['weightDist'])#.connect_fixed_number_pre(number = params['numInputs'], weights=params['weightDist'])
    corIL1_corEL1.connect_all_to_all(weights=params['weightDist'])#.connect_fixed_number_pre(number = params['numInputs'], weights=params['weightDist'])
    corIL1_corIL1.connect_all_to_all(weights=params['weightDist'])#.connect_fixed_number_pre(number = params['numInputs'], weights=params['weightDist'])
else:
    ### connect fixed number pre with optimized number
    corEL1_corIL1.connect_fixed_number_pre(number = params['numInputs'], weights=params['weightDist'])
    corIL1_corEL1.connect_fixed_number_pre(number = params['numInputs'], weights=params['weightDist'])
    corIL1_corIL1.connect_fixed_number_pre(number = params['numInputs'], weights=params['weightDist'])











