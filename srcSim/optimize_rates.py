from ANNarchy import *
import pylab as plt
from scipy import stats
import sys
from model import params, rng
from extras import getFiringRateDist, lognormalPDF, plot_input_and_raster, addMonitors, startMonitors, getMonitors, generateInputs

"""
    this input current optimization + number of synapses optimization needs:
        params['optimizeRates'] = True
        params['increasingInputs'] = False
        params['input']='Current'
"""

# hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.pyll import scope

# multiprocessing
from multiprocessing import Process
import multiprocessing


### standard model compilation
simID=int(sys.argv[1])
mode=str(sys.argv[2])
compile('annarchy'+str(simID))


def set_number_of_synapses(weights, number):
    """
        weights: list (for each post neuron) of lists (list of weights to all pre neurons)
        
        For each postneuron (idx 0 in weights) sets random weights to zero, "number" weights are kept. Therefore, each neuron only has "number" effective inputs.
    """
    for idx, w in enumerate(weights):
        ### select "number" random indices in weight vector
        keep_weights_idx=rng.choice(np.arange(0,len(weights[idx]),1).astype(int),np.amin([len(weights[idx]),number]),replace=False)
        mask=np.zeros(len(weights[idx]))
        mask[keep_weights_idx]=1
        ### multiply weight vector with zero, except the "number" values
        weights[idx]=list(np.array(weights[idx])*mask.astype(int))

    return weights

### DEFINE SIMULATOR
def simulator(fitparams, m_list=[0,0]):
    """
        fitparams: list
            fitparams[0]: shift of logNormal distribution for the input current values of CorE
            fitparams[1]: mean of logNormal distribution for the input current values of CorE
            fitparams[2]: sigma of logNormal distribution for the input current values of CorE
            fitparams[3]: general number of input synapses of all neurons
            
        m_list: variable to store results, from multiprocessing
    """
    ### should plots be generated?
    plotting=True

    ### get params
    init_offsetVal = generateInputs(fitparams[0],fitparams[1],fitparams[2],params['corE_popsize'],rng)['values']
    number_of_synapses = int(fitparams[3])

    ### reset model to compilation state
    reset()

    ### create monitors
    monDict={'pop;inputPop':['r'],
             'pop;corEL1':['v', 'spike'],
             'pop;corIL1':['v', 'spike']}
    mon={}
    mon = addMonitors(monDict,mon)
    

    ### Set input strenght
    get_population('inputPop').offsetVal=init_offsetVal

    ### Set number of synapses in projections = set all weights except number of synapses to zero (non-zero weights=random)
    get_projection('corEL1_corIL1').w=set_number_of_synapses(get_projection('corEL1_corIL1').w, number_of_synapses)
    get_projection('corIL1_corEL1').w=set_number_of_synapses(get_projection('corIL1_corEL1').w, number_of_synapses)
    get_projection('corIL1_corIL1').w=set_number_of_synapses(get_projection('corIL1_corIL1').w, number_of_synapses)

    ### simulate
    startMonitors(monDict,mon)
    simulate(20000)
    
    ### get monitors
    recordings={}
    recordings=getMonitors(monDict,mon,recordings)
    

    ### get firing rate distributions
    loss=0
    obtained=[]
    for pop in ['corEL1', 'corIL1']:
        dist=getFiringRateDist(recordings[pop+';spike'], 20000)

        ### fit lognorm to dist
        if len(dist[dist>0])>0:
            shape,loc,scale = stats.lognorm.fit(dist[dist>0])
            sigma=shape
            mu=np.log(scale)
        else:
            mu, sigma, loc = 0, 0.1, -2
        obtained.append(np.array([mu,sigma,loc]))

        ### compute loss
        x = np.linspace(0,100,1000)
        target = lognormalPDF(x, mu=1.2, sigma=1.1)
        ist = lognormalPDF(x, mu=mu, sigma=sigma, shift=loc)

        loss+=np.sqrt(np.sum((ist - target)**2))

    if plotting:
        ### plot1
        plot_input_and_raster(recordings, 'optimize_rates_input_and_raster_'+str(simID)+'.png')
        
        ### plot2
        for pop in ['corEL1', 'corIL1']:
            recordings[pop+';rateDist']=getFiringRateDist(recordings[pop+';spike'], 20000)

        plt.figure()
        x=np.linspace(0,100,1000)

        plt.subplot(211)
        plt.title('E')
        plt.hist(recordings['corEL1;rateDist'], 100, density=True, align='mid')
        plt.plot(x,lognormalPDF(x, mu=1.2, sigma=1.1), label='targetDist (Buzsaki & Mizuseki, 2014)')
        plt.plot(x, lognormalPDF(x, mu=obtained[0][0], sigma=obtained[0][1], shift=obtained[0][2]), ls='dashed', label='simulatedDist')
        plt.legend()

        plt.subplot(212)
        plt.title('I')
        plt.hist(recordings['corIL1;rateDist'], 100, density=True, align='mid')
        plt.plot(x, lognormalPDF(x, mu=1.2, sigma=1.1), label='targetDist (Buzsaki & Mizuseki, 2014)')
        plt.plot(x, lognormalPDF(x, mu=obtained[1][0], sigma=obtained[1][1], shift=obtained[1][2]), ls='dashed', label='simulatedDist')
        plt.legend()
        plt.savefig('../results/optimize_rates_distributions_'+str(simID)+'.svg')

    m_list[0]=loss
    m_list[1]=obtained

def run_simulator(fitparams):
    """
        runs the function simulator with the multiprocessing manager (if function is called sequentially, this stores memory, otherwise same as calling sumulator directly)
        
        fitparams: list, for description see function simulator
        return: returns dictionary needed for optimization with hyperopt
    """
    manager = multiprocessing.Manager()
    m_list = manager.dict()


    proc = Process(target=simulator,args=(fitparams,m_list))

    proc.start()

    proc.join()

    loss=m_list[0]
    obtained=m_list[1]
    
    return {
        'loss': loss,
        'status': STATUS_OK,
        'obtained': obtained
        }
        
### TODO implement run_simulator_parallel function, which averages the loss over mnultiple runs (with different seeds), to mean over the random implementation (input curretn, which synapses are kept)
        
def testFit(fitparamsDict):
    """
        fitparamsDict: dictionary with parameters, format = as hyperopt returns fit results
        
        Thus, this function can be used to run the simulator function directly with fitted parameters obtained with hyperopt
        
        Returns the loss computed in simulator function.
    """
    return run_simulator([fitparamsDict['shift'],fitparamsDict['mean'],fitparamsDict['sigma'],fitparamsDict['number synapses']])['loss']


if mode=='optimize':
    ### RUN OPTIMIZATION
    best = fmin(
        fn=run_simulator,
        space=[
            hp.uniform('shift', 40, 150),
            hp.uniform('mean', 0, 3),
            hp.uniform('sigma', 0.1, 2),
            scope.int(hp.quniform('number synapses', 5, 49, q=1))
        ],
        algo=tpe.suggest,
        max_evals=1000)
    best['loss'] = testFit(best)
    
    ### SAVE OPTIMIZED PARAMS AND LOSS
    np.save('../dataRaw/optimize_rates_obtainedParams'+str(simID)+'.npy',best)
    
if mode=='test':
    ### LOAD FITTED PARAMETERS
    fit=np.load('../dataRaw/optimize_rates_obtainedParams'+str(simID)+'.npy', allow_pickle=True).item()
    """results=0
    for i in range(10):
        result+=testFit(fit)['loss']#TODO would run 10 times exactly the same, should use different rng seeds"""
    ### PRINT LOSS
    result=testFit(fit)
    print(simID, fit, result)
