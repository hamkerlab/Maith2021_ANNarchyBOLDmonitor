from ANNarchy import *
import pylab as plt
from scipy import stats
import sys
from model import params, rng, add_scaled_projections
from extras import getFiringRateDist, lognormalPDF, plot_input_and_raster, addMonitors, startMonitors, getMonitors, generateInputs

"""
    this weight scaling optimization needs:
        params['optimizeRates'] = 'v2'
        params['increasingInputs'] = False
"""
if (params['optimizeRates']=='v2' and params['increasingInputs']==False) == False:
    print('other parameters needed for optimize rates v2')
    print("params['optimizeRates'] =",params['optimizeRates'],"--> v1 ?")
    print("params['increasingInputs'] =",params['increasingInputs'],"--> False ?")
    quit()

# hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.pyll import scope

# multiprocessing
from multiprocessing import Process
import multiprocessing

def generateWeights(name, rng, scale=1.0):
    """
        function generates new scaled weights for one projection of model v2
    """
    weights = np.array(get_projection(name).w)
    newWeights = params['weightDist'](rng,scale).get_values(int(weights.shape[0]*weights.shape[1]))
    newWeights = np.reshape(newWeights, weights.shape).tolist()
    
    return newWeights
    

def scaleProjections(S_INP, S_EI, S_IE, S_II, rng):
    """
        generates new scaled weights for the projections of model v2 after compilation
    """
    get_projection('inputPop_corEL1').w = generateWeights('inputPop_corEL1', rng, scale=S_INP)
    get_projection('inputPop_corIL1').w = generateWeights('inputPop_corIL1', rng, scale=S_INP)
    get_projection('corEL1_corIL1').w   = generateWeights('corEL1_corIL1', rng, scale=S_EI)
    get_projection('corIL1_corEL1').w   = generateWeights('corIL1_corEL1', rng, scale=S_IE)
    get_projection('corIL1_corIL1').w   = generateWeights('corIL1_corIL1', rng, scale=S_II)



### add the projections, scaling factors are adjusted during fitting below
add_scaled_projections(params['fittedParams']['S_INP'], params['fittedParams']['S_EI'], params['fittedParams']['S_IE'], params['fittedParams']['S_II'], rng)



### standard model compilation
simID=int(sys.argv[1])
mode=str(sys.argv[2])
compile('annarchy'+str(simID))



### DEFINE SIMULATOR
def simulator(fitparams, rng, m_list=[0,0]):
    """
        fitparams: list
            fitparams[0]: S_INP for weight scaling
            fitparams[1]: S_EI for weight scaling
            fitparams[2]: S_IE for weight scaling
            fitparams[3]: S_II for weight scaling
            
        m_list: variable to store results, from multiprocessing
    """
    ### should plots be generated?
    plotting=False

    ### get params
    S_INP, S_EI, S_IE, S_II = fitparams

    ### reset model to compilation state
    reset()

    ### create monitors
    monDict={'pop;inputPop':['r'],
             'pop;corEL1':['v', 'spike'],
             'pop;corIL1':['v', 'spike']}
    mon={}
    mon = addMonitors(monDict,mon)
    

    ### Set input strenght, if function is called multiple times --> different inputs (but same distribution)
    get_population('inputPop').offsetVal = generateInputs(0,1.2,1.1,params['corE_popsize'],rng)['values']

    ### Set scaled weights of the projections
    scaleProjections(S_INP, S_EI, S_IE, S_II, rng)

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
        runs the function simulator with the multiprocessing manager (if function is called sequentially, this stores memory, otherwise same as calling simulator directly)
        
        fitparams: list, for description see function simulator
        return: returns dictionary needed for optimization with hyperopt
    """
    manager = multiprocessing.Manager()
    m_list = manager.dict()

    num_rep = 10
    loss = np.zeros(num_rep)
    for i in range(num_rep):
        rng = np.random.default_rng()
        proc = Process(target=simulator,args=(fitparams,rng,m_list))
        proc.start()
        proc.join()

        loss[i]=m_list[0]
        
    loss = np.mean(loss)
    obtained=m_list[1]
    
    return {
        'loss': loss,
        'status': STATUS_OK,
        'obtained': obtained
        }
        
       
def testFit(fitparamsDict):
    """
        fitparamsDict: dictionary with parameters, format = as hyperopt returns fit results
        
        Thus, this function can be used to run the simulator function directly with fitted parameters obtained with hyperopt
        
        Returns the loss computed in simulator function.
    """
    return run_simulator([fitparamsDict['I_INP'],fitparamsDict['I_EI'],fitparamsDict['I_IE'],fitparamsDict['I_II']])['loss']


if mode=='optimize':
    ### RUN OPTIMIZATION
    best = fmin(
        fn=run_simulator,
        space=[
            hp.uniform('I_INP', 0.5, 20),
            hp.uniform('I_EI', 0.5, 20),
            hp.uniform('I_IE', 0.5, 20),
            hp.uniform('I_II', 0.5, 20)
        ],
        algo=tpe.suggest,
        max_evals=1000)
    best['loss'] = testFit(best)
    
    ### SAVE OPTIMIZED PARAMS AND LOSS
    np.save('../dataRaw/optimize_ratesv2_obtainedParams'+str(simID)+'.npy',best)
    
if mode=='test':
    ### LOAD FITTED PARAMETERS
    fit=np.load('../dataRaw/optimize_ratesv2_obtainedParams'+str(simID)+'.npy', allow_pickle=True).item()
    """results=0
    for i in range(10):
        result+=testFit(fit)['loss']#TODO would run 10 times exactly the same, should use different rng seeds"""
    #fit = {'I_INP':1, 'I_EI':1, 'I_IE':1, 'I_II':1,}
    ### PRINT LOSS
    result=testFit(fit)
    print(simID, fit, result)
