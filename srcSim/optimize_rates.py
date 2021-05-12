from ANNarchy import *
import pylab as plt
from scipy import stats
import sys
from model import params, rng
from extras import getFiringRateDist
from extras import lognormalPDF

# hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.pyll import scope

# multiprocessing
from multiprocessing import Process
import multiprocessing


### standard model compilation
simID=int(sys.argv[1])
compile('annarchy'+str(simID))


def set_number_of_synapses(weights, number):
    """
        weights: list (for each post neuron) of lists (list of weights to all pre neurons)
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
def simulator(fitparams, m_list):
    plotting=False

    ### get params
    init_offsetVal = fitparams[0]+rng.lognormal(mean=fitparams[1], sigma=fitparams[2], size=params['corE_popsize'])
    number_of_synapses = int(fitparams[3])

    ### reset model to compilation state
    reset()

    ### create monitors
    monDict={'pop;inputPop':['r'],
             'pop;corEL1':['v', 'spike'],
             'pop;corIL1':['v', 'spike']}
    mon={}
    for key, val in monDict.items():
        compartmentType, compartment = key.split(';')
        if compartmentType=='pop':
            mon[compartment] = Monitor(get_population(compartment),val)

    ### Set input strenght
    get_population('inputPop').offsetVal=init_offsetVal

    ### Set number of synapses in projections = set all weights except number of synapses to zero (non-zero weights=random)
    get_projection('corEL1_corIL1').w=set_number_of_synapses(get_projection('corEL1_corIL1').w, number_of_synapses)
    get_projection('corIL1_corEL1').w=set_number_of_synapses(get_projection('corIL1_corEL1').w, number_of_synapses)
    get_projection('corIL1_corIL1').w=set_number_of_synapses(get_projection('corIL1_corIL1').w, number_of_synapses)

    ### simulate
    simulate(20000)
    ### get monitors
    recordings={}
    for key, val in monDict.items():
        compartmentType, compartment = key.split(';')
        for val_val in val:
            recordings[compartment+';'+val_val] = mon[compartment].get(val_val)

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
        plt.figure(figsize=(18,12))
        plt.subplot(221)
        for idx in range(recordings['inputPop;r'].shape[1]):
            plt.plot(recordings['inputPop;r'][:,idx])

        plt.subplot(223)
        t,n = raster_plot(recordings['corEL1;spike'])
        plt.plot(t,n,'k.',markersize=0.2)

        plt.subplot(224)
        t,n = raster_plot(recordings['corIL1;spike'])
        plt.plot(t,n,'k.',markersize=0.2)
        plt.savefig('test1.png')


        ### plot2
        for pop in ['corEL1', 'corIL1']:
            recordings[pop+';rateDist']=getFiringRateDist(recordings[pop+';spike'], 20000)

        plt.figure()
        x=np.linspace(0,100,1000)

        plt.subplot(211)
        plt.title('E')
        plt.hist(recordings['corEL1;rateDist'], 100, density=True, align='mid')
        plt.plot(x,lognormalPDF(x, mu=1.2, sigma=1.1))
        plt.plot(x, lognormalPDF(x, mu=obtained[0][0], sigma=obtained[0][1], shift=obtained[0][2]), ls='dashed')
        #plt.xlim(10**-3,30)

        plt.subplot(212)
        plt.title('I')
        plt.hist(recordings['corIL1;rateDist'], 100, density=True, align='mid')
        plt.plot(x, lognormalPDF(x, mu=1.2, sigma=1.1))
        plt.plot(x, lognormalPDF(x, mu=obtained[1][0], sigma=obtained[1][1], shift=obtained[1][2]), ls='dashed')
        #plt.xlim(10**-3,30)
        plt.savefig('test2.svg')

    m_list[0]=loss
    m_list[1]=obtained

def run_simulator(fitparams):
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


"""result=run_simulator([-80,1.2,1.1,3000])
print(result['obtained'], result['loss'])"""
#scope.int(hp.quniform('number synapses', 5, 50, q=1))
#hp.uniform('number synapses', 5, 50),        

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

print(best)
np.save('../dataRaw/optimize_rates_obtainedParams'+str(simID)+'.npy',best)
