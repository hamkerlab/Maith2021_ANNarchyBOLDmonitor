from ANNarchy import *
import pylab as plt
from scipy import signal, stats
from extras import lognormalPDF

rng = np.random.default_rng()
setup(dt=0.1)


### create both neuron types

Izhikevich2007RS = Neuron(
    parameters="""
        C        = 100.   : population
        k        =   0.70 : population
        v_r      = -60.   : population
        v_t      = -40.   : population
        a        =   0.03 : population
        b        =  -2.   : population
        c        = -50.   : population
        d        = 100.   : population
        v_peak   =  35.   : population
        tau_ampa =  10.   : population
        tau_gaba =  10.   : population
        E_ampa   =   0.   : population
        E_gaba   = -90.   : population
        I_ext    =   0.
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa : init = 0
        dg_gaba/dt = -g_gaba/tau_gaba : init = 0
        I          = I_ext - g_ampa*(v - E_ampa) - g_gaba*(v - E_gaba)
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I : init = -60
        du/dt      = a*(b*(v - v_r) - u) : init = 0
    """,
    spike = "v >= v_peak",
    reset = """
        v = c
        u = u + d
    """,
    name = "Izhikevich2007RS",
    description = "RS cortical neuron model from Izhikevich (2007) with additional conductance based synapses."
)

Izhikevich2007FS = Neuron(
    parameters="""
        C        =  20.    : population
        k        =   1.    : population
        v_r      = -55.    : population
        v_t      = -40.    : population
        v_b      = -55.    : population
        a        =   0.2   : population
        b        =   0.025 : population
        c        = -45.    : population
        d        =   0.    : population
        v_peak   =  25.    : population
        tau_ampa =  10.    : population
        tau_gaba =  10.    : population
        E_ampa   =   0.    : population
        E_gaba   = -90.    : population
        I_ext    =   0.
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa : init = 0
        dg_gaba/dt = -g_gaba/tau_gaba : init = 0
        I          = I_ext - g_ampa*(v - E_ampa) - g_gaba*(v - E_gaba)
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I : init = -55
        U_v        = if v<v_b: 0 else: b*(v - v_b)**3
        du/dt      = a*(U_v - u) : init = 0
    """,
    spike = "v >= v_peak",
    reset = """
        v = c
        u = u + d
    """,
    name = "Izhikevich2007FS",
    description = "FS cortical interneuron model from Izhikevich (2007) with additional conductance based synapses."
)

### create 1000 neurons of each neurontype
corE = Population(1000, neuron=Izhikevich2007RS, name='CorE')
corI = Population(1000, neuron=Izhikevich2007FS, name='CorI')


### create spiketimearray input with one spike
spike_times = [
  [10] for i in range(1)
]
inputPop = SpikeSourceArray(spike_times=spike_times)


### direct this spike to both populations
inputPopcorE = Projection( 
    pre    = inputPop,
    post   = corE, 
    target = 'ampa',
    name = 'inputPopcorE'
).connect_all_to_all(weights = LogNormal(mu=-1.5, sigma=0.93))#weights scaled to obtain PSP dist

inputPopcorI = Projection( 
    pre    = inputPop,
    post   = corI, 
    target = 'ampa',
    name = 'inputPopcorI'
).connect_all_to_all(weights = LogNormal(mu=-1.5, sigma=0.93))#weights scaled to obtain PSP dist

"""#how new mu, sigma were obtained:
#inputPopcorE --> weights * 0.5
test=0.3*rng.lognormal(-0.7,0.93,10000)
shape,loc,scale = stats.lognorm.fit(test)
sigma=shape
mu=np.log(scale)
print(mu,sigma)
quit()
#inputPopcorE --> weights * 0.3-0.4
... both about mu=1.5"""



### compile
compile()



### compare weight distribution with target EPSP distribution

plt.figure()
x=np.arange(0.01,10,0.01)
plt.hist(np.array(inputPopcorE.w).flatten(), 100, density=True, align='mid')
plt.plot(x,lognormalPDF(x))
plt.xlim(0,10)
plt.savefig('../results/get_weightDist_weightsE_dist.svg')


### record membranepotential of both neurons
mon_v_corE = Monitor(corE, 'v')
mon_v_corI = Monitor(corI, 'v')


### simulate 20 ms
simulate(100)


### plot membrane potentials
v_corE = mon_v_corE.get('v')
v_corI = mon_v_corI.get('v')


### get peak values from membrane potentials
maxPosE=np.zeros(1000)*np.nan
maxPosI=np.zeros(1000)*np.nan
maxValE=np.zeros(1000)*np.nan
maxValI=np.zeros(1000)*np.nan
for idx in range(1000):
    if len(signal.argrelmax(v_corE[:,idx])[0])>0:
        maxPosE[idx]=signal.argrelmax(v_corE[:,idx])[0][0]
        maxValE[idx]=v_corE[int(maxPosE[idx]),idx]
    if len(signal.argrelmax(v_corI[:,idx])[0])>0:
        maxPosI[idx]=signal.argrelmax(v_corI[:,idx])[0][0]
        maxValI[idx]=v_corI[int(maxPosI[idx]),idx]



plt.figure()
plt.subplot(211)
plt.plot(v_corE[:,0],color='C0')
plt.plot(v_corE[:,1],color='C1')
plt.plot(v_corE[:,999],color='C2')
plt.axvline(maxPosE[0],color='C0')
plt.axvline(maxPosE[1],color='C1')
plt.axvline(maxPosE[999],color='C2')
plt.axhline(maxValE[0],color='C0')
plt.axhline(maxValE[1],color='C1')
plt.axhline(maxValE[999],color='C2')
plt.subplot(212)
plt.plot(v_corI[:,0],color='C0')
plt.plot(v_corI[:,1],color='C1')
plt.plot(v_corI[:,999],color='C2')
plt.axvline(maxPosI[0],color='C0')
plt.axvline(maxPosI[1],color='C1')
plt.axvline(maxPosI[999],color='C2')
plt.axhline(maxValI[0],color='C0')
plt.axhline(maxValI[1],color='C1')
plt.axhline(maxValI[999],color='C2')
plt.savefig('../results/get_weightDist_PSPexamples.svg')



### calculate PSP distribution from membrane potentials

## difference between resting potential and peaks below 0
PSPsE=maxValE[maxValE<0]+60
PSPsI=maxValI[maxValE<0]+55

plt.figure()
plt.subplot(211)
plt.hist(PSPsE[PSPsE<=10], 100, density=True, align='mid')
plt.plot(x,lognormalPDF(x))
plt.xlim(0,10)
plt.subplot(212)
plt.hist(PSPsI[PSPsI<=10], 100, density=True, align='mid')
plt.plot(x,lognormalPDF(x))
plt.xlim(0,10)
plt.savefig('../results/get_weightDist_PSP_dist.svg')

