import numpy as np
import pylab as plt


### Load I_CBF and I_CMRO2 for corE and corI without normalization
recordingsB = np.load('../dataRaw/simulations_initialTestofBOLD_recordingsB.npy', allow_pickle=True).item()
I_CBF_corE = recordingsB['10;r'][:,0]*(np.arange(1,300001,1)/300000+1)
I_CBF_corI = recordingsB['11;r'][:,0]*(np.arange(1,300001,1)/300000+1)
I_CMRO2_corE = recordingsB['6;I_CMRO2']
I_CMRO2_corI = recordingsB['7;I_CMRO2']

### divide I_CMRO2 by (k_CMRO2 / k_CBF), thus get the signals which came from the ACC projections
k_CBF = 1/2.46
k_CMRO2 = 2*k_CBF
I_CMRO2_corE /= k_CMRO2 / k_CBF
I_CMRO2_corI /= k_CMRO2 / k_CBF

plt.figure()
plt.subplot(211)
plt.plot(I_CBF_corE, label='I_CBF')
plt.plot(I_CMRO2_corE, label='I_CMRO2')
plt.subplot(212)
plt.plot(I_CBF_corI, label='I_CBF')
plt.plot(I_CMRO2_corI, label='I_CMRO2')
plt.savefig('test.svg')

def normalizeSignal(signal):
    mode='z_score + tanh'
    

    mean_signal = np.mean(signal[:100000])
    std_signal = np.std(signal[:100000])

    ### signal during baseline period
    signal[:100000] = 0
    
    ### signal after baseline period
    if mode=='tanh':
        signal[100000:] = np.tanh( np.log10( (signal[100000:] + 0.00000001)/(mean_signal + 0.00000001) ) )
    if mode=='z_score':
        if std_signal>0:
            signal[100000:] = (signal[100000:] - mean_signal)/(std_signal*10)
        else:
            signal[100000:] = (signal[100000:] - mean_signal)/10
    if mode=='z_score + tanh':
        if std_signal>0:
            signal[100000:] = np.tanh( (signal[100000:] - mean_signal)/(std_signal*10) )
        else:
            signal[100000:] = np.tanh( (signal[100000:] - mean_signal)/10. )
            
    return signal
    
### calculate normalized I_CBF and I_CMRO2 for corE and corI with first 100000 values (first 10000 ms)
I_CBF_corE   = normalizeSignal(I_CBF_corE)
I_CBF_corI   = normalizeSignal(I_CBF_corI)
I_CMRO2_corE = normalizeSignal(I_CMRO2_corE)
I_CMRO2_corI = normalizeSignal(I_CMRO2_corI)



### combine I_CBF from corE and corI and combine I_CMRO2 from corE and corI
I_CBF_combined = (4/5)*I_CBF_corE + (1/5)*I_CBF_corI
I_CMRO2_combined = (4/5)*I_CMRO2_corE + (1/5)*I_CMRO2_corI


### generate for loop which calculates ODEs of BOLD neuron

if 0:
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
        I_CBF           = sum(I_f)                                                     : init=0
        I_CMRO2         = sum(I_r) * (k_CMRO2 / k_CBF)                                 : init=0
        1000*dsCBF/dt   = ea * I_CBF - c_CBF * sCBF - k_CBF * (CBF - 1)                : init=0
        1000*dCBF/dt    = sCBF                                                         : init=1, max=2, min=0
        1000*dsCMRO2/dt = ea * I_CMRO2 - c_CMRO2 * sCMRO2 - k_CMRO2 * (CMRO2 - 1)      : init=0
        1000*dCMRO2/dt  = sCMRO2                                                       : init=1, max=2, min=0

        1000*dq/dt      = 1 / tau_0 * (CMRO2 - (q / v) * f_out)                        : init=1
        1000*dv/dt      = 1 / tau_0 * (CBF - f_out)                                    : init=1
        f_out           = v**(1 / alpha)                                               : init=1

        k_1             = 4.3 * v_0 * E_0 * TE
        k_2             = epsilon * r_0 * E_0 * TE
        k_3             = 1 - epsilon
        BOLD            = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v))  : init=0
        r=0
    """,
        name = "-",
        description = "-"
    )

dt=0.1
duration=30000

### Define/Initialize parameters, variables
k_CBF   = 1/2.46
k_CMRO2 = 2*k_CBF
c_CBF   = 0.6*np.sqrt(4*k_CBF)
c_CMRO2 = np.sqrt(4*k_CMRO2)
ea      = 1.0
E_0     = 0.34
tau_0   = 0.98
alpha   = 0.33
V_0     = 0.02
v_0     = 40.3
TE      = 40/1000.
epsilon = 1
r_0     = 25
k_1     = 4.3 * v_0 * E_0 * TE
k_2     = epsilon * r_0 * E_0 * TE
k_3     = 1 - epsilon

I_CBF_init   = 0
I_CMRO2_init = 0
sCBF_init    = 0
CBF_init     = 1
sCMRO2_init  = 0
CMRO2_init   = 1
q_init       = 1
v_init       = 1
f_out_init   = 1
BOLD_init    = 0

I_CBF   = np.ones(int(duration/dt))*I_CBF_init
I_CMRO2 = np.ones(int(duration/dt))*I_CMRO2_init
sCBF    = np.ones(int(duration/dt))*sCBF_init
CBF     = np.ones(int(duration/dt))*CBF_init
sCMRO2  = np.ones(int(duration/dt))*sCMRO2_init
CMRO2   = np.ones(int(duration/dt))*CMRO2_init
q       = np.ones(int(duration/dt))*q_init
v       = np.ones(int(duration/dt))*v_init
f_out   = np.ones(int(duration/dt))*f_out_init
BOLD    = np.ones(int(duration/dt))*BOLD_init



### first timestep
t=0
### not DFGs
I_CBF[t]        = I_CBF_combined[t]
I_CMRO2[t]      = I_CMRO2_combined[t] * (k_CMRO2 / k_CBF)
f_out[t]        = v_init**(1 / alpha)

### DFGs
sCBF[t]         = sCBF_init + dt*(ea * I_CBF[t] - c_CBF * sCBF_init - k_CBF * (CBF_init - 1))/1000.
CBF[t]          = CBF_init + dt*sCBF_init/1000.
sCMRO2[t]       = sCMRO2_init + dt*(ea * I_CMRO2[t] - c_CMRO2 * sCMRO2_init - k_CMRO2 * (CMRO2_init - 1))/1000.
CMRO2[t]        = CMRO2_init + dt*sCMRO2_init/1000.

q[t]            = q_init + dt*(1 / tau_0 * (CMRO2_init - (q_init / v_init) * f_out[t]))/1000.
v[t]            = v_init + dt*(1 / tau_0 * (CBF_init - f_out[t]))/1000.

### all other timesteps
for t in range(1,int((duration)/dt)):

    ### not DFGs
    I_CBF[t]        = I_CBF_combined[t]
    I_CMRO2[t]      = I_CMRO2_combined[t] * (k_CMRO2 / k_CBF)
    f_out[t]        = v[t-1]**(1 / alpha)
    
    ### DFGs
    sCBF[t]         = sCBF[t-1] + dt*(ea * I_CBF[t] - c_CBF * sCBF[t-1] - k_CBF * (CBF[t-1] - 1))/1000.
    CBF[t]          = CBF[t-1] + dt*sCBF[t-1]/1000.
    sCMRO2[t]       = sCMRO2[t-1] + dt*(ea * I_CMRO2[t] - c_CMRO2 * sCMRO2[t-1] - k_CMRO2 * (CMRO2[t-1] - 1))/1000.
    CMRO2[t]        = CMRO2[t-1] + dt*sCMRO2[t-1]/1000.

    q[t]            = q[t-1] + dt*(1 / tau_0 * (CMRO2[t-1] - (q[t-1] / v[t-1]) * f_out[t]))/1000.
    v[t]            = v[t-1] + dt*(1 / tau_0 * (CBF[t-1] - f_out[t]))/1000.
BOLD = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v))




### plot BOLD variables

### load params for x axis and times
simParams   = np.load('../dataRaw/simulations_initialTestofBOLD_simParams.npy', allow_pickle=True).item()
times=np.arange(simParams['rampUp']+simParams['dt'],simParams['rampUp']+simParams['simDur']+simParams['dt'],simParams['dt'])


plt.subplot(321)
plt.plot(I_CBF[:])
plt.subplot(323)
plt.plot(I_CMRO2[:])

plt.subplot(322)
plt.title('Self-defined model')
plt.plot(times,BOLD)
plt.xlim(times[0],times[-1])

plt.subplot(324)
plt.title('CBF & CMRO2')
plt.plot(times,CBF,label='CBF')
plt.plot(times,CMRO2,label='CMRO2')
plt.xlim(times[0],times[-1])
plt.legend()

plt.subplot(326)
plt.title('I_CBF & I_CMRO2')
plt.plot(times,I_CBF,label='I_CBF '+str(round(np.mean(I_CBF),2)))
plt.plot(times,I_CMRO2,label='I_CMRO2 '+str(round(np.mean(I_CMRO2),2)))
plt.xlim(times[0],times[-1])
plt.legend()

plt.savefig('test.svg')






































