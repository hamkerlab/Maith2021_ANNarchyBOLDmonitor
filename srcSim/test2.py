import numpy as np
import pylab as plt

def normalizeSignal(signal):
    mode='new'
    

    mean_signal = np.mean(signal[:100000])
    std_signal = np.std(signal[:100000])
    print(mean_signal,std_signal)

    ### signal during baseline period
    signal[:100000] = 0
    
    ### signal after baseline period
    if mode=='classic':
        if mean_signal>0:
            signal[100000:] = np.tanh( (signal[100000:] - mean_signal)/(mean_signal) )
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
    if mode=='new':
        signal[100000:] = (signal[100000:] - mean_signal)/(mean_signal + std_signal + 0.00000001)
    #signal[100000:]-=np.mean(signal[100000:])
    return signal

def normalizeSignal2(x, base=[]):
    if len(base)==0:
        base = x
    m = np.mean(base)
    s = np.std(base)
    
    dif = np.sqrt((base.min()-base.max())**2)
    
    ret = (x - m) / (m + s + 1e-8)
    return ret
    



recordingsB = np.load('../dataRaw/simulations_initialTestofBOLD_recordingsB.npy', allow_pickle=True).item()

mod = np.ones(300000)
#mod[150000:]=2
sigA = (recordingsB['10;r'][:,0]+0.1)*mod
sigB = recordingsB['7;I_CMRO2'][:,0]*mod
sigC = (recordingsB['10;r'][:,0]+10)*mod

for sig in [sigA, sigB, sigC]:
    print(np.mean(sig), np.std(sig), np.mean(sig)/np.std(sig))
    print('m',np.mean(normalizeSignal2(sig, base=sig)))
    print('m3',np.mean(normalizeSignal2(sig*3, base=sig)))
    print('std',np.std(normalizeSignal2(sig, base=sig)))
    print('\n')
quit()

sigA2 = normalizeSignal(sigA.copy())
sigB2 = normalizeSignal(sigB.copy())
sigC2 = normalizeSignal(sigC.copy())

print(sigA[:150000].mean()/sigA[150000:].mean())
print(sigB[:150000].mean()/sigB[150000:].mean())
print(sigC[:150000].mean()/sigC[150000:].mean())
print(sigA2[150000:].mean())
print(sigB2[150000:].mean())
print(sigC2[150000:].mean())

plt.figure()
plt.subplot(321)
plt.plot(sigA)
plt.subplot(322)
plt.plot(sigA2)
plt.subplot(323)
plt.plot(sigB)
plt.subplot(324)
plt.plot(sigB2)
plt.subplot(325)
plt.plot(sigC)
plt.subplot(326)
plt.plot(sigC2)
plt.savefig('test.svg')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
