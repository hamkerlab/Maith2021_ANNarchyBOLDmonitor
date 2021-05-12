import numpy as np
import pylab as plt
rng = np.random.default_rng()

def targetDist(x, mu=-0.702, sigma=0.9355):
    return np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))

x=np.logspace(-3,2,100)
dist=rng.lognormal(mean=1.2, sigma=1.1, size=1000)
plt.figure()
plt.plot(x,targetDist(x, mu=1.2, sigma=1.1))
plt.hist(dist, 100, density=True, align='mid')
plt.hist(2*dist, 100, density=True, align='mid')
#plt.xscale('log')
plt.savefig('test.svg')
