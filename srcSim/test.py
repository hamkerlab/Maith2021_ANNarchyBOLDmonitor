import numpy as np
from scipy import stats
import pylab as plt
from extras import lognormalPDF

rng = np.random.default_rng()

test=-2+rng.lognormal(0,0.1,20000)

shape,loc,scale = stats.lognorm.fit(test)
sigma=shape
mu=np.log(scale)

x=np.linspace(0,100,1000)
test2=lognormalPDF(x, mu=mu, sigma=sigma, shift=loc)

plt.figure()
plt.hist(test, 100, density=True, align='mid')
plt.plot(x,test2)
plt.xlim(0,5)
plt.savefig('test.png')

print(mu,sigma,loc)
