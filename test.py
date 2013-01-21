# coding: utf-8

# A exponential fit
import numpy, pymc
import numpy as np
# create some test data
x = numpy.linspace(0,100,100)
f = 0.3 * np.exp(-x/20) + 0.001
numpy.random.seed(76523654)
noise = numpy.random.normal(size=100) * .001     # create some Gaussian noise
f = f + noise                                # add noise to the data

import matplotlib.pyplot as plt

plt.figure()
plt.plot(x,f)
plt.show()

z = numpy.polyfit(x, f, 2)   # the traditional chi-square fit
print 'The chi-square result: ',  z

#priors
sig = pymc.Uniform('sig', 0.0, 100.0, value=1.)

a = pymc.Uniform('a', 0.0, 0.5, value= 0.0)
b = pymc.Uniform('b', 0.0, 100.0, value= 0.0)
c = pymc.Uniform('c', -0.01, 0.01, value= 0.0)

#model
@pymc.deterministic(plot=False)
def mod_expon(x=x, a=a, b=b, c=c):
      return a*np.exp(-x/b) + c

#likelihood
y = pymc.Normal('y', mu=mod_expon, tau=1.0/sig**2, value=f, observed=True)

