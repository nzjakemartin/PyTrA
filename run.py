#--RUN--
import pymc, test
R = pymc.MCMC(test)
R.sample(100000)

from pylab import *
from pymc import *

import numpy as np

a = R.a.value
b = R.b.value
c = R.c.value

Matplot.plot(R)
x = numpy.linspace(0,100,100)
z = a*np.exp(-x/b) + c
f = 0.3 * np.exp(-x/20) + 0.001
print f,z
numpy.random.seed(76523654)
noise = numpy.random.normal(size=100) * .001     # create some Gaussian noise
f = f + noise

import matplotlib.pyplot as plt
plt.figure()
plt.plot(x,f,'o')
plt.plot(x,z)

plt.show()
