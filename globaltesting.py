import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy import special
from nmpfit import mpfit
from numpy.random import randn
import pymodelfit as pym

num_points = 300
num_traces = 20

#Function 0:T1 1:A1 2:w 3:mu 4:y0 Convoluted function

fitfunc = lambda p, x: p[1]*np.exp(-x/p[0])

def fitfunc2(p, fjac=None, x=None, y=None, err=None):
    # Parameter values are passed in "p"
    # If fjac==None then partial derivatives should not be
    # computed.  It will always be None if MPFIT is called with default
    # flag.
    model = p[1]*np.exp(-x/p[0])
    # Non-negative status value means MPFIT should continue, negative means
    # stop the calculation.
    status = 0
    return([status, (y-model)])

#Generate some data

Tx = np.linspace(1,100.,num_points)
Ty = np.zeros((num_points,num_traces))
for i in range(num_traces):
    p_1 = [19.85494,(i*0.001)+0.2]
    Ty[:,i] = (fitfunc(p_1,Tx)+(0.001*randn(num_points))).transpose()

plt.figure()
plt.plot(Tx,Ty)
plt.title("Simulated data time constant 20ps")
plt.xlabel("time (ps)")
plt.ylabel("abs")
plt.show()



#setting parameters

parinfo = [{'value':0., 'fixed':0, 'limited':[0,0], 'limits':[0.,0.]},
           {'value':0., 'fixed':0, 'limited':[0,0], 'limits':[0.,0.]}]

parinfo[0]['fixed'] = 1

print parinfo

#function to fit the traces
def fittraces(initialglo):
    print initialglo
    guess = [initialglo,0.2]
    for i in range(len(parinfo)):
        parinfo[i]['value']=guess[i]
    return avgchi2(Tx,Ty,guess,parinfo)

def avgchi2(x,y,guess,parinfo):
    chi2 = 0
    err = num_points
    for i in range(len(Ty[0,:])):
        y = Ty[:,i]
        fa = {'x':x, 'y':y, 'err':err}
        #m = mpfit('fitfunc', guess, parinfo, functkw=fa)
        m = mpfit(fitfunc2, guess, parinfo=parinfo, functkw=fa)
        chi2 = (chi2+m.fnorm)/2
    return np.log(chi2)

initialglo = 19

xopt = opt.fmin(fittraces, initialglo)

print xopt
