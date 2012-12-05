import numpy as np
import matplotlib.pylab as plt
import numpy.random as ran

#Setting the random generator up
ran.seed(6041510)

#Functions
def decay(t,tau,A):
    return A*np.exp(-(t)/tau)

def gaussdis(d,f,sig):
    return np.sum(-(d-f)**2/sig**2)

def randchange(var,d):
    return (var+ran.uniform(-d,d))

#Generate test data
size = 1000
tdata = np.linspace(0, 100, size)
tau = 20
A = 0.03
Adata = decay(tdata,tau,A)+ran.uniform(-0.001,0.001,size)

#Monte Carlo scheme

simsize = 500000
burnin = 9000
tau_d = 0.5
A_d = 0.01
tau = 30
A = 0.040
sig = 0.002

Res = np.ones((simsize,3))

for i in range(simsize):
    dist1 = gaussdis(Adata,decay(tdata,tau,A),sig)
    tau1 = randchange(tau,tau_d)
    A1 = randchange(A,A_d)
    dist2 = gaussdis(Adata,decay(tdata,tau1,A1),sig)
    logB = dist2-dist1
    
    #Sampling method
    #non negativity
    if tau1<=0 or A1<=0:
        tau = tau
        A = A
    else:
        # B postive good change in dist
        if logB>=0:
            tau = tau1
            A = A1
        else:
            # if closer to one then more likely to be a good step
            if ran.rand()<=np.exp(logB):
                tau = tau1
                A = A1

    #print to results matrix
    Res[i,0] = logB
    Res[i,1] = tau
    Res[i,2] = A
    

#Plotting of results

step = np.linspace(0, simsize, simsize)
plt.plot(tdata,Adata)
plt.title("sample data, 0.001 rms noise, tau=20, A=0.03")

plt.figure()
plt.subplot(222)
plt.hist(Res[burnin:,2],50)
plt.title("A")

plt.subplot(221)
plt.hist(Res[burnin:,1],50)
plt.title("tau")

plt.subplot(223)
plt.plot(step,Res[:,1])
plt.title("tau")

plt.subplot(224)
plt.plot(step,Res[:,2])
plt.title("A")

plt.show()

    
    