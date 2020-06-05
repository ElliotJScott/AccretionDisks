import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as igr

def InitialMassFunction(X):
    if(X == Xin or X == Xout):
        return 0
    else:
        k = -(((X-1)**2)/0.001)
        return np.exp(k)

def Getfval(X, sigma):
    return 4 * (X**3) * sigma

def Getgval(X, sigma):
    return 3 * eta * sigma * X

def CalcAngularMomentum(sigmaVals): #will be normalised so can ignore factor of sqrt(GM)
    densityVals = np.zeros(len(sigmaVals))
    for i in range(len(sigmaVals)):
        densityVals[i] = (rValues[i] ** 1.5) * sigmaVals[i]
    return igr.trapz(densityVals, rValues)

def CalcTotalMass(sigmaVals):
    func = np.zeros(len(sigmaVals))
    for i in range(len(sigmaVals)):
        func[i] = rValues[i]  * sigmaVals[i]
    return igr.trapz(func, rValues)

def GetAngMomDensityPeak(sigmaVals):
    angmomdens = np.zeros(len(sigmaVals))
    for i in range(len(sigmaVals)):
        angmomdens[i] = (rValues[i] ** 0.5) * sigmaVals[i]
    return rValues[np.argmax(angmomdens)]

eta = 1
Xin = 0.02
Xout = 2
deltat = (2/(3 * eta)) * (Xin ** 4)
tmax = 0.52
specificTimeValues = [0.002, 0.008, 0.032, 0.128, 0.512]
dataCounter = 0
numtvals = int(tmax/deltat)
numXvals = 100
plotData = np.zeros((numXvals, len(specificTimeValues)))
deltaX = Xin 
fInitvals = np.zeros(numXvals)
lastfValues = np.zeros(numXvals)
currfValues = np.zeros(numXvals)
gInitvals = np.zeros(numXvals)
lastgValues = np.zeros(numXvals)
currgValues = np.zeros(numXvals)
sigmaInitvals = np.zeros(numXvals)
rValues = np.zeros(numXvals)
for i in range(numXvals):
    X = (i+1)*deltaX
    rValues[i] = X**2
    sigma = InitialMassFunction(X)
    sigmaInitvals[i] = sigma
    fInitvals[i] = Getfval(X, sigma)
    gInitvals[i] = Getgval(X, sigma)

lastfValues = fInitvals
lastgValues = gInitvals
print(str(numtvals / 1000))
nstep = numtvals / 1000
totalAngMomvals = np.zeros(1000)
totalAngMomvals[0] = 1
initTotalAngularMomentum = CalcAngularMomentum(sigmaInitvals)
initTotalMass = CalcTotalMass(sigmaInitvals)
finalTotalMass = 0
densityMaxPositions = np.zeros(1000)
densityMaxPositions[0] = GetAngMomDensityPeak(sigmaInitvals)
tvals = np.zeros(1000)
tvals[0] = 0
fig, ax = plt.subplots()
ax.set(xlabel='r', ylabel='$\sigma$',
           title='A plot of $\sigma$ against r at t = 0')
ax.plot(rValues, sigmaInitvals)
plt.show()
for n in range(1, numtvals):
    #if (n % 1000 == 0):
    #    print(str(0.01 * np.floor(n * 10000 / numtvals)) + "% done")
    sigmaPlots = np.zeros(numXvals)
    for i in range(numXvals):
        X = (i+1)*deltaX
        if(X == Xin or X == Xout):
            currfValues[i] = 0
            currgValues[i] = 0
        else:
            fval = lastfValues[i] + ((deltat / (deltaX**2)) * (lastgValues[i+1] + lastgValues[i-1] - (2 * lastgValues[i])))
            currfValues[i] = fval
            sigma = fval / (4 * (X**3))
            currgValues[i] = Getgval(X, sigma)
            sigmaPlots[i] = sigma
    for i in range(numXvals):
        lastfValues[i] = currfValues[i]
        lastgValues[i] = currgValues[i]

    plt.show()
    if (n % nstep == 0):
        index = int(n / nstep)
        sigmaVals = np.zeros(numXvals)
        for p in range(numXvals):
            X = (p+1)*deltaX
            fval = currfValues[p]
            sigma = fval / (4 * (X**3))
            sigmaVals[p] = sigma
        totalAngMomvals[index] = CalcAngularMomentum(sigmaVals) / initTotalAngularMomentum
        densityMaxPositions[index] = GetAngMomDensityPeak(sigmaVals)
        tvals[index] = n * deltat
    if (dataCounter <= 4):
        if (n * deltat >= specificTimeValues[dataCounter]):
            for q in range(numXvals):
                X = (q+1)*deltaX
                fval = currfValues[q]
                sigma = fval / (4 * (X**3))
                plotData[q, dataCounter] = sigma
            sigmaVals = plotData[:, dataCounter]
            maxSigma = max(sigmaVals)
            print("tau = " + str(specificTimeValues[dataCounter]) + " | max sigma = " + str(maxSigma) + " | r at max sigma = " + str(rValues[np.argmax(sigmaVals)]))
            dataCounter = dataCounter + 1

finalTotalMass = CalcTotalMass(plotData[:, len(specificTimeValues) - 1])
fracMassRemaining = finalTotalMass/initTotalMass
print("Fraction of mass remaining at t=0.512 = " + str(fracMassRemaining))
#code to plot sigma at various values of tau

fig, ax = plt.subplots()
plots = []
ax.set(xlabel='r', ylabel='$\sigma$',
           title='A plot of $\sigma$ against r for various values of t')
for d in range(len(specificTimeValues)):
    p = ax.plot(rValues, plotData[:, d])
    plots = np.append(plots, p)
ax.legend((plots[0], plots[1], plots[2], plots[3], plots[4]), ('t = 0.002', 't = 0.008', 't = 0.032', 't = 0.128', 't = 0.512'), loc='lower right', shadow=False)
plt.show()

fig, ax = plt.subplots(2)
topPlot = ax[0].plot(tvals, totalAngMomvals)
botPlot = ax[1].plot(tvals, densityMaxPositions)
ax[0].set(xlabel='tau', ylabel='Total Normalised Angular Momentum',
           title='A plot of Total Angular Momentum against tau normalised by the angular momentum at tau = 0')
ax[1].set(xlabel='tau', ylabel='r',
           title='A plot of of the position of the maximum of the angular momentum surface density against tau')
plt.show()

