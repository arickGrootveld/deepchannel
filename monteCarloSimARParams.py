from mismatch_data_gen import ARCoeffecientGeneration
import numpy as np

firstCoeff = 0.5
secondCoeff = 0.4

arMeans = (firstCoeff, secondCoeff)

arStd = 0.2

seed=1

numSimulations = 10000


firstCoefficients = np.empty(numSimulations)
secondCoefficients = np.empty(numSimulations)

# Begin Monte Carlo Simulation
for i in range(0,numSimulations):
    F = ARCoeffecientGeneration(arMeans, arStd)
    firstCoefficients[i] = F[0, 0]
    secondCoefficients[i] = F[0, 1]

print('mean of first set of coefficients is: ', np.mean(firstCoefficients))
print('mean of second set of coefficients is: ', np.mean(secondCoefficients))
print('standard deviation of first set of coefficients is: ', np.std(firstCoefficients))
print('standard deviation of second set of coefficients is: ', np.std(secondCoefficients))

# calculating how much our actual AR parameters deviate from what we expected them to be
varFromExMeanFir = np.sqrt(np.sum(np.square(firstCoefficients - np.full((numSimulations), firstCoeff))) / numSimulations)

varFromExMeanSec = np.sqrt(np.sum(np.square(secondCoefficients - np.full((numSimulations), secondCoeff))) / numSimulations)


print('variance of our first coefficients actual values from our expected mean is: ', 
        varFromExMeanFir)

print('variance of our second coefficients actual values from our expect mean is: ', 
        varFromExMeanSec)


