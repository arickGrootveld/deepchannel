import numpy as np

mean1 = 0.5
mean2 = 0.4

standardDev = 0.2

numSims = 100

coeff1s = np.empty(numSims)
coeff2s = np.empty(numSims)

for i in range(0,numSims):
    coeff1s[i] = standardDev * np.random.randn(1) + mean1
    coeff2s[i] = standardDev * np.random.randn(1) + mean2

print(np.mean(coeff1s))
print(np.std(coeff1s))

print(np.mean(coeff2s))
print(np.std(coeff2s))