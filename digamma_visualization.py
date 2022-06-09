from scipy.special import digamma
import matplotlib.pyplot as plt
import numpy as np

y	= np.linspace(-100, 100, 101)
z	= 1/2+ 1j*y/(2*np.pi)
print(digamma(z) )
plt.plot(y, digamma(z).real )
#plt.yscale('log')
plt.show()
