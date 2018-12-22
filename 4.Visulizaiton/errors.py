import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')

x = np.linspace(0,10,50)
dy = 0.8
y = np.sin(x)+dy*np.random.randn(50)
#plt.errorbar(x,y,yerr=dy,fmt='.g')
plt.errorbar(x,y,yerr=dy,fmt='o',color='blue',ecolor = 'red',elinewidth = 3, \
             capsize =0)
plt.show()


