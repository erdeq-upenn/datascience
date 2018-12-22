import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')

fig = plt.figure()
plt.subplot(2,1,1)
x = np.linspace(0, 10, 100)
dy = np.random.random(100)*0.2
#plt.plot(x,np.sin(x))
plt.plot(x, np.sin(x - 0), color='blue',linestyle =':',label = 'origin')        # specify color by name
plt.plot(x, np.sin(x - 1)+dy,'o', color='g')           # short color code (rgbcmyk)
plt.plot(x, np.sin(x - 2), color='0.75')        # Grayscale between 0 and 1
plt.plot(x, np.sin(x - 3), color='#FFDD44')     # Hex code (RRGGBB from 00 to FF)
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3)) # RGB tuple, values 0 to 1
plt.plot(x, np.sin(x - 5), color='chartreuse'); # all HTML color names supported
plt.plot(x, np.cos(x), '-p', color='gray',markersize=15, linewidth=4,markerfacecolor='white', markeredgecolor='gray', markeredgewidth=2)

plt.axis([-1, 11, -1.5, 1.5]);
plt.legend()
plt.subplot(2,1,2)
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000*rng.rand(100)
plt.scatter(x,y, c = colors, s = sizes, alpha = 0.3, cmap = 'viridis')
plt.colorbar()
plt.axis('equal')

#from sklearn.datasets import load_iris
#iris = load_iris()
#features = iris.data.T

#fig2 = plt.scatter(features[0], features[1], alpha=0.2,				            s=100*features[3], c=iris.target, cmap='viridis')
#plt.xlabel(iris.feature_names[0])
#plt.ylabel(iris.feature_names[1]);

plt.show()
