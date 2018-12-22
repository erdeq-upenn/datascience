# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 15:43:41 2018

@author: Miao
"""
#%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np


def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
###########################################
#plt.contour(X, Y, Z, colors='black')
#plt.contourf(X, Y, Z, 50, cmap='RdGy');
#plt.colorbar();
#plt.axis('equal')
###################################
#plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',# extent is [xmin, xmax,ymin,ymax]
#           cmap='RdGy')
#plt.colorbar()
#plt.axis(aspect='image');
##############################

contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=10)

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
           cmap='RdGy', alpha=0.6)
plt.colorbar();
plt.show()
