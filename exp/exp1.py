# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 20:23:13 2019

@author: Alex
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()  # for plot styling

rand = np.random.RandomState(42)

mean = [0, 0]
cov = [[1, 2],
       [2, 5]]
X = rand.multivariate_normal(mean, cov, 100);


#%matplotlib inline


plt.scatter(X[:, 0], X[:, 1]);
indices = np.random.choice(X.shape[0], 20, replace=False);
selection = X[indices]
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(selection[:, 0], selection[:, 1], facecolor='none', s=100)
plt.show()
