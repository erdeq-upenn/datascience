# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 11:18:16 2019

@author: Alex
"""

import numpy as np
rand = np.random.RandomState(42)

x = rand.randint(100, size=10)
print(x)

ind = [3,4,7]
X = np.arange(12).reshape((3, 4))
print(x[ind])

row = np.array([0, 1, 2])
col = np.array([2, 1, 3])

Y = X[row,col]
