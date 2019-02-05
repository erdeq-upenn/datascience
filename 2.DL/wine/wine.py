# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:27:16 2019

@author: Dequan
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab

#print(os.listdir("../data"))


# Any results you write to the current directory are saved as output.

header = ['country','description','designation','points',	'price',	'province',	
          'region_1',	'region_2',	'taster_name',	'taster_twitter_handle',
          'title',	'variety',	'winery']
df = pd.read_csv('data/winemag-data-130k-v2.csv')

n_bins = 40;
#plt.hist(df.points.dropna(),bins=n_bins,alpha=0.6);
n, bins, patches = plt.hist(df.points.dropna(), n_bins, normed=1, facecolor='green', alpha=0.75)

(mu, sigma) = norm.fit(df.points.dropna())
y = mlab.normpdf(bins, mu, sigma)
fig = plt.plot(bins, y, 'r--', linewidth=2)
plt.xlabel('Points')
plt.ylabel('Probability density')
tit = 'Histogram of points: $\mu=%.2f$, $\sigma=%.2f$' %(mu,sigma)
plt.title(tit)

plt.show()
