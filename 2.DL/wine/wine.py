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
plt.style.use('seaborn-white')

#print(os.listdir("../data"))


# Any results you write to the current directory are saved as output.

header = ['country','description','designation','points',	'price',	'province',	
          'region_1',	'region_2',	'taster_name',	'taster_twitter_handle',
          'title',	'variety',	'winery']
df = pd.read_csv('data/winemag-data-130k-v2.csv')

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
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


plt.subplot(1,2,2)
n_bins = 40;
#plt.hist(df.points.dropna(),bins=n_bins,alpha=0.6);
normal_p = 100
prices2 = df[df.price <normal_p].price
n, bins, patches = plt.hist(prices2, n_bins, normed=1, facecolor='red', alpha=0.75)

(mu, sigma) = norm.fit(prices2)
y = mlab.normpdf(bins, mu, sigma)
fig = plt.plot(bins, y, 'b--', linewidth=2)
plt.xlabel('prices')

plt.ylabel('Probability density')
tit = 'Histogram of prices: $\mu=%.2f$, $\sigma=%.2f$' %(mu,sigma)
plt.title(tit)

#Plot 2D histogram picture; initial is sharpe; to visulize added random
plt.show()
plt.figure()
jiaqian = df[df.price <normal_p].price+np.random.random(jiaqian.size)*0.2
pingfen = df[df.price <normal_p].points +np.random.random(jiaqian.size)*0.2
plt.hexbin(jiaqian.fillna(0),pingfen.fillna(0),bins=40,cmap = 'Blues')
#plt.xlim([0,100])
plt.show()
