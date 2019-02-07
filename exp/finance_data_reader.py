# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:18:17 2019

@author: Dequan Er
"""
import os
import numpy as np
#import pandas_datareader as pdr
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
#
##df = pdr.get_data_tiingo('GOOG', api_key=os.getenv('TIINGO_API_KEY'))
start = datetime.datetime(2019,1,1)
end = datetime.datetime(2019,2,6)
#f = web.DataReader('F','morningstar',start,end)

df = web.DataReader('F', 'iex', start, end)
plt.figure()
#plt.subplot(1,2,1)
plt.plot(df.index,df['volume']/np.max(df.volume)*10,'--r')
#plt.subplot(1,2,2)
plt.plot(df.index,df['high'])
plt.show()