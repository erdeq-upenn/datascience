#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 23:47:55 2019

@author: Dequan
"""

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import seaborn 
style.use('ggplot')


#import pandas_datareader as pdr
#x=pdr.get_data_fred('GS10')
#



start = dt.datetime(2019, 1, 1)
end = dt.datetime.now()

df = web.DataReader('TSLA', 'iex', start, end)
df.reset_index(inplace=True)
df.set_index("Date", inplace=True)
df = df.drop("Symbol", axis=1)

print(df.head())
df.to_csv('tesla.csv')
