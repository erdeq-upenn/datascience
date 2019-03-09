#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:43:12 2019

@author: Dequan
"""

# multiply 

def make_df(cols,ind):
    data = {c:[str(c) +str(i) for i in ind] for c in cols}
    return pd.DataFrame(data,ind)

