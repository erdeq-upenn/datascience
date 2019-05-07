# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:53:10 2019

@author: Dequan Er
"""
import math

def seq(n,k,j):
    fac = 1
    for i in range(n-2):
        for j in range(k):
            fac *=i*(k-i)/k
    
    return fac