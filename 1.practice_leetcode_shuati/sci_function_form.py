# import pandas as pd
import numpy as np
import scipy
# import seaborn as sns
# import matplotlib.pyplot as plt

# Dequan Er
# 2019-05-06
# print('Hello World\n')


def bdf(f,x0,h=0.001):
    df = (f(x0+h)-f(x0))/h
    return df


print(bdf(np.cos,0,h=0.001))
