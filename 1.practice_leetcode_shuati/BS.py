#quick check program of BS pricing
#Dequan

import numpy as np
import scipy.stats as si
import datetime
import matplotlib.pyplot as plt
#import sympy as sy
#import sympy.statistics as systats

# This is a funtion of european vanilla call/put withou paying dividend
def euro_vanilla(S, K, T, r, sigma, option = 'dummy'):

    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option == 'call':
        result = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    if option == 'put':
        result = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))

    return result

# print('European vanilla option without D:',euro_vanilla(50, 100, 1, 0.05, 0.25,option='put'))

# This is a funtion of european vanilla call/put withou paying dividend
def euro_vanilla_dividend(S, K, T, r, q, sigma, option = 'dummy'):

    #S: stock price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #q: rate of continuous dividend paying asset
    #sigma: volatility of underlying asset

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option == 'call':
        result = (S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    if option == 'put':
        result = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0))

    return result
# print('European vanilla option with D:',euro_vanilla_dividend(50, 100, 1, 0.05, 0.2,0.25,option='put'))


########################################
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
#########################################
r = 0.03
mat_T1 = datetime.datetime(2019,6,21)
mat_T2 = datetime.datetime(2019,5,17)
mat_T3 = datetime.datetime(2019,4,26)
tau1=(mat_T1-datetime.datetime.now()).days/365
tau2=(mat_T2-datetime.datetime.now()).days/365
tau3=(mat_T3-datetime.datetime.now()).days/365

S, K, T, r, sigma = 215,200,tau1,r,0.3766  # NVDA 200 call on Jun 21
print('NVDA call %.4f' % (euro_vanilla(S, K, T, r, sigma,option='call')))

S, K, T, r, sigma = 32,27.5,tau1,r,0.5430  # IQ 27.5 call on Jun 21
print('IQ call %.4f' % (euro_vanilla(S, K, T, r, sigma,option='call')))


S, K, T, r, sigma = 23,28,tau2,r,0.44  # LEVI 27.5 call on May 17
print('LEVI call %.4f' % (euro_vanilla(S, K, T, r, sigma,option='call')))



# S, K, T, r, sigma = 32,30,tau3,r,0.6829  # AMD 30 call on 04/26/
S, K, T, r, sigma = 215,200,tau1,r,0.3766  # NVDA 200 call on Jun 21
def delta_S(S):
    call_IQ = euro_vanilla(S,K, T, r, sigma,option='call')

    return call_IQ
dS = np.linspace(185,205,100)
call = delta_S(dS)
callNoise = delta_S(dS)+np.random.randn(100)*dS*0.001

plt.scatter(dS,callNoise)
plt.plot(dS,call,'--r')
plt.show()

def ss(S):
    ss_price = euro_vanilla(S,K, T, r, sigma,option='call')+euro_vanilla(S,K, T, r, sigma,option='put')

    return ss_price
plt.figure()
dS = np.linspace(185,205,100)
straddle = ss(dS)-euro_vanilla(K,K, T, r, sigma,option='call')-euro_vanilla(K,K, T, r, sigma,option='put')
callNoise = delta_S(dS)+np.random.randn(100)*dS*0.001
# plt.scatter(dS,callNoise)
plt.plot(dS,straddle,'--r')
plt.show()
