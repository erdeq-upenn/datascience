#quick check program of BS pricing
#Dequan

import numpy as np
import scipy.stats as si
import datetime
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

mat_T = datetime.datetime(2019,6,21)
tau=(mat_T-datetime.datetime.now()).days/365

S, K, T, r, sigma = 183*1.0,200,tau,0.03,0.3766  # NVDA 200 call on Jun 21
print('NVDA call %.4f' % (euro_vanilla(S, K, T, r, sigma,option='call')))

S, K, T, r, sigma = 24.95,27.5,65/365,0.03,0.5429  # IQ 27.5 call on Jun 21
print('IQ call %.4f' % (euro_vanilla(S, K, T, r, sigma,option='call')))


S, K, T, r, sigma = 23,28,55/365,0.03,0.44  # LEVI 27.5 call on May 17
print('LEVI call %.4f' % (euro_vanilla(S, K, T, r, sigma,option='call')))
