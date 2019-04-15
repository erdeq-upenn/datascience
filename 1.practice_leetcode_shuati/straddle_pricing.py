import numpy as np
import scipy.stats as si
import datetime
import matplotlib.pyplot as plt
import matplotlib.collections as collections


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

def straddle(num_put,num_call,put_price,call_price,tau,S):
    res = num_call* euro_vanilla(S, K, T, r, sigma, option = 'call')\
    +num_put* euro_vanilla(S, K, T, r, sigma, option = 'put')
    cost = call_price*num_call+put_price*num_put
    return (res-cost)

def strangle(num_put,num_call,call_price,put_price,k_call,k_put,tau,S):
    res = num_call* euro_vanilla(S, k_call, tau, r, sigma, option = 'call')\
    +num_put* euro_vanilla(S, k_put, tau, r, sigma, option = 'put')
    cost = call_price*num_call+put_price*num_put

    return (res-cost)

########################################
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
#########################################
r = 0.03
mat_T0 = datetime.datetime.now()

mat_T1 = datetime.datetime(2019,4,26)
mat_T2 = datetime.datetime(2019,5,17)
mat_T3 = datetime.datetime(2019,6,21)

tau0=(mat_T0-datetime.datetime.now()).days/365
tau1=(mat_T1-datetime.datetime.now()).days/365
tau2=(mat_T2-datetime.datetime.now()).days/365
tau3=(mat_T3-datetime.datetime.now()).days/365


S, K, T, r, sigma = 31.09,30,tau2,r,0.8368  # NVDA 200 call on Jun 21
# print('NVDA call %.4f' % (straddle(1,1,tau1,S)))

num_put,num_call,put_price,call_price = 1,1,2.8,3.8
k_call,k_put = 31,29
k_call_p,k_put_p = 2.9, 1.75
dS = np.linspace(20,35,100)
straddle = straddle(num_put,num_call,put_price,call_price,tau2,dS)
strangle_price = strangle(num_put,num_call,k_call_p,k_put_p,k_call,k_put,tau2,dS)

f, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 4) ,sharey=True)
ax1.plot(dS,straddle,'--g',lw=2)
ax1.axhline(0, color='black', lw=2)
ax1.set_title('Straddle')
# collection = collections.BrokenBarHCollection.span_where(
#     dS, ymin=min(straddle), ymax=0, where=straddle < 0, facecolor='red', alpha=0.5)
# ax.add_collection(collection)

ax1.fill_between(dS, straddle, where=straddle <=0, facecolor='red', interpolate=True)
# visulize current price

# plt.axvline(x=S, ymin=y_curr, ymax =0, linewidth=2, color='k')
# figure2,ax = plt.subplots(2,1,2,sharex='row')
ax2.plot(dS,strangle_price,'--r',lw=2)
ax2.axhline(0, color='black', lw=2)
ax2.fill_between(dS, strangle_price, where=strangle_price <=0, facecolor='green', interpolate=True)
ax2.set_title('Strangle')
plt.show()
print(euro_vanilla(S, 28, T, r, sigma, option = 'put'))
