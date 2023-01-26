import yfinance as yf
from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from py_vollib.black_scholes.greeks.analytical import vega
from datetime import datetime
from datetime import date

# obtain the risk free rate
rf_ticker = yf.Ticker("^TNX")  # 10 Year Treasury US
info = rf_ticker.info
r = (info['regularMarketPrice']) / 100  # obtaining risk free rate

current_date = datetime.now().date()  # defining current date


class Vol:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

    def black_scholes(self, option_type, S0, K, T, r, sigma):
        # flag, S0, K, T, r, vol_old
        # step 1: define d1 and d2
        d1 = 1 / (sigma * np.sqrt(T)) * (np.log(S0 / K) + (r + sigma ** 2 / 2) * T)

        d2 = d1 - sigma * np.sqrt(T)

        nd1 = norm.cdf(d1)

        nd2 = norm.cdf(d2)

        n_d1 = norm.cdf(-d1)

        n_d2 = norm.cdf(-d2)

        try:
            if option_type == 'c':
                price = nd1 * S0 - nd2 * K * np.exp(-r * T)
            else:
                price = K * np.exp(-r * T) * n_d2 - S0 * n_d1
            return price
        except:
            print('Please confirm all parameters are correct')

    def implied_vol(self, S0, K, T, r, market_price, flag='c', precision=0.00001):
        max_iter = 200  # max no. of iterations no convergence = no defined zero with function specifed
        vol_old = 0.3  # initial guess

        for k in range(max_iter):
            bs_price = self.black_scholes(flag, S0, K, T, r, vol_old)
            Cprime = vega(flag, S0, K, T, r, vol_old) * 100  # one percent step change in volatility
            C = bs_price - market_price

            vol_new = vol_old - C / Cprime  # formula
            new_bs_price = self.black_scholes(flag, S0, K, T, r, vol_new)
            if abs(vol_old - vol_new) < precision or abs(new_bs_price - market_price) < precision:
                break
            vol_old = vol_new
        implied_vol = vol_new

        return implied_vol


def get_market_price(ticker):
    # obtains the stock price of the ticker
    # step 2. finetune according to brownian motion
    stock = yf.Ticker(ticker)
    price = stock.info["regularMarketPrice"]
    return price


def calculate_T(expiration_date, current_date):
    expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")
    current_date = datetime.strptime(current_date, '%Y-%m-%d')
    T = (expiration_date - current_date).days / 365
    return T

def graph_calc(ticker, option_type = 'c'): # WKHS
    vol = Vol(ticker)
    tick = yf.Ticker(ticker)
    option_dates = tick.options
    expiration_date = option_dates[0]
    if option_type == 'c':
        df = tick.option_chain(expiration_date).calls
    elif option_type == 'p':
        df = tick.option_chain(expiration_date).puts

    df['Black_Scholes'] = (vol.black_scholes(option_type, get_market_price(ticker), df['strike'],
                                             calculate_T(str(expiration_date), str(current_date)), r,
                                             df['impliedVolatility']))

    plt.scatter(df['strike'], df['lastPrice'], s=25, c='blue', marker='o', label='Actual Price')
    plt.scatter(df['strike'], df['Black_Scholes'], s=25, c='red', marker='o', label='Black Scholes Price')
    plt.xlabel('Strike Price')
    plt.ylabel('Option Price')
    plt.title('Option Price Vs Strike Price')
    plt.legend()
    plt.show()

    plt.scatter(df['impliedVolatility'] / 100, df['lastPrice'], s=20, c='blue', marker='o', label='Actual Price')
    plt.scatter(df['impliedVolatility'] / 100, df['Black_Scholes'], s=20, c='red', marker='o', label='Black Scholes Price')
    plt.xlabel('Implied Volatility')
    plt.ylabel('Option Price')
    plt.title('Option Price vs Implied Volatility')
    plt.legend()
    plt.show()

graph_calc("AMZN",'p')



