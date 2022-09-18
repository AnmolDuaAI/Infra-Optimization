import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
from matplotlib import pyplot
from matplotlib import pyplot as plt


'''
    Helper Utilies for Time Series - ARIMA Model
'''

'''
    Stationary Test for time series::
    1. ADF ( Augmented Dicky Fuller Test ) - Currently supported
    2. KPSS
    3. PP (Phillips - Pearson Test)
'''

def test_stationarity(timeseries, method = "ADF"):
    print ("Testing for Stationarity : ")
    if (method == "ADF"):
        print ("--------------------------------------")
        print ("Results for Dickey-Fuller Test")
        dftest = adfuller(timeseries, autolag = 'AIC')
        print ("ADF STatistics : " + str(dftest[0]))
        print ("PValue : " + str(dftest[1]))
        print ("Critical Values :::")
        for key, value in dftest[4].items():
            print (key + " : " + str(value))
        if (dftest[0] < dftest[4]["5%"]):
            print ("Reject H0 - TIme Series is Stationary!!")
        else:
            print ("Failed to reject H0 - Time Series is Non Stationary!")
        print ("--------------------------------------")
    else:
        print ("Not Supported Method!!")

def plot_autocorrelation(timeseries, save_path):
    plot_acf(timeseries)
    pyplot.savefig(save_path)

def  plot_partial_autocorrelation(timeseries, save_path, lags = 8):
    plot_pacf(timeseries, lags = lags)
    pyplot.savefig(save_path)


def order_of_differencing_vis(timeseries, save_path, orders_l = [1]):
    plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
    fig, axes = plt.subplots(1 + len(orders_l), 2, sharex=True)
    axes[0, 0].plot(timeseries)
    axes[0, 0].set_title('Original Series')
    plot_acf(timeseries, ax=axes[0, 1])
    curr = 0
    for order in orders_l:
        diff = remove_trend(timeseries, order)
        axes[1 + curr, 0].plot(diff)
        axes[1 + curr, 0].set_title('Order Differencing : ' + str(order))
        plot_acf(diff, ax=axes[1+curr, 1])
        curr += 1
    pyplot.savefig(save_path)

def order_of_ar_vis(timeseries, save_path):
    # PACF plot of 1st differenced series
    diff = remove_trend(timeseries, 1)
    plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
    fig, axes = plt.subplots(1, 2, sharex=True)
    axes[0].plot(diff); axes[0].set_title('1st Differencing')
    axes[1].set(ylim=(0,5))
    plot_pacf(diff, ax=axes[1], lags = 3)
    pyplot.savefig(save_path)

def order_of_mv_vis(timeseries, save_path):
    # PACF plot of 1st differenced series
    diff = remove_trend(timeseries, 1)
    plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
    fig, axes = plt.subplots(1, 2, sharex=True)
    axes[0].plot(diff); axes[0].set_title('1st Differencing')
    axes[1].set(ylim=(0,5))
    plot_acf(diff, ax=axes[1])
    pyplot.savefig(save_path)


def remove_trend(timeseries, interval = 1, method = "diff"):
    
    diff = []
    if (method == "diff"):
        for i in range(interval, len(timeseries)):
            value = timeseries[i] - timeseries[i - interval]
            diff.append(value)
        return np.array(diff)
    else:
        print ("Method Not Supported!!!")
    return 0
