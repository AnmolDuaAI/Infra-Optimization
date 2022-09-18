import numpy as np
import cv2
import pandas as pd
import math
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm

def plottings_1d(arr,arr1,peaks = None):
    indexes = [i for i in range(len(arr))]
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.plot(indexes,arr)
    ax.plot(indexes,arr1)
    if (peaks is not None):
        ax.plot(peaks,arr[peaks],"x")
    canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) 
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image_from_plot

def polygon_fitting(x,y):
    z = np.polyfit(x,y,3)
    f = np.poly1d(z)
    optimal_func = f
    return optimal_func

def Croston_TSB(ts,extra_periods=1,alpha=0.3559,beta=0.1):
    d = np.array(ts) # Transform the input into a numpy array
    cols = len(d) # Historical period length
    d = np.append(d,[np.nan]*extra_periods) # Append np.nan into the demand array to cover future periods
    
    #level (a), probability(p) and forecast (f)
    a,p,f = np.full((3,cols+extra_periods),np.nan)
    # Initialization
    first_occurence = np.argmax(d[:cols]>0)
    a[0] = d[first_occurence]
    p[0] = 1/(1 + first_occurence)
    f[0] = p[0]*a[0]
                 
    # Create all the t+1 forecasts
    for t in range(0,cols):
        # print (a)
        if d[t] > 0:
            a[t+1] = alpha*d[t] + (1-alpha)*a[t] 
            # print (a[t+1])
            p[t+1] = beta*(1) + (1-beta)*p[t]  
        else:
            a[t+1] = a[t]
            p[t+1] = (1-beta)*p[t]       
        f[t+1] = p[t+1]*a[t+1]
        
    # Future Forecast
    a[cols+1:cols+extra_periods] = a[cols]
    p[cols+1:cols+extra_periods] = p[cols]
    f[cols+1:cols+extra_periods] = f[cols]

    return f

def eva(y):
    res = []
    beta = 0.9
    curr = y[0]
    res.append(curr)
    for i in range(1, len(y)):
        curr = curr * (1-beta) + beta * y[i]
        res.append(curr)
    return res

def demand_history(data):

    # y = []
    # for year in range(2010, 2019):
    #     y.append(np.mean(data.Di[year]))

    # avg_ratio_l = []
    # for i in range(len(y)-1):
    #     c = y[i]
    #     c1 = y[i+1]
    #     if (c!=0):
    #         avg_ratio_l.append(c1/c)
    #     else:
    #         avg_ratio_l.append(0)
    # print (y)
    # print (avg_ratio_l)

    # exit()
    demand_pred_2019 = []
    demand_pred_2020 = []
    for i in tqdm(range(64*64)):
        x = []
        y = []
        curr = 0
        for year in range(2010, 2019):
            x.append(curr)
            y.append(data.Di[year][i])
            curr += 1

        # print (y)
        # model = pm.auto_arima(y,
        #                     d=None,           # let model determine 'd'
        #                     seasonal=False,   # No Seasonality
        #                     start_P=0, 
        #                     D=0, 
        #                     trace=True,
        #                     error_action='ignore',  
        #                     suppress_warnings=True, 
        #                     stepwise=True)

        # # print(model.summary())
        # # print (y)
        # if (y[-1] != 0):
        #     fc, confint = model.predict(n_periods=1, return_conf_int=True)
        #     pred = fc[0]
        # else:
        #     pred = y[-1]
        # print (fc)
        # exit()
        # print (y)
        # pred = Croston_TSB(y)
        # pred = y[-1] + (y[-1] - pred[-1])
        # pred = pred[-1]

        # pred = eva(y)
        # print (pred)
        # exit()
        # print (pred)
        # exit()
        # y = y[-4:]
        # avg_ratio_l = []
        # for i in range(len(y)-1):
        #     c = y[i]
        #     c1 = y[i+1]
        #     if (c!=0):
        #         avg_ratio_l.append(c1/c)
        #     else:
        #         avg_ratio_l.append(0)
        # avg_ratio = np.min(np.array(avg_ratio_l))
        # pred_2019 = y[-1] * avg_ratio


        # print (pred_2019)
        # print (y)
        # print (avg_ratio_l)
        # print (avg_ratio)
        # exit()

        f = polygon_fitting(x,y)
        # poly_pred = f(curr)
        # y = np.array(y)
        # y_diff = y[1:len(y)] - y[0:len(y)-1]
        # if (y_diff[-1] - y_diff[0] < 0):
        #     pred_2019 = y[-1] - abs(y[-1] - f(curr))
        # else:
        #     pred_2019 = y[-1] + abs(y[-1] - f(curr))
        # y1 = [f(i) for i in x]
        # fig = plottings_1d(y, y1)
        # average_diff_rate = np.median(y_diff)
        # cv2.imwrite("./demand_plots/" + str(i) + ".jpg", fig)
        # cv2.imwrite("./demand_plots1/" + str(i) + ".jpg", fig)

        # f = polygon_fitting(x[0:len(y_diff)],y_diff)
        # average_diff_rate = f(curr)

        # f = polygon_fitting(x,y)
        demand_pred_2019.append(f(curr))
        demand_pred_2020.append(f(curr + 1))

        # f = polygon_fitting(x,y)
        # demand_pred_2019.append(pred)
        # demand_pred_2020.append(pred)

        # print ("--------------")

    # exit()
    return demand_pred_2019, demand_pred_2020
