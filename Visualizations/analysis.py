import math
import random
from data_struct import GetData
import cv2
import numpy as np
import pandas as pd
from helper_time_series import *
from Models import *
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def dp_autocorrelations(demand_matrix):
    save_dir = "./DemandPointsAutoCorr/"
    MakeDir(save_dir)
    for i in tqdm(range(demand_matrix.shape[0])):
        plot_autocorrelation(demand_matrix[i], save_path = save_dir + str(i) + ".jpg")


def dp_partial_autocorr(demand_matrix):
    save_dir = "./DemandPointsPartialAutoCorr/"
    MakeDir(save_dir)
    for i in tqdm(range(demand_matrix.shape[0])):
        plot_partial_autocorrelation(demand_matrix[i], lags = 3, save_path = save_dir + str(i) + ".jpg")


def model_training():
    total_err = 0
    search_lag = [1,2,3,4,5]
    order = [1,2]
    mva = [0,1,2]
    for i in range(demand_points):
        print ("Current Demand Point : " + str(i))
        
        best_err = 10000
        best_params = None
        for sl in search_lag:
            for o in order:
                for m in mva:
                    rmse_err = fit_arima_model(demand_matrix[i], sl,o,m, save_dir)
                    if (rmse_err < best_err):
                        best_err = rmse_err
                        best_params = [sl,o,m]
        print (best_err)
        print (best_params)
        total_err += best_err
    print ("Total Err : " + str(total_err))


save_dir = "./experiments/"
MakeDir(save_dir)

data = GetData()

min_year = 2010
max_year = 2018
demand_points = 64 * 64
demand_matrix = []
for year in range(min_year, max_year + 1):
    demand_matrix.append(data.Di[year])

demand_matrix = np.array(demand_matrix).T

# Plotting Autocorrelations
# dp_autocorrelations(demand_matrix)
# dp_partial_autocorr(demand_matrix)

# order of diff = 1 seems to ok
# order_of_differencing_vis(demand_matrix[3000], orders_l = [1,2,3], save_path = "demand_point.jpg")

# p = 1 seems to be right choice
# order_of_ar_vis(demand_matrix[3000], save_path = "ar_order.jpg")

# q = 1 seems to right choice
# order_of_mv_vis(demand_matrix[3000], save_path = "mv_order.jpg")

# After checking out significance of p value mv = 0
timeseries = demand_matrix[100]
arima = ArimaTraining(save_dir, timeseries, (1,1,0), test_size = 2, debug = False)
# arima.OutOfTimeCrossValidation()
arima.submission_forecast(demand_matrix = demand_matrix, steps = 2)