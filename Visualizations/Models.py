import numpy as np
import cv2
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import os
import math
from matplotlib import pyplot
from tqdm import tqdm
from matplotlib import pyplot as plt

class Logger:
    def __init__(self, root, heading):
        f = open(root + "log.txt", "w")
        f.write("\n")
        f.write("-" * 100 + "\n")
        f.write(heading + "\n")
        f.write("-" * 100 + "\n")
        f.write("\n")
        self.f = f

    def add_text(self, text):
        self.f.write(text + "\n\n")

    def close(self):
        self.f.close()

def MakeDir(path):
    if (not os.path.exists(path)):
        os.mkdir(path)


class ArimaTraining:
    def __init__(self, save_dir, timeseries, params, test_size = 1, debug = True):
        MakeDir(save_dir)
        self.l = Logger(save_dir, "ARIMA_MODEL")
        self.timeseries = timeseries
        self.test_size = test_size
        self.debug = debug
        self.params = params

    
    def fit_arima_model(self, timeseries):
        model = ARIMA(timeseries, order = self.params)
        model_fit = model.fit()
        if (self.debug):
            print (model_fit.summary())
        return model_fit

    def fit_model(self):
        print ("Fitting ARIMA Model")        
        train = self.timeseries[0:-self.test_size]
        test = self.timeseries[-self.test_size:]
        predictions = []
        history = [x for x in train]
        for t in range(test.shape[0]):
            model_fit = self.fit_arima_model(train)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            if (self.debug):
                print ("History : " + str(history[-2]) + " Prediction : " + str(yhat) + " Actual : " + str(test[t]))

        rmse = math.sqrt(mean_squared_error(test, predictions))
        print ("RMSE : " + str(rmse))
        return rmse
    
    def OutOfTimeCrossValidation(self):
        print ("Original Time Series Length : " + str(self.timeseries.shape))
        test_size = int(0.4 * self.timeseries.shape[0])
        train = self.timeseries[:-test_size]
        test = self.timeseries[-test_size:]
        fitted = self.fit_arima_model(train)
        fc = fitted.forecast(test_size, alpha = 0.05) # 95% conf
        print (test)
        print (fc)
        original_plot = np.concatenate((train,test), axis = 0)
        prediction_plot = np.concatenate((train,fc), axis = 0)
        # Plot
        plt.figure(figsize=(12,5), dpi=100)
        plt.plot(original_plot, label='actual')
        plt.plot(prediction_plot, label='forecast')
        plt.plot(train, label='train')
        plt.title('Forecast vs Actuals')
        plt.legend(loc='upper left', fontsize=8)
        plt.savefig("./oocv.jpg")

    # ---------------------------------------
    # Custom functions
    def submission_forecast(self, steps, demand_matrix):

        forecast_matrix = []
        for i in tqdm(range(demand_matrix.shape[0])):
            dp = demand_matrix[i]
            fitted = self.fit_arima_model(dp)
            fc = fitted.forecast(steps, alpha = 0.05) # 95% conf
            forecast_matrix.append(fc)

        forecast_matrix = np.array(forecast_matrix) 
        with open('forecast_matrix.npy', 'wb') as f:
            np.save(f, forecast_matrix)
