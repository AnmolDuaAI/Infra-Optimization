import math
import random
from data_struct import GetData
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot
from helper_time_series import *

data = GetData()

min_year = 2010
max_year = 2018
demand_points = 64 * 64
demand_matrix = []
for year in range(min_year, max_year + 1):
    demand_matrix.append(data.Di[year])

demand_matrix = np.array(demand_matrix).T
print ("Demand Matrix :: " + str(demand_matrix))
print (demand_matrix.shape)

test_stationarity(demand_matrix[0])

detrend_signal = remove_trend(demand_matrix[0])
test_stationarity(detrend_signal)

