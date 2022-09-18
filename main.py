from utils import *
from visualize_maps import *
from cost import *
import math
import math
import random
from demand_forecast import demand_history
from pos import PSO
from Clustering import *
from clustering_pso import PSODS

with open('./Visualizations/forecast_matrix.npy', 'rb') as f:
    a = np.load(f)

data = GetData()
# demand_pred_2019, demand_pred_2020 = demand_history(data)
demand_pred_2019 = a[:,0]
demand_pred_2019[np.where(demand_pred_2019 < 0)[0]] = 0
data.Di[2019] = demand_pred_2019
# demand_pred_2019 = data.Di[2018]
# demand_pred_2019 = np.array([i*1.2 for i in demand_pred_2019])
# demand_pred_2019 = np.array([i*1.2 for i in demand_pred_2019])
# data.Di[2019] = demand_pred_2019
fcs_pred, scs_pred = PSO(data, 15, 100)
demand_supply_matrix = process_data1(data, demand_pred_2019, fcs_pred, scs_pred)
# PSODS(data, demand_supply_matrix, fcs_pred, scs_pred, 15, 100)