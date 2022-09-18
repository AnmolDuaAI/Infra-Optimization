
import numpy as np
import cv2
import pandas as pd
from data_struct import *
from utils import *
from visualize_maps import *
from cost import *
import math


# Processing Data Function
def process_data(data_obj, demand_pred_2019):

    # vm = VisualizeMap()

    # Clustering
    # dp_supply_cluster = np.argmin(data.Dist_ij, axis = 1)
    # supply_dp_cluster = np.argmin(data.Dist_ij, axis = 0)
    

     # print (dp_supply_cluster)

    # vm.put_demand_points(data.i, data.dp_station_points)
    # vm.put_supply_points(data.i, data.supply_points)
    # vm.put_dp_cluster_indexes(data.supply_points, data.dp_station_points, data.dp_supply_cluster)
    # vm.draw_map()


    fcs_l = data.FCSj.copy()
    scs_l = data.SCSj.copy()
    distance_matrix = data.Dist_ij.copy()
    demand_points_satisfied = np.array([False for i in range(len(data_obj.i))])
    available_supply_indexes = np.array([True for i in range(100)])
    available_supply_list = data.Smax_j.copy()
    # demand_list = data.Di[2018].copy()
    demand_list = demand_pred_2019.copy()
    # print (np.sum(available_supply_list))

    demand_supply_matrix = np.zeros((len(data.i), len(data.j)), dtype = np.float64)
    while len(np.where(demand_points_satisfied == False)[0]) != 0:
        # print (np.sum(available_supply_list))
        # print (np.sum(demand_list))
        # print ("--------------")
        # exit()
        dp_supply_cluster = np.argmin(distance_matrix, axis = 1)
        supply_dp_cluster = np.argmin(distance_matrix, axis = 0)

        for i in range(100):
            # if (i == 3):
            #     print (np.sum(available_supply_list))
            #     print (np.sum(demand_list))
            #     print ("--------------")
            #     exit()

            if (available_supply_indexes[i]):
                max_supply_available = available_supply_list[i]
                # print (max_supply_available)
                dp_indexes = np.where(dp_supply_cluster == i)[0]
                dp_demand = demand_list[dp_indexes]
                dp_distance = distance_matrix[dp_indexes,i]
                demand_point_l = sorted([[dp_indexes[j], dp_demand[j], dp_distance[j]] for j in range(len(dp_indexes)) if demand_points_satisfied[dp_indexes[j]] == False], key = lambda x: int(x[2]))
                # print (demand_point_l)
                current_supply = max_supply_available
                for demand_point in demand_point_l:
                    demand_point_index, demand_point_d, demand_point_distance = demand_point
                    if (demand_point_d <= current_supply):
                        demand_points_satisfied[demand_point_index] = True
                        current_supply = current_supply - demand_point_d
                        demand_supply_matrix[demand_point_index, i] = demand_point_d
                        available_supply_list[i] = current_supply
                        demand_list[demand_point_index] = 0
                    else:
                        demand_list[demand_point_index] -= current_supply
                        demand_supply_matrix[demand_point_index, i] = current_supply

                    # print (current_supply)
                    # if (current_supply <= 0):
                        # print ("I am done")
                        available_supply_indexes[i] = False
                        distance_matrix[:,i] = np.inf
                        available_supply_list[i] = 0
                        # current_supply = 0

                        break
    # print ("---------> " + str(available_supply_indexes[i]))
    # print (len(np.where(demand_points_satisfied == True)[0]))
    # print (len(np.where(available_supply_indexes == True)[0]))
    # print (demand_list[np.where(demand_points_satisfied == False)[0]])
    # print (demand_supply_matrix)

    for i in range(demand_supply_matrix.shape[0]):
        for j in range(demand_supply_matrix.shape[1]):
            if (demand_supply_matrix[i,j] != 0):
                demand_supply_matrix[i,j] -= 0.000000001
    # print (demand_supply_matrix[0,38])

    constraints_satisfied(demand_supply_matrix, data.SCSj, data.FCSj, data.Di[2018], data)
    d = {
        "2019":{
            "scs": data.SCSj,
            "fcs": data.FCSj,
            "ds": demand_supply_matrix
        },
        "2020":{
            "scs": data.SCSj,
            "fcs": data.FCSj,
            "ds": demand_supply_matrix
        }
    }      

    submission_out(d)

# Processing Data Function
def process_data1(data, demand_pred_2019, fcs_pred, scs_pred):

    fcs_l = fcs_pred.copy()
    scs_l = scs_pred.copy()
    distance_matrix = data.Dist_ij.copy()
    demand_points_satisfied = np.array([False for i in range(len(data.i))])
    available_supply_indexes = np.array([True for i in range(100)])
    # available_supply_list = data.Smax_j.copy()
    available_supply_list = []
    for i in range(len(fcs_l)):
        n_fcs = fcs_l[i]
        n_scs = scs_l[i]
        available_supply_list.append(n_fcs * 400 + n_scs * 200)
    demand_list = np.array(demand_pred_2019.copy())

    # for i in range(distance_matrix.shape[0]):
    #     for j in range(distance_matrix.shape[1]):
    #         distance_matrix[i,j] = distance_matrix[i,j] * abs(demand_list[i]/available_supply_list[j])

    demand_supply_matrix = np.zeros((len(data.i), len(data.j)), dtype = np.float64)
    while len(np.where(demand_points_satisfied == False)[0]) != 0:
        dp_supply_cluster = np.argmin(distance_matrix, axis = 1)
        supply_dp_cluster = np.argmin(distance_matrix, axis = 0)

        for i in range(100):
            if (available_supply_indexes[i]):
                max_supply_available = available_supply_list[i]
                dp_indexes = np.where(dp_supply_cluster == i)[0]
                dp_demand = demand_list[dp_indexes]
                dp_distance = distance_matrix[dp_indexes,i]
                demand_point_l = sorted([[dp_indexes[j], dp_demand[j], dp_distance[j]] for j in range(len(dp_indexes)) if demand_points_satisfied[dp_indexes[j]] == False], key = lambda x: int(x[2]))
                current_supply = max_supply_available
                for demand_point in demand_point_l:
                    demand_point_index, demand_point_d, demand_point_distance = demand_point
                    if (demand_point_d <= current_supply):
                        demand_points_satisfied[demand_point_index] = True
                        current_supply = current_supply - demand_point_d
                        demand_supply_matrix[demand_point_index, i] = demand_point_d
                        available_supply_list[i] = current_supply
                        demand_list[demand_point_index] = 0
                    else:
                        demand_list[demand_point_index] -= current_supply
                        demand_supply_matrix[demand_point_index, i] = current_supply

                        available_supply_indexes[i] = False
                        distance_matrix[:,i] = np.inf
                        available_supply_list[i] = 0

                        break

    for i in range(demand_supply_matrix.shape[0]):
        for j in range(demand_supply_matrix.shape[1]):
            if (demand_supply_matrix[i,j] != 0):
                demand_supply_matrix[i,j] -= 0.000000001

    # return demand_supply_matrix

    constraints_satisfied(demand_supply_matrix, scs_pred, fcs_pred, demand_pred_2019, data)
    d = {
        "2019":{
            "scs": scs_pred,
            "fcs": fcs_pred,
            "ds": demand_supply_matrix
        },
        "2020":{
            "scs": scs_pred,
            "fcs": fcs_pred,
            "ds": demand_supply_matrix
        }
    }

    submission_out(d)
