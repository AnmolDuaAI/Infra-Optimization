import numpy as np
import cv2
import pandas as pd


def constraints_satisfied(demand_supply_matrix, scs_number, fcs_number, demand_forecast, data):
    
    demand_supply_matrix = np.array(demand_supply_matrix)
    scs_number = np.array(scs_number)
    fcs_number = np.array(fcs_number)
    
    # all values must be non-negative
    negative_present1 = len(np.where(demand_supply_matrix < 0)[0]) == 0
    negative_present2 = len(np.where(scs_number < 0)[0]) == 0
    negative_present3 = len(np.where(fcs_number < 0)[0]) == 0

    if (negative_present1 and negative_present2 and negative_present3):
        print ("Check 1 Pass!")
    else:
        print ("Check 1 Failed")

    # FCS + SCS <= TPS
    cond = True
    for i in range(len(scs_number)):
        scs_i = scs_number[i]
        fcs_i = fcs_number[i]
        tps_i = data.PSj[i]
        if (scs_i + fcs_i > tps_i):
            cond = False
            break

    if (cond):
        print ("Check 2 Pass!")
    else:
        print ("Check 2 Fail")

    # Incremental FCS/SCS
    cond = True
    for i in range(len(scs_number)):
        scs_i = scs_number[i]
        fcs_i = fcs_number[i]
        scs_2018 = data.SCSj[i]
        fcs_2018 = data.FCSj[i]
        if (scs_i < scs_2018 or fcs_i < fcs_2018):
            cond = False
            break

    if (cond):
        print ("Check 3 Pass!")
    else:
        print ("Check 3 Fail")
    
    # Demand Satisfied
    cond = True
    for j in range(len(data.Smax_j)):
        smax_j = data.Smax_j[j]
        sum_supply = 0
        for i in range(len(demand_supply_matrix)):
            sum_supply += demand_supply_matrix[i,j]

        if (sum_supply > smax_j):
            print (sum_supply)
            print (smax_j)
            cond = False
            break

    if (cond):
        print ("Check 4 Pass!")
    else:
        print ("Check 4 Fail")


    # Sum of Fractional
    cond = True
    for i in range(len(demand_forecast)):
        df = demand_forecast[i]
        sum_demand = 0
        for j in range(len(data.Smax_j)):
            sum_demand += demand_supply_matrix[i, j]
        if (df != sum_demand):
            print (df)
            print (sum_demand)
            cond = False
            break

    if (cond):
        print ("Check 5 Pass!")
    else:
        print ("Check 5 Fail")

# Submission Output
def submission_out(d):

    scs_number = d["2019"]["scs"]
    fcs_number = d["2019"]["fcs"]
    demand_supply_matrix = d["2019"]["ds"]
    year = 2019

    year_l = []
    data_type_l = []
    demand_point_index = []
    supply_point_index = []
    value = []

    for i in range(len(scs_number)):
        year_l.append(year)
        data_type_l.append("SCS")
        demand_point_index.append("")
        supply_point_index.append(i)
        value.append(scs_number[i])

    for i in range(len(fcs_number)):
        year_l.append(year)
        data_type_l.append("FCS")
        demand_point_index.append("")
        supply_point_index.append(i)
        value.append(fcs_number[i])

    for i in range(demand_supply_matrix.shape[0]):
        for j in range(demand_supply_matrix.shape[1]):
            year_l.append(year)
            data_type_l.append("DS")
            demand_point_index.append(i)
            supply_point_index.append(j)
            value.append(demand_supply_matrix[i,j])

    scs_number = d["2020"]["scs"]
    fcs_number = d["2020"]["fcs"]
    demand_supply_matrix = d["2020"]["ds"]
    year = 2020

    for i in range(len(scs_number)):
        year_l.append(year)
        data_type_l.append("SCS")
        demand_point_index.append("")
        supply_point_index.append(i)
        value.append(scs_number[i])

    for i in range(len(fcs_number)):
        year_l.append(year)
        data_type_l.append("FCS")
        demand_point_index.append("")
        supply_point_index.append(i)
        value.append(fcs_number[i])

    for i in range(demand_supply_matrix.shape[0]):
        for j in range(demand_supply_matrix.shape[1]):
            year_l.append(year)
            data_type_l.append("DS")
            demand_point_index.append(i)
            supply_point_index.append(j)
            value.append(demand_supply_matrix[i,j])

    df = pd.DataFrame(list(zip(year_l, data_type_l, demand_point_index, supply_point_index, value)),
                    columns =['year', "data_type", "demand_point_index", "supply_point_index", "value"])

    df.to_csv("./submission.csv", index = False)