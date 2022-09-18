import numpy as np
import cv2
import pandas as pd

class GetData:
    def __init__(self):
        p = "../Files/Demand_History.csv"
        demand_history = pd.read_csv(p)
        self.demand_point_index = demand_history["demand_point_index"].values
        self.dp_x_coordinates = demand_history["x_coordinate"].values
        self.dp_y_coordinates = demand_history["y_coordinate"].values
        self.year_demand_history = {}
        for year in range(2010, 2019):
            self.year_demand_history[year] = demand_history[str(year)].values

        p1 = "../Files/exisiting_EV_infrastructure_2018.csv"
        infras = pd.read_csv(p1)
        self.supply_point_index = infras["supply_point_index"].values
        self.infra_x_coordinates = infras["x_coordinate"].values
        self.infra_y_coordinates = infras["y_coordinate"].values
        self.infra_tps = infras["total_parking_slots"].values
        self.infra_scs = infras["existing_num_SCS"].values
        self.infra_fcs = infras["existing_num_FCS"].values

        self.make_datastructures()
    

    def make_datastructures(self):
        # Index of demand points
        self.i = np.unique(self.demand_point_index)

        # Index of supply points
        self.j = np.unique(self.supply_point_index)

        # EV Charging demand
        self.Di = self.year_demand_history

        # Number of slow slow charging stations
        self.SCSj = self.infra_scs
        
        # Number of fast charging stations
        self.FCSj = self.infra_fcs

        # Total Parking slots
        self.PSj = self.infra_tps

        # Charging Capacity of slow charging station
        self.cap_scs = 200

        # Charging Capacity of fast charging station
        self.cap_fcs = 400

        # Maximum supply that can be given from charging station
        self.Smax_j = [self.cap_scs * self.SCSj[i] + self.cap_fcs * self.FCSj[i] for i in range(len(self.SCSj))]

        # Distance Matrix - distance between i-th demand point and j-th supply point 
        dp_station_points = np.array([[self.dp_x_coordinates[i], self.dp_y_coordinates[i]] for i in range(len(self.dp_x_coordinates))])
        supply_points = np.array([[self.infra_x_coordinates[i], self.infra_y_coordinates[i]] for i in range(len(self.infra_x_coordinates))])
        self.Dist_ij = np.sqrt(np.sum(np.square(np.repeat(np.expand_dims(dp_station_points, 1), 100, axis = 1) - np.repeat(np.expand_dims(supply_points, 0), 4096, axis = 0)), axis = 2))

        # Demand-Supply Matrix
        # construct_demand_supply_matrix(self.Dist_ij, self.Di[2018], self.FCSj, self.SCSj, self.PSj)

        self.dp_station_points = dp_station_points
        self.supply_points = supply_points
