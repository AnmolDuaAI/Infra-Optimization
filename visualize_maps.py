import numpy as np
import cv2
import pandas as pd

class VisualizeMap:
    def __init__(self):
        self.block_width = 256
        self.block_height = 256
        self.rows = 64
        self.cols = 64

        self.map = self.make_new_map_image()

        # Cluster
        self.cluster_map = self.make_new_map_image()
        self.colors_supply_l = []
        for i in range(100):
            self.colors_supply_l.append((np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)))

    def make_new_map_image(self):
        map = np.zeros((self.block_height * self.cols, self.block_width * self.rows, 3), dtype = np.uint8)
        for x in range(self.cols):
            cv2.line(map, (x*self.block_width, 0), (x*self.block_width, self.rows * self.block_height), (128,128,128), 2)
        for y in range(self.rows):
            cv2.line(map, (0, y*self.block_height), (self.cols * self.block_width, y*self.block_height), (128,128,128), 2)
        return map

    def put_demand_points(self, demand_indexes, demand_points):
        for idx in range(len(demand_points)):
            dp = demand_points[idx]
            x,y = dp
            x = int(x * self.block_width)
            y = int(y * self.block_height)
            self.map = cv2.circle(self.map, (x,y), 10, (0,0,255), -1)
    
    def put_supply_points(self, supply_indexes, supply_points):
        for idx in range(len(supply_points)):
            dp = supply_points[idx]
            x,y = dp
            x = int(x * self.block_width)
            y = int(y * self.block_height)
            self.map = cv2.circle(self.map, (x,y), 10, (0,255,0), 4)

    def put_dp_cluster_indexes(self, supply_points, demand_points, cluster_indexes):
        for idx in range(len(supply_points)):
            dp = supply_points[idx]
            x,y = dp
            x = int(x * self.block_width)
            y = int(y * self.block_height)
            self.custer_map = cv2.circle(self.cluster_map, (x,y), 30, self.colors_supply_l[idx], 4)

        for idx in range(len(cluster_indexes)):
            dp = demand_points[idx]
            c = self.colors_supply_l[cluster_indexes[idx]]
            x,y = dp
            x = int(x * self.block_width)
            y = int(y * self.block_height)
            self.cluster_map = cv2.circle(self.cluster_map, (x,y), 10, c, -1)

    def draw_map(self):
        cv2.imwrite("./Map.jpg", self.map)
        cv2.imwrite("./ClusterMap.jpg", self.cluster_map)
