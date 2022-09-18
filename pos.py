
import numpy as np
import cv2
import pandas as pd
from data_struct import *
from utils import *
from visualize_maps import *
from cost import *
import math
import random
from demand_forecast import demand_history

# Particle Swarm Optimization
#--- MAIN ---------------------------------------------------------------------+

def cost_function(params_l, fcs_len, scs_len, total_demand, fcs_charging = 400, scs_charging = 200):
    fcs = params_l[:fcs_len]
    scs = params_l[fcs_len:fcs_len + scs_len]

    total_supply = 0
    cost_infrasture = 0
    for i in range(len(fcs)):
        fcs_n = fcs[i]
        scs_n = scs[i]
        total_supply += (fcs_n * fcs_charging + scs_n * scs_charging)
        cost_infrasture += (scs_n + 1.5 * fcs_n)

    if (total_demand > total_supply):
        return (total_demand - total_supply) + cost_infrasture
    else:
        return cost_infrasture

class Particle:
    def __init__(self,data):
                
        fcs_initial = data.FCSj.copy()
        scs_initial = data.SCSj.copy()


        ps = data.PSj.copy()
        total_demand = np.sum(data.Di[2019])

        self.fcs_len = len(fcs_initial)
        self.scs_len = len(scs_initial)
        self.total_demand = total_demand
        self.fcs_initial = fcs_initial
        self.scs_initial = scs_initial
        self.ps = ps


        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        self.num_dimensions = len(fcs_initial) + len(scs_initial)
        self.fcs_len = len(fcs_initial)
        self.scs_len = len(scs_initial)

        for i in range(len(fcs_initial)):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(fcs_initial[i])

        for i in range(len(scs_initial)):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(scs_initial[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i, self.fcs_len, self.scs_len, self.total_demand)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.7       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant

        for i in range(0,self.num_dimensions):
            r1=random.random()
            r2=random.random()
            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self):
        
        current_pos = self.position_i
        fcs = current_pos[0:self.fcs_len]
        v_fcs = self.velocity_i[0:self.fcs_len]
        scs = current_pos[self.fcs_len:self.fcs_len + self.scs_len]
        v_scs = self.velocity_i[self.fcs_len:self.fcs_len + self.scs_len]

        new_position_fcs = []
        new_position_scs = []
        for i in range(len(fcs)):
            np_fcs = round(fcs[i] + 2*v_fcs[i])
            np_scs = round(scs[i] + 2*v_scs[i])
            if (np_fcs < self.fcs_initial[i]):
                np_fcs = self.fcs_initial[i]
            if (np_scs < self.scs_initial[i]):
                np_scs = self.scs_initial[i]
            if (np_fcs + np_scs > self.ps[i]):
                while (np_fcs + np_scs > self.ps[i]):
                    choice = random.randint(0,1)
                    if (choice == 1):
                        np_scs = np_scs - 1
                    else:
                        np_fcs = np_fcs - 1
                    if (np_fcs < self.fcs_initial[i]):
                        np_fcs = self.fcs_initial[i]
                    if (np_scs < self.scs_initial[i]):
                        np_scs = self.scs_initial[i]

                # np_scs = self.ps[i] - np_fcs
            if (np_fcs < 0):
                np_fcs = 0
            if (np_scs < 0):
                np_scs = 0
            new_position_fcs.append(np_fcs)
            new_position_scs.append(np_scs)
        # print (fcs)
        # print (v_fcs)
        # print (new_position_fcs)
        # exit()
        self.position_i = new_position_fcs + new_position_scs

# 64 x 64
# 4096 - 
def compare_result(data, pos_best_g, err_best_g):
    fcs_initial = data.FCSj.copy()
    scs_initial = data.SCSj.copy()
    ps = data.PSj.copy()
    total_demand = np.sum(data.Di[2019])

    total_supply_before = 0
    cost_infrasture_before = 0
    for i in range(len(fcs_initial)):
        fcs_n = fcs_initial[i]
        scs_n = scs_initial[i]
        total_supply_before += (fcs_n * 400 + scs_n * 200)
        cost_infrasture_before += (scs_n + 1.5 * fcs_n)

    fcs_len = len(fcs_initial)
    scs_len = len(scs_initial)

    fcs = pos_best_g[:fcs_len]
    scs = pos_best_g[fcs_len:fcs_len + scs_len]

    total_supply_after = 0
    cost_infrasture_after = 0
    for i in range(len(fcs)):
        fcs_n = fcs[i]
        scs_n = scs[i]
        total_supply_after += (fcs_n * 400 + scs_n * 200)
        cost_infrasture_after += (scs_n + 1.5 * fcs_n)

    print ("Demand : " + str(total_demand))
    print ("Before Optimization : ")
    print ("Supply : " + str(total_supply_before))
    print ("Cost Infra : " + str(cost_infrasture_before))
    print ("After Optimization : ")
    print ("Supply : " + str(total_supply_after))
    print ("Cost Infra : " + str(cost_infrasture_after))

def PSO(data,num_particles,maxiter):
    
    err_best_g=-1                   # best error for group
    pos_best_g=[]                   # best position for group

    # establish the swarm
    swarm=[]
    for i in range(0,num_particles):
        swarm.append(Particle(data))
    
    # begin optimization loop
    i=0
    while i < maxiter:
        # cycle through particles in swarm and evaluate fitness
        for j in range(0,num_particles):
            swarm[j].evaluate(cost_function)

            # determine if current particle is the best (globally)
            if swarm[j].err_i < err_best_g or err_best_g == -1:
                pos_best_g=list(swarm[j].position_i)
                err_best_g=float(swarm[j].err_i)

        # cycle through swarm and update velocities and position
        for j in range(0,num_particles):
            swarm[j].update_velocity(pos_best_g)
            swarm[j].update_position()
        i+=1

        print ("i : " + str(i))
        print (pos_best_g)
        print (err_best_g)
        print ()

    print ('FINAL:')
    print (pos_best_g)
    print (err_best_g)

    compare_result(data, pos_best_g, err_best_g)

    fcs_len = len(data.FCSj)
    scs_len = len(data.SCSj)

    fcs = pos_best_g[:fcs_len]
    scs = pos_best_g[fcs_len:fcs_len + scs_len]

    return fcs, scs


# data = GetData()
# # res = process_data(data)
# demand_pred_2019, demand_pred_2020 = demand_history(data)
# # res = process_data(data, demand_pred_2019)
# data.Di[2019] = demand_pred_2019
# PSO(data, 15, 100)