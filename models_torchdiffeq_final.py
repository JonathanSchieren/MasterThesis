#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# imports
import os
import sys
import torch

# needed?
torch.cuda.empty_cache()


# In[ ]:


# paths local and colab
path_vars_lamah_local = #
path_qobs_lamah_local = #

path_vars_lamah_colab = #
path_qobs_lamah_colab = #

path_vars_eobs_local = #
path_vars_eobs_colab = #

# attributes local and colab
path_catchment_attrs_colab = #
path_catchment_attrs_local = #

path_calibrated_parameters_v1 = #
path_calibrated_parameters_v2 = #

# paths to save the outputs of the different models local and colab
path_conceptual_v1_colab = #
path_conceptual_v2_colab = #

path_conceptual_v1_local = #
path_conceptual_v2_local = #

# paths to save the outputs of the different HYBRID models local and colab
path_hybrid_v1_colab = #
path_hybrid_v2_colab = #

path_hybrid_v1_local = #
path_hybrid_v2_local = #


# paths to save the outputs of the different LSTM models local and colab
path_lstm_v1_colab = #
path_lstm_v2_colab = #

path_lstm_v1_local = #
path_lstm_v2_local = #


# paths to save the outputs of the different LSTM Large models local and colab
path_lstm_v1_large_colab = #
path_lstm_v2_large_colab = #

path_lstm_v1_large_local = #
path_lstm_v2_large_local = #


# In[ ]:


# user input
"""
Set environment to either "colab" or "local"
"""
environment = "colab"
device = torch.device("cuda"
                      if torch.cuda.is_available() else "cpu") # "cuda:0" instead of "cuda"???
print(device)

# do not include 'qobs'
variables_lamah = ['tmax', 'tmean', 'tmin', 'tdpmax', 'tdpmean', 'tdpmin', 'windu', 'windv', 'albedo', 'swe', 'sradmean', 'tradmean', 'et', 'prcp']
variables_eobs = ['prcp', 'tmean', 'tmin', 'tmax', 'seapress', 'humidity', 'windspeed', 'srad', 'albedo', 'pet', 'daylength', 'pev'] # also has "DOY"

# variables for LSTM model
variables_lamah_lstm_small = ['tmean', 'et', 'prcp']
variables_eobs_lstm_small = ['prcp', 'tmean', 'pet'] # also has "DOY"

# variables fot LSTM large
variables_lamah_lstm_large = ['tmean', 'et', 'prcp', 'sradmean', 'tmin', 'tmax']
variables_eobs_lstm_large = ['prcp', 'tmean', 'pet', 'srad', 'tmin', 'tmax']


print("Hydrological year from October 1 to September 30")

# added for hydrological year shift
reference = "1981-10-01" #

start_cal = "1981-10-01" 
stop_cal = "2001-10-01" 
start_train = "1981-10-01"
stop_train = "2001-10-01" 
start_val = "2001-10-01"
stop_val = "2005-10-01" 
start_test = "2005-10-01"
stop_test = "2017-10-01" 


# In[ ]:


# set environment
if environment=='colab':
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    # path to scripts in drive
    sys.path.append('/content/drive/MyDrive/msc_thesis/python/scripts/')

    # installations for colab
    get_ipython().system('pip install torchdiffeq')

if environment=='local':
    """
    Use conda environment "pytorch"
    """
    ### check if this works
    sys.path.append('/Users/jonathanschieren/Desktop/msc_thesis/python/scripts')


# In[ ]:


# imports ### see what can go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import time
import copy
import random
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchdiffeq
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize # for calibration of exp-hydro might replace this with torch.optim
from pathlib import Path
from typing import Tuple, List
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# set seeeds 
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


# In[ ]:


# import scripts
from loss_functions import *
from data_processing_new import *


# In[ ]:


# create array of catchment IDs with no gaps in runoff timeseries and also covering the whole timespan from 1981 to 2017
if environment=='colab':
    catchment_attributes = pd.read_csv(path_catchment_attrs_colab + "Catchment_attributes_w_clusters.csv", delimiter=",") # load the modified versions that contains the clustering

if environment=='local':
    catchment_attributes = pd.read_csv(path_catchment_attrs_local + "Catchment_attributes_w_clusters.csv", delimiter=",") # load the modified versions that contains the clustering

IDs = catchment_attributes['ID'].values


# In[ ]:


# create array of catchment IDs (this is for colab, adapt to local if needed)
if environment=='colab':
    catchment_attributes_red = pd.read_csv(path_catchment_attrs_colab + "Catchment_attributes_nogaps_fullperiod.csv", delimiter=",") # load the modified versions that contains the clustering

if environment=='local':
    catchment_attributes_red = pd.read_csv(path_catchment_attrs_local + "Catchment_attributes_nogaps_fullperiod.csv", delimiter=",") # load the modified versions that contains the clustering

IDs_red = catchment_attributes_red['ID'].values


# Representative catchments per cluster:
# 
# ConceptualV1 (E-OBS data):
# 
# 0:          241 \
# 1:          215 \
# 2:          581 \
# 3:           21 \
# 4:          277 \
# 5:          797 \
# 7:          432 \
# 
# ConceptualV2 (LamaH-CE data):
# 
# 0:          334 \
# 1:          743 \
# 2:          439 \
# 3:           24 \
# 4:          330 \
# 5:           75 \
# 7:          383 \

# # 1. Classes

# ## 1.1 EXP-HYDRO
# Two versions:
# - ConceptualV1: input (p, pet, temp) almost the same as original EXP-HYDRO
# - ConceptualV2: input (p, **et**, temp) also similar to EXP-HYDRO but takes directly et as input

# ### Conceptual V1Cal (E-OBS) [EXP-HYDRO Calibration, Euler, RK4]

# In[ ]:


# ConceptualV1Cal: Calibration version of single-run model above
# with PSO
class ConceptualV1CalEULER():
    def __init__(self, p, pet, temp, qobs_mm, initial_storages): # this does not take parameters as input anymore!
        self.P = p
        self.PET = pet
        self.T = temp

        self.qobs_mm = qobs_mm
        self.timespan = self.P.shape[0]
        self.storage = initial_storages

        self.qsim = torch.zeros(self.timespan)
        self.melt = torch.zeros(self.timespan)

        # this is needed to create an empy array for ET which will be calculated based on PET
        self.et = torch.zeros(self.timespan)  # Simulated ET (mm/day)

        # time
        self.t = torch.arange(0, self.timespan, 1, dtype=torch.float32)

    def waterbalance(self, t, s):

        # convert time
        t = t.to(torch.long)

        # load input data for current time step
        p = self.P[t]
        temp = self.T[t]
        pet = self.PET[t]

        # partitioning
        [ps, pr] = self.rainsnowpartition(p, temp)

        # snowbucket
        m = self.snowbucket(s[0], self.T[t])

        # soil bucket ### this is different compared to V2
        [et, qsub, qsurf] = self.soilbucket(s[1], pet)

        # water balance equations
        ds1 = ps - m
        ds2 = pr + m - et - qsub - qsurf
        ds = torch.tensor([ds1, ds2])

        # write discharge into output variable
        self.qsim[t] = qsub + qsurf

        # this is also differentn compared to V2
        self.et[t] = et
        self.melt[t] = m
        return ds
    """@staticmethod"""
    def rainsnowpartition(self, p, temp):

        mint = self.static_parameters[4]

        if temp < mint:
            psnow = p
            prain = 0
        else:
            psnow = 0
            prain = p

        return psnow, prain
    """@staticmethod"""
    def snowbucket(self, s, temp):

        maxt = self.static_parameters[5]
        ddf = self.static_parameters[3]

        if temp > maxt:
            if s > 0:
                melt = min(s, ddf * (temp - maxt))
            else:
                melt = 0
        else:
            melt = 0

        """
        melt = torch.where(temp > maxt, torch.where(s > 0, torch.min(s, ddf * (temp - maxt)), torch.zeros_like(s)), torch.zeros_like(s))
        """
        return melt
    """@staticmethod"""
    def soilbucket(self, s, pet):

        smax = self.static_parameters[1]
        qmax = self.static_parameters[2]
        f = self.static_parameters[0]


        if s < 0:
            et = 0
            qsub = 0
            qsurf = 0
        elif s > smax:
            et = pet
            qsub = qmax
            qsurf = s - smax
        else:
            qsub = qmax * torch.exp(-f * (smax - s))
            qsurf = 0
            et = pet * (s / smax) ### this is ofc also different

        """
        qsub = torch.where(s < 0, torch.zeros_like(s), torch.where(s > smax, qmax, qmax * torch.exp(-f * (smax - s))))
        qsurf = torch.where(s > smax, s - smax, torch.zeros_like(s))
        """
        return [et, qsub, qsurf] # this is ofc also new

    def simulate(self):
        torchdiffeq.odeint(self.waterbalance, self.storage, self.t, method='euler')
        return self.qsim

    def objective_function_cal(self, params):
        self.static_parameters = torch.from_numpy(params)
        qsim = self.simulate()
        qsim_cal = self.qsim[cal_period]
        qobs_cal = self.qobs_mm[cal_period]

        # objective function option 1, sum of squares
        """
        diff = qsim_cal - qobs_cal
        obj_value = torch.sum(diff ** 2) ### this is the loss function REPLACE if needed
        """

        # objective function option 2 NSE WE CAN use only the part after the 1 and minimze it, this way the optimal value is ZERO!!
        obj_value = torch.sum((qsim_cal - qobs_cal)**2) / torch.sum((qobs_cal - torch.mean(qobs_cal))**2)  # so the potimal value here is 0
        # actual NSE looks like this:
        # 1 - torch.sum((qsim_cal - qobs_cal)**2) / torch.sum((qobs_cal - torch.mean(qobs_cal))**2)


        return obj_value.item()

    def pso_calibration(self, cal_period, swarm_size=10, max_iterations=100, c1=2.0, c2=2.0, w=0.9):

        num_parameters = 6

        parameters_min = np.array([0, 100, 10, 0, -3, 0])  ### 30 --> 100 # Minimum parameter values
        parameters_max = np.array([0.1, 1500, 50, 5, 0, 3])   # Maximum parameter values

        def objective_function(params):
            return self.objective_function_cal(params)

        # Initialize particle positions and velocities
        particles = np.random.uniform(parameters_min, parameters_max, (swarm_size, num_parameters))
        velocities = np.zeros_like(particles)

        # Initialize best positions and best objective values for particles and swarm
        best_positions = particles.copy()
        best_objectives = np.full(swarm_size, np.inf)
        best_swarm_position = np.zeros(num_parameters)
        best_swarm_objective = np.inf

        # Perform PSO iterations
        iteration = 0
        while iteration < max_iterations:
            for i in range(swarm_size):
                # Update particle velocity
                velocities[i] = (w * velocities[i] +
                                 c1 * np.random.random() * (best_positions[i] - particles[i]) +
                                 c2 * np.random.random() * (best_swarm_position - particles[i]))

                # Update particle position
                particles[i] += velocities[i]

                # Clip particle position within bounds
                particles[i] = np.clip(particles[i], parameters_min, parameters_max)

                # Evaluate objective function
                objective_value = objective_function(particles[i]) ### here we call the "objective_function" with the current parameters, this calls "objective_function_cal" which SETS the current parameters as STATIC_PARAMETERS and then calls "simulate()"

                # Update personal best position and objective
                if objective_value < best_objectives[i]:
                    best_objectives[i] = objective_value
                    best_positions[i] = particles[i]

                # Update swarm best position and objective
                if objective_value < best_swarm_objective:
                    best_swarm_objective = objective_value
                    best_swarm_position = particles[i]

                # Print current iteration and parameter values
                print(f"Iteration: {iteration}, Parameters: {particles[i]}")

            iteration += 1

        # Return the best swarm position (calibrated parameters)
        return best_swarm_position

    def calibrate_parameters(self, cal_period, max_iterations=100):
        calibrated_params = self.pso_calibration(cal_period, max_iterations=max_iterations)
        self.static_parameters = torch.from_numpy(calibrated_params) ### so here the calibrated parameters are assigned as static parameters, right?
        return calibrated_params


# In[ ]:


# ConceptualV1Cal: Calibration version of single-run model above
# with PSO
class ConceptualV1CalRK4():
    def __init__(self, p, pet, temp, qobs_mm, initial_storages): # this does not take parameters as input anymore!
        self.P = p
        self.PET = pet
        self.T = temp

        self.qobs_mm = qobs_mm
        self.timespan = self.P.shape[0]
        self.storage = initial_storages

        self.qsim = torch.zeros(self.timespan)
        self.melt = torch.zeros(self.timespan)

        # this is needed to create an empy array for ET which will be calculated based on PET
        self.et = torch.zeros(self.timespan)  # Simulated ET (mm/day)

        # time
        self.t = torch.arange(0, self.timespan, 1, dtype=torch.float32)

    def waterbalance(self, t, s):

        # convert time
        t = t.to(torch.long)

        # load input data for current time step
        p = self.P[t]
        temp = self.T[t]
        pet = self.PET[t]

        # partitioning
        [ps, pr] = self.rainsnowpartition(p, temp)

        # snowbucket
        m = self.snowbucket(s[0], self.T[t])

        # soil bucket ### this is different compared to V2
        [et, qsub, qsurf] = self.soilbucket(s[1], pet)

        # water balance equations
        ds1 = ps - m
        ds2 = pr + m - et - qsub - qsurf
        ds = torch.tensor([ds1, ds2])

        # write discharge into output variable
        self.qsim[t] = qsub + qsurf

        # this is also differentn compared to V2
        self.et[t] = et
        self.melt[t] = m
        return ds
    """@staticmethod"""
    def rainsnowpartition(self, p, temp):

        mint = self.static_parameters[4]

        if temp < mint:
            psnow = p
            prain = 0
        else:
            psnow = 0
            prain = p

        return psnow, prain
    """@staticmethod"""
    def snowbucket(self, s, temp):

        maxt = self.static_parameters[5]
        ddf = self.static_parameters[3]

        if temp > maxt:
            if s > 0:
                melt = min(s, ddf * (temp - maxt))
            else:
                melt = 0
        else:
            melt = 0

        """
        melt = torch.where(temp > maxt, torch.where(s > 0, torch.min(s, ddf * (temp - maxt)), torch.zeros_like(s)), torch.zeros_like(s))
        """
        return melt
    """@staticmethod"""
    def soilbucket(self, s, pet):

        smax = self.static_parameters[1]
        qmax = self.static_parameters[2]
        f = self.static_parameters[0]


        if s < 0:
            et = 0
            qsub = 0
            qsurf = 0
        elif s > smax:
            et = pet
            qsub = qmax
            qsurf = s - smax
        else:
            qsub = qmax * torch.exp(-f * (smax - s))
            qsurf = 0
            et = pet * (s / smax) ### this is ofc also different

        """
        qsub = torch.where(s < 0, torch.zeros_like(s), torch.where(s > smax, qmax, qmax * torch.exp(-f * (smax - s))))
        qsurf = torch.where(s > smax, s - smax, torch.zeros_like(s))
        """
        return [et, qsub, qsurf] # this is ofc also new

    def simulate(self):
        torchdiffeq.odeint(self.waterbalance, self.storage, self.t, method='rk4')
        return self.qsim

    def objective_function_cal(self, params):
        self.static_parameters = torch.from_numpy(params)
        qsim = self.simulate()
        qsim_cal = self.qsim[cal_period]
        qobs_cal = self.qobs_mm[cal_period]

        # objective function option 1, sum of squares
        """
        diff = qsim_cal - qobs_cal
        obj_value = torch.sum(diff ** 2) ### this is the loss function REPLACE if needed
        """

        # objective function option 2 NSE WE CAN use only the part after the 1 and minimze it, this way the optimal value is ZERO!!
        obj_value = torch.sum((qsim_cal - qobs_cal)**2) / torch.sum((qobs_cal - torch.mean(qobs_cal))**2)  # so the potimal value here is 0
        # actual NSE looks like this:
        # 1 - torch.sum((qsim_cal - qobs_cal)**2) / torch.sum((qobs_cal - torch.mean(qobs_cal))**2)


        return obj_value.item()

    def pso_calibration(self, cal_period, swarm_size=10, max_iterations=100, c1=2.0, c2=2.0, w=0.9):

        num_parameters = 6

        parameters_min = np.array([0, 100, 10, 0, -3, 0])  ### 30 --> 100 # Minimum parameter values
        parameters_max = np.array([0.1, 1500, 50, 5, 0, 3])   # Maximum parameter values

        def objective_function(params):
            return self.objective_function_cal(params)

        # Initialize particle positions and velocities
        particles = np.random.uniform(parameters_min, parameters_max, (swarm_size, num_parameters))
        velocities = np.zeros_like(particles)

        # Initialize best positions and best objective values for particles and swarm
        best_positions = particles.copy()
        best_objectives = np.full(swarm_size, np.inf)
        best_swarm_position = np.zeros(num_parameters)
        best_swarm_objective = np.inf

        # Perform PSO iterations
        iteration = 0
        while iteration < max_iterations:
            for i in range(swarm_size):
                # Update particle velocity
                velocities[i] = (w * velocities[i] +
                                 c1 * np.random.random() * (best_positions[i] - particles[i]) +
                                 c2 * np.random.random() * (best_swarm_position - particles[i]))

                # Update particle position
                particles[i] += velocities[i]

                # Clip particle position within bounds
                particles[i] = np.clip(particles[i], parameters_min, parameters_max)

                # Evaluate objective function
                objective_value = objective_function(particles[i]) ### here we call the "objective_function" with the current parameters, this calls "objective_function_cal" which SETS the current parameters as STATIC_PARAMETERS and then calls "simulate()"

                # Update personal best position and objective
                if objective_value < best_objectives[i]:
                    best_objectives[i] = objective_value
                    best_positions[i] = particles[i]

                # Update swarm best position and objective
                if objective_value < best_swarm_objective:
                    best_swarm_objective = objective_value
                    best_swarm_position = particles[i]

                # Print current iteration and parameter values
                print(f"Iteration: {iteration}, Parameters: {particles[i]}")

            iteration += 1

        # Return the best swarm position (calibrated parameters)
        return best_swarm_position

    def calibrate_parameters(self, cal_period, max_iterations=100):
        calibrated_params = self.pso_calibration(cal_period, max_iterations=max_iterations)
        self.static_parameters = torch.from_numpy(calibrated_params) ### so here the calibrated parameters are assigned as static parameters, right?
        return calibrated_params


# ### Conceptual V2 (LamaH-CE) [EXP-HYDRO Single-run, Euler, RK4]

# In[ ]:


# exp-hydro single run  with torchdiffeq
class ConceptualV2EULER():
    """
    This class creates a model that is similar to EXP-Hydro by Patil et al. (2014).
    Instead of calculating ET based on PET, the model receives ET observations as
    input. Also the solver from "hydroutils" by Patil is replaced with an ODE solver
    form torchdiffeq (Chen, 2018).

    The model is currently only working with fixed time-step methods in the ODE solver
    such as 'euler' and 'rk4'.
    """
    def __init__(self, p, et, temp, qobs_mm, static_parameters, initial_storages): ### add qobs_mm as input
        self.P = p
        self.ET = et
        self.T = temp

        ### add qobs for objective function
        self.qobs_mm = qobs_mm

        self.timespan = self.P.shape[0]

        ### self.storage = torch.tensor([10.0, 10.0])
        self.storage = initial_storages

        self.qsim = torch.zeros(self.timespan)
        self.melt = torch.zeros(self.timespan)

        # just to check the order
        # static_parameters = [f, smax, qmax, ddf, mint, maxt]

        # this is a bit
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]
        self.t = torch.arange(0, self.timespan, 1, dtype=torch.float32)

        ###
        #print(self.t)

    def waterbalance(self, t, s):

        """CHANGE: in terms of coding efficiency no self.T[t] should be used when it is already assigned to temp I think"""
        t = t.to(torch.long)
        p = self.P[t]
        temp = self.T[t]
        et = self.ET[t]
        [ps, pr] = self.rainsnowpartition(p, temp)
        m = self.snowbucket(s[0], temp) # replaced self.T[t] with temp
        [qsub, qsurf] = self.soilbucket(s[1])
        ds1 = ps - m
        ds2 = pr + m - et - qsub - qsurf # replace self.ET[t] with et
        ds = torch.tensor([ds1, ds2])
        self.qsim[t] = qsub + qsurf
        self.melt[t] = m
        return ds

    """@staticmethod"""
    def rainsnowpartition(self, p, temp):
        ### mint = static_parameters[4]
        mint = self.mint

        if temp < mint: ###
            psnow = p
            prain = 0
        else:
            psnow = 0
            prain = p
        return [psnow, prain]

    """@staticmethod"""
    def snowbucket(self, s, temp):
        ### maxt = static_parameters[5]
        ### ddf = static_parameters[3]

        maxt = self.maxt
        ddf = self.ddf
        if temp > maxt: ###
            if s > 0:
                melt = min(s, ddf * (temp - maxt)) ###
            else:
                melt = 0
        else:
            melt = 0
        return melt

    """@staticmethod"""
    def soilbucket(self, s):
        ### smax = static_parameters[1]
        ### qmax = static_parameters[2]
        ### f = static_parameters[0]

        smax = self.smax
        qmax = self.qmax
        f = self.f

        if s < 0:
            qsub = 0
            qsurf = 0
        elif s > smax: ###
            qsub = qmax ###
            qsurf = s - smax ###
        else:
            qsub = qmax * torch.exp(-f * (smax - s)) ###
            qsurf = 0
        return [qsub, qsurf]

    def simulate(self):
        torchdiffeq.odeint(self.waterbalance, self.storage, self.t, method='euler')
        return self.qsim

    # objective function
    def objective_fucntion(self, time_period):
        """
        Calculates the Nash-Sutcliffe Efficiency.
        time_period:    Indeces of range over which to calcualte the loss
        """

        # make prediction
        qsim = self.simulate()
        qsim = self.qsim[time_period]
        qobs = self.qobs_mm[time_period]

        mean_qobs = torch.mean(qobs)
        num = torch.sum((qobs - qsim)**2)
        den = torch.sum((qobs - mean_qobs)**2)
        nse = 1 - (num / den)

        return nse.item()


# In[ ]:


""" Final for now """
# exp-hydro single run  with torchdiffeq
class ConceptualV2RK4():
    """
    This class creates a model that is similar to EXP-Hydro by Patil et al. (2014).
    Instead of calculating ET based on PET, the model receives ET observations as
    input. Also the solver from "hydroutils" by Patil is replaced with an ODE solver
    form torchdiffeq (Chen, 2018).

    The model is currently only working with fixed time-step methods in the ODE solver
    such as 'euler' and 'rk4'.
    """
    def __init__(self, p, et, temp, qobs_mm, static_parameters, initial_storages): ### add qobs_mm as input
        self.P = p
        self.ET = et
        self.T = temp

        ### add qobs for objective function
        self.qobs_mm = qobs_mm

        self.timespan = self.P.shape[0]

        ### self.storage = torch.tensor([10.0, 10.0])
        self.storage = initial_storages

        self.qsim = torch.zeros(self.timespan)
        self.melt = torch.zeros(self.timespan)

        # just to check the order
        # static_parameters = [f, smax, qmax, ddf, mint, maxt]

        # this is a bit
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]
        self.t = torch.arange(0, self.timespan, 1, dtype=torch.float32)

        ###
        #print(self.t)

    def waterbalance(self, t, s):

        """CHANGE: in terms of coding efficiency no self.T[t] should be used when it is already assigned to temp I think"""
        t = t.to(torch.long)
        p = self.P[t]
        temp = self.T[t]
        et = self.ET[t]
        [ps, pr] = self.rainsnowpartition(p, temp)
        m = self.snowbucket(s[0], temp) # replaced self.T[t] with temp
        [qsub, qsurf] = self.soilbucket(s[1])
        ds1 = ps - m
        ds2 = pr + m - et - qsub - qsurf # replace self.ET[t] with et
        ds = torch.tensor([ds1, ds2])
        self.qsim[t] = qsub + qsurf
        self.melt[t] = m
        return ds

    """@staticmethod"""
    def rainsnowpartition(self, p, temp):
        ### mint = static_parameters[4]
        mint = self.mint

        if temp < mint: ###
            psnow = p
            prain = 0
        else:
            psnow = 0
            prain = p
        return [psnow, prain]

    """@staticmethod"""
    def snowbucket(self, s, temp):
        ### maxt = static_parameters[5]
        ### ddf = static_parameters[3]

        maxt = self.maxt
        ddf = self.ddf
        if temp > maxt: ###
            if s > 0:
                melt = min(s, ddf * (temp - maxt)) ###
            else:
                melt = 0
        else:
            melt = 0
        return melt

    """@staticmethod"""
    def soilbucket(self, s):
        ### smax = static_parameters[1]
        ### qmax = static_parameters[2]
        ### f = static_parameters[0]

        smax = self.smax
        qmax = self.qmax
        f = self.f

        if s < 0:
            qsub = 0
            qsurf = 0
        elif s > smax: ###
            qsub = qmax ###
            qsurf = s - smax ###
        else:
            qsub = qmax * torch.exp(-f * (smax - s)) ###
            qsurf = 0
        return [qsub, qsurf]

    def simulate(self):
        torchdiffeq.odeint(self.waterbalance, self.storage, self.t, method='rk4')
        return self.qsim

    # objective function
    def objective_fucntion(self, time_period):
        """
        Calculates the Nash-Sutcliffe Efficiency.
        time_period:    Indeces of range over which to calcualte the loss
        """

        # make prediction
        qsim = self.simulate()
        qsim = self.qsim[time_period]
        qobs = self.qobs_mm[time_period]

        mean_qobs = torch.mean(qobs)
        num = torch.sum((qobs - qsim)**2)
        den = torch.sum((qobs - mean_qobs)**2)
        nse = 1 - (num / den)

        return nse.item()


# ###  Conceptual V2Cal (LamaH-CE) [EXP-HYDRO Calibration, Euler, RK4]
# - This version now has a particle swarm optimization method.

# In[ ]:


# with PSO
class ConceptualV2CalEULER():
    def __init__(self, p, et, temp, qobs_mm, initial_storages): # this does not take static_parameters as input anymore
        self.P = p
        self.ET = et
        self.T = temp
        self.qobs_mm = qobs_mm
        self.timespan = self.P.shape[0]
        self.storage = initial_storages

        self.qsim = torch.zeros(self.timespan)
        self.melt = torch.zeros(self.timespan)


        self.t = torch.arange(0, self.timespan, 1, dtype=torch.float32)

    def waterbalance(self, t, s):
        t = t.to(torch.long)
        p = self.P[t]
        temp = self.T[t]
        et = self.ET[t]
        [ps, pr] = self.rainsnowpartition(p, temp)
        m = self.snowbucket(s[0], temp) # replace self.T[t] with temp
        [qsub, qsurf] = self.soilbucket(s[1])
        ds1 = ps - m
        ds2 = pr + m - et - qsub - qsurf # replace self.ET[t] with et
        ds = torch.tensor([ds1, ds2])
        self.qsim[t] = qsub + qsurf
        self.melt[t] = m
        return ds

    def rainsnowpartition(self, p, temp):
        mint = self.static_parameters[4]

        if temp < mint:
            psnow = p
            prain = 0
        else:
            psnow = 0
            prain = p


        """why did I do this?"""
        ### psnow = torch.where(temp < mint, p, torch.zeros_like(p))
        ### prain = torch.where(temp >= mint, p, torch.zeros_like(p))
        return psnow, prain

    def snowbucket(self, s, temp):
        maxt = self.static_parameters[5]
        ddf = self.static_parameters[3]

        if temp > maxt:
            if s > 0:
                melt = min(s, ddf * (temp - maxt))
            else:
                melt = 0
        else:
            melt = 0

        """" why? """
        ### melt = torch.where(temp > maxt, torch.where(s > 0, torch.min(s, ddf * (temp - maxt)), torch.zeros_like(s)), torch.zeros_like(s))
        return melt

    def soilbucket(self, s):
        smax = self.static_parameters[1]
        qmax = self.static_parameters[2]
        f = self.static_parameters[0]

        if s < 0:
            qsub = 0
            qsurf = 0
        elif s > smax: ###
            qsub = qmax ###
            qsurf = s - smax ###
        else:
            qsub = qmax * torch.exp(-f * (smax - s)) ###
            qsurf = 0


        """ why? """
        ### qsub = torch.where(s < 0, torch.zeros_like(s), torch.where(s > smax, qmax, qmax * torch.exp(-f * (smax - s))))
        ### qsurf = torch.where(s > smax, s - smax, torch.zeros_like(s))
        return [qsub, qsurf] # square brackets or not?

    def simulate(self):
        torchdiffeq.odeint(self.waterbalance, self.storage, self.t, method='euler')
        return self.qsim

    def objective_function_cal(self, params):
        self.static_parameters = torch.from_numpy(params)
        qsim = self.simulate()
        qsim_cal = self.qsim[cal_period]
        qobs_cal = self.qobs_mm[cal_period]

        # objective function version 1 sum of squares
        """
        diff = qsim_cal - qobs_cal
        obj_value = torch.sum(diff ** 2)
        """

        # objective function option 2 NSE WE CAN use only the part after the 1 and minimze it, this way the optimal value is ZERO!!
        obj_value = torch.sum((qsim_cal - qobs_cal)**2) / torch.sum((qobs_cal - torch.mean(qobs_cal))**2)  # so the potimal value here is 0
        # actual NSE looks like this:
        # 1 - torch.sum((qsim_cal - qobs_cal)**2) / torch.sum((qobs_cal - torch.mean(qobs_cal))**2)

        return obj_value.item()

    def pso_calibration(self, cal_period, swarm_size=10, max_iterations=100, c1=2.0, c2=2.0, w=0.9):
        num_parameters = 6
        parameters_min = np.array([0, 100, 10, 0, -3, 0])  ### 30 --> 100 # Minimum parameter values
        parameters_max = np.array([0.1, 1500, 50, 5, 0, 3])   # Maximum parameter values

        def objective_function(params):
            return self.objective_function_cal(params)

        # Initialize particle positions and velocities
        particles = np.random.uniform(parameters_min, parameters_max, (swarm_size, num_parameters))
        velocities = np.zeros_like(particles)

        # Initialize best positions and best objective values for particles and swarm
        best_positions = particles.copy()
        best_objectives = np.full(swarm_size, np.inf)
        best_swarm_position = np.zeros(num_parameters)
        best_swarm_objective = np.inf

        # Perform PSO iterations
        iteration = 0
        while iteration < max_iterations:
            for i in range(swarm_size):
                # Update particle velocity
                velocities[i] = (w * velocities[i] +
                                 c1 * np.random.random() * (best_positions[i] - particles[i]) +
                                 c2 * np.random.random() * (best_swarm_position - particles[i]))

                # Update particle position
                particles[i] += velocities[i]

                # Clip particle position within bounds
                particles[i] = np.clip(particles[i], parameters_min, parameters_max)

                # Evaluate objective function
                objective_value = objective_function(particles[i])

                # Update personal best position and objective
                if objective_value < best_objectives[i]:
                    best_objectives[i] = objective_value
                    best_positions[i] = particles[i]

                # Update swarm best position and objective
                if objective_value < best_swarm_objective:
                    best_swarm_objective = objective_value
                    best_swarm_position = particles[i]

                # Print current iteration and parameter values
                print(f"Iteration: {iteration}, Parameters: {particles[i]}")

            iteration += 1

        # Return the best swarm position (calibrated parameters)
        return best_swarm_position

    def calibrate_parameters(self, cal_period, max_iterations=100):
        calibrated_params = self.pso_calibration(cal_period, max_iterations=max_iterations)
        self.static_parameters = torch.from_numpy(calibrated_params)
        return calibrated_params


# In[ ]:


# with PSO
class ConceptualV2CalRK4():
    def __init__(self, p, et, temp, qobs_mm, initial_storages): # this does not take static_parameters as input anymore
        self.P = p
        self.ET = et
        self.T = temp
        self.qobs_mm = qobs_mm
        self.timespan = self.P.shape[0]
        self.storage = initial_storages

        self.qsim = torch.zeros(self.timespan)
        self.melt = torch.zeros(self.timespan)


        self.t = torch.arange(0, self.timespan, 1, dtype=torch.float32)

    def waterbalance(self, t, s):
        t = t.to(torch.long)
        p = self.P[t]
        temp = self.T[t]
        et = self.ET[t]
        [ps, pr] = self.rainsnowpartition(p, temp)
        m = self.snowbucket(s[0], temp) # replace self.T[t] with temp
        [qsub, qsurf] = self.soilbucket(s[1])
        ds1 = ps - m
        ds2 = pr + m - et - qsub - qsurf # replace self.ET[t] with et
        ds = torch.tensor([ds1, ds2])
        self.qsim[t] = qsub + qsurf
        self.melt[t] = m
        return ds

    def rainsnowpartition(self, p, temp):
        mint = self.static_parameters[4]

        if temp < mint:
            psnow = p
            prain = 0
        else:
            psnow = 0
            prain = p


        """why did I do this?"""
        ### psnow = torch.where(temp < mint, p, torch.zeros_like(p))
        ### prain = torch.where(temp >= mint, p, torch.zeros_like(p))
        return psnow, prain

    def snowbucket(self, s, temp):
        maxt = self.static_parameters[5]
        ddf = self.static_parameters[3]

        if temp > maxt:
            if s > 0:
                melt = min(s, ddf * (temp - maxt))
            else:
                melt = 0
        else:
            melt = 0

        """" why? """
        ### melt = torch.where(temp > maxt, torch.where(s > 0, torch.min(s, ddf * (temp - maxt)), torch.zeros_like(s)), torch.zeros_like(s))
        return melt

    def soilbucket(self, s):
        smax = self.static_parameters[1]
        qmax = self.static_parameters[2]
        f = self.static_parameters[0]

        if s < 0:
            qsub = 0
            qsurf = 0
        elif s > smax: ###
            qsub = qmax ###
            qsurf = s - smax ###
        else:
            qsub = qmax * torch.exp(-f * (smax - s)) ###
            qsurf = 0


        """ why? """
        ### qsub = torch.where(s < 0, torch.zeros_like(s), torch.where(s > smax, qmax, qmax * torch.exp(-f * (smax - s))))
        ### qsurf = torch.where(s > smax, s - smax, torch.zeros_like(s))
        return [qsub, qsurf] # square brackets or not?

    def simulate(self):
        torchdiffeq.odeint(self.waterbalance, self.storage, self.t, method='rk4')
        return self.qsim

    def objective_function_cal(self, params):
        self.static_parameters = torch.from_numpy(params)
        qsim = self.simulate()
        qsim_cal = self.qsim[cal_period]
        qobs_cal = self.qobs_mm[cal_period]

        # objective function version 1 sum of squares
        """
        diff = qsim_cal - qobs_cal
        obj_value = torch.sum(diff ** 2)
        """

        # objective function option 2 NSE WE CAN use only the part after the 1 and minimze it, this way the optimal value is ZERO!!
        obj_value = torch.sum((qsim_cal - qobs_cal)**2) / torch.sum((qobs_cal - torch.mean(qobs_cal))**2)  # so the potimal value here is 0
        # actual NSE looks like this:
        # 1 - torch.sum((qsim_cal - qobs_cal)**2) / torch.sum((qobs_cal - torch.mean(qobs_cal))**2)

        return obj_value.item()

    def pso_calibration(self, cal_period, swarm_size=10, max_iterations=100, c1=2.0, c2=2.0, w=0.9):
        num_parameters = 6
        parameters_min = np.array([0, 100, 10, 0, -3, 0])  ### 30 --> 100 # Minimum parameter values
        parameters_max = np.array([0.1, 1500, 50, 5, 0, 3])   # Maximum parameter values

        def objective_function(params):
            return self.objective_function_cal(params)

        # Initialize particle positions and velocities
        particles = np.random.uniform(parameters_min, parameters_max, (swarm_size, num_parameters))
        velocities = np.zeros_like(particles)

        # Initialize best positions and best objective values for particles and swarm
        best_positions = particles.copy()
        best_objectives = np.full(swarm_size, np.inf)
        best_swarm_position = np.zeros(num_parameters)
        best_swarm_objective = np.inf

        # Perform PSO iterations
        iteration = 0
        while iteration < max_iterations:
            for i in range(swarm_size):
                # Update particle velocity
                velocities[i] = (w * velocities[i] +
                                 c1 * np.random.random() * (best_positions[i] - particles[i]) +
                                 c2 * np.random.random() * (best_swarm_position - particles[i]))

                # Update particle position
                particles[i] += velocities[i]

                # Clip particle position within bounds
                particles[i] = np.clip(particles[i], parameters_min, parameters_max)

                # Evaluate objective function
                objective_value = objective_function(particles[i])

                # Update personal best position and objective
                if objective_value < best_objectives[i]:
                    best_objectives[i] = objective_value
                    best_positions[i] = particles[i]

                # Update swarm best position and objective
                if objective_value < best_swarm_objective:
                    best_swarm_objective = objective_value
                    best_swarm_position = particles[i]

                # Print current iteration and parameter values
                print(f"Iteration: {iteration}, Parameters: {particles[i]}")

            iteration += 1

        # Return the best swarm position (calibrated parameters)
        return best_swarm_position

    def calibrate_parameters(self, cal_period, max_iterations=100):
        calibrated_params = self.pso_calibration(cal_period, max_iterations=max_iterations)
        self.static_parameters = torch.from_numpy(calibrated_params)
        return calibrated_params


# ### Exp-Hydro Conceptual but without Methods [This is only the building block for the hybrid models, not used like this]
# To be used later for the hybrid approach.
# This model version is modified in two way:
# 1. It takes et as input instead of pet
# 2. It does not have any static methods, instead it just has if else statements

# ###### Conceptual V1 Mod (not needed but keep for documentation)

# In[ ]:


# this behaves the same as the exp hydro single run but has different structure
class ExpHydroModV1():
    """
    This class creates a model that is similar to EXP-Hydro by Patil et al. (2014).
    Instead of calculating ET based on PET, the model receives ET observations as
    input. Also the solver from "hydroutils" by Patil is replaced with an ODE solver
    from torchdiffeq (Chen, 2018).

    The model is currently only working with fixed time-step methods in the ODE solver
    such as 'euler' and 'rk4'.
    """
    def __init__(self, p, et, temp, static_parameters):
        self.P = p
        self.ET = et
        self.T = temp

        self.timespan = self.P.shape[0]
        self.storage = torch.zeros(2)
        self.qsim = torch.zeros(self.timespan)
        self.melt = torch.zeros(self.timespan)

        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]
        self.t = torch.arange(0, self.timespan, 1, dtype=torch.float32)

    def waterbalance(self, t, s):
        t = t.to(torch.long)
        p = self.P[t]
        temp = self.T[t]
        et = self.ET[t]

        mint = self.mint
        maxt = self.maxt
        ddf = self.ddf
        smax = self.smax
        qmax = self.qmax
        f = self.f

        # rain snow partition
        if temp < mint:
            psnow = p
            prain = 0
        else:
            psnow = 0
            prain = p

        # snow bucket
        if temp > maxt:
            if s[0] > 0: ### correct?
                melt = min(s[0], ddf * (temp - maxt))
            else:
                melt = 0
        else:
            melt = 0

        # soil bucket
        if s[1] < 0: ### correct?
            qsub = 0
            qsurf = 0
        elif s[1] > smax: ### correct?
            qsub = qmax
            qsurf = s[1] - smax
        else:
            qsub = qmax * torch.exp(-f * (smax - s[1]))
            qsurf = 0

        ds1 = psnow - melt
        ds2 = prain + melt - et - qsub - qsurf

        ds = torch.tensor([ds1, ds2])
        self.qsim[t] = qsub + qsurf
        self.melt[t] = melt
        return ds

    def simulate(self):
        torchdiffeq.odeint(self.waterbalance, self.storage, self.t, method='euler')
        return self.qsim


# ###### Pre-train V1 Version (works!)
# This outputs psnow, prain, melt, et and qsim which we can try to use for the pre-training of the M100 models

# For the pre-training we need the following:
# 
# Note, et for LamaH might be taken from the observations instead of from the model outputs, or we output it out of the neural network and "correct" it so to say.
# 
# **We need:** \\
# prain \\
# psnow \\
# melt \\
# et \\
# qsim \\
# AND \\
# S0 & S1

# In[ ]:


# Start from Hybrid V1 but remove the NN component so that it is purely conceptual again, then add the output of the internal fluxes for the pretrain model
class ExpHydroModV1(nn.Module):
    """
    In this model the neural network is connected to both storages.
    """
    def __init__(self, p, pet, temp, static_parameters):
        super().__init__()
        self.P = p
        self.PET = pet
        self.T = temp


        # set the static parameters
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension

        # define the variables we want to output
        qsim = torch.tensor([], requires_grad=False) #
        psnow_ts = torch.tensor([], requires_grad=False)
        prain_ts = torch.tensor([], requires_grad=False)
        melt_ts = torch.tensor([], requires_grad=False)
        et_ts = torch.tensor([], requires_grad=False)
        S0_ts = torch.tensor([], requires_grad=False)
        S1_ts = torch.tensor([], requires_grad=False)

        # return also precipitation and temperature, so the inputs? just for simplicity for the pretraining
        prcp_ts = torch.tensor([], requires_grad=False)
        temp_ts = torch.tensor([], requires_grad=False)



        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)

        q0 = q0.to(device)
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim, psnow_ts, prain_ts, melt_ts, et_ts, S0_ts, S1_ts, prcp_ts, temp_ts

            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]
            pet = self.PET[t]

            # GPU
            t = t.to(device)
            p = p.to(device)
            temp = temp.to(device)
            pet = pet.to(device)
            y = y.to(device)


            """ conceptual part """
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            if temp < self.mint:
                psnow = p
                prain = torch.tensor(0, dtype=torch.float32)
            else:
                psnow = torch.tensor(0, dtype=torch.float32)
                prain = p


            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)

            # 2. melt -> in original done with snowbucket -> works
            if temp > self.maxt:
                if y[0,0] > 0:
                    melt = torch.min(y[0,0], self.ddf * (temp - self.maxt)).to(dtype=torch.float32) # try to create melt as torch tensor directly # check dtype
                else:
                    melt =  torch.tensor(0, dtype=torch.float32)
            else:
                melt =  torch.tensor(0, dtype=torch.float32)

            # GPU
            melt = melt.to(device)


            # 3. qsub qsurf -> in original done with soilbucket
            if y[0,1] < 0:
                qsub = torch.tensor(0, dtype=torch.float32)
                qsurf = torch.tensor(0, dtype=torch.float32)

                ######## added
                et = torch.tensor(0, dtype=torch.float32)

            elif y[0,1] > self.smax:
                qsub = self.qmax # might have to turn this into a tensor?
                qsurf = y[0,1] - self.smax # also turn into tensor

                ####### added
                et = pet

            else:
                qsub = self.qmax * torch.exp(-self.f * (self.smax - y[0,1]))
                qsurf = torch.tensor(0, dtype=torch.float32)

                #### added
                et = pet * (y[0,1] / self.smax)


            # qsim
            qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)


            # dS
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)
            dS1 = prain + melt - et - qsub - qsurf # water storage
            dS1 = torch.unsqueeze(dS1, 0)



            ds_conceptual  = torch.cat([dS0, dS1], axis=0)



            # now save all the arrays
            psnow_ts = torch.cat((psnow_ts, psnow.unsqueeze(0)), dim=0)
            prain_ts = torch.cat((prain_ts, prain.unsqueeze(0)), dim=0)
            melt_ts = torch.cat((melt_ts, melt.unsqueeze(0)), dim=0)
            et_ts = torch.cat((et_ts, et.unsqueeze(0)), dim=0)
            # storage 0 and 1
            y00 = y[0,0]
            y01 = y[0,1]
            S0_ts = torch.cat((S0_ts, y00.unsqueeze(0)), dim=0)
            S1_ts = torch.cat((S1_ts, y01.unsqueeze(0)), dim=0)

            # return (unchanged) inputs for later
            prcp_ts = torch.cat((prcp_ts, p.unsqueeze(0)), dim=0)
            temp_ts = torch.cat((temp_ts, temp.unsqueeze(0)), dim=0)

            return ds_conceptual

        torchdiffeq.odeint(func, q0, t_solver, method='euler') # options=dict(step_size=4)

        return qsim, psnow_ts, prain_ts, melt_ts, et_ts, S0_ts, S1_ts, prcp_ts, temp_ts


# In[ ]:


id_num = 534


# In[ ]:


# try to make a prediction on it with the trained parameters
# define initial storages
S0_hybrid_v1 = torch.tensor([0.0, 0.0])
S0_hybrid_v1 = torch.unsqueeze(S0_hybrid_v1, 0) # reshape


# load the parameters for the model
if environment == "colab":
    path = os.path.join(path_conceptual_v1_colab, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
    calibrated_parameters = np.load(path)
    # load df hybrid # IMPORTANT: full period from start train to stop test!
if environment == "colab":
    df_hybrid = load_data_new(path_vars_eobs_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_eobs, start_date=start_train, stop_date=stop_test, dataset="eobs", qobs_mm=True)

    # create tensors from the dataframe
tensors_hybrid = df_to_tensors(df_hybrid, name_suffix="hybrid")

    # Processing tensors
for name, tensor in tensors_hybrid.items():
        # Moving to GPU
    tensor = tensor.to(device)
    tensor = tensor.to(dtype=torch.float32)
    tensor.requires_grad = False  # inplace operation
    locals()[name] = tensor

    # instantiate model
test_v1 = ExpHydroModV1(prcp_hybrid, pet_hybrid, tmean_hybrid, calibrated_parameters) ### note vor V2 we need et here instead!
    # hybridmodel_v1.to(device) # put back if using GPU
qsim_test_hybrid_v1, psnow_ts, prain_ts, melt_ts, et_ts, S0_ts, S1_ts, prcp_ts, temp_ts = test_v1.forward(test_period, S0_hybrid_v1)


# In[ ]:


plt.plot(qsim_test_hybrid_v1.detach())


# In[ ]:


plt.figure(figsize=(10,2))
plt.plot(S0_ts[365:365*4].detach())


# In[ ]:


plt.figure(figsize=(10,2))
plt.plot(S1_ts[365:365*4].detach())


# In[ ]:


plt.plot(melt_ts.detach())


# In[ ]:


# save temp
path_euler_training = os.path.join(path_analysis, "S1_conceptual_test_534.csv")
np.savetxt(path_euler_training, S1_ts, delimiter=",")


# ###### Add Pre-train V2 Version 🌵
# 
# 

# ## 1.2 Hybrid Models
# There are two different versions. The first version is a very straight-forward combination of the Neural ODE model and EXP-HYDRO. BUT it does not work in the backward pass, there is an issue with the tracking of gradients in Autograd.
# 
# V1 (E-OBS) and V2 (LamaH-CE) are connected to both storages.
# 
# V1 Single (E-OBS) and V2 Single (LamaH-CE) are connected to one storage.
# 
# V3 Large takes additional input forcings from the same dataset and V3 Duo can take additional forcings from another dataset. (To Do: **Check if they work**)
# 
# V4 is similar to M100 by Hoege

# ### Hybrid Model - V1 and V2
# Neural network connects to both storages.
# 
# To Do:
# - Check if we want to add extra layer(s) as in the other version.

# #### Hybrid V1 (E-OBS) [Euler, RK4]
# 

# In[ ]:


# Hybrid V1 with Euler
""" Final for now """
class Hybrid_V1(nn.Module):
    """
    In this model the neural network is connected to both storages.
    """
    def __init__(self, p, pet, temp, static_parameters):
        super().__init__()
        self.P = p
        self.PET = pet
        self.T = temp

        self.layer1 = nn.Linear(5, 16) # input
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(16, 16)
        self.activation1a = nn.LeakyReLU()
        self.layer2 = nn.Linear(16, 2)

        nn.init.normal_(self.layer1.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        ###
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        ###
        nn.init.normal_(self.layer2.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)

        # set the static parameters
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]

    def forward(self, t, q0):

        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension
        qsim = torch.tensor([], requires_grad=False) #


        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)

        q0 = q0.to(device) ###############??????????????????????????????????????????????????????????????????????????????????????????????????
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]
            pet = self.PET[t]

            # GPU
            t = t.to(device)
            p = p.to(device)
            temp = temp.to(device)
            pet = pet.to(device)
            y = y.to(device)

            """ neural network """
            input = torch.cat([
                y[0,0].view(-1, 1),
                y[0,1].view(-1, 1),
                self.P[t].view(-1, 1),
                self.PET[t].view(-1, 1),
                self.T[t].view(-1, 1),
            ], dim=1)


            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)
            ###
            output = self.layer1a(output)
            output = self.activation1a(output)
            ###
            ds_nn = self.layer2(output)

            # GPU
            output = output.to(device)


            # I thinkt this should be automatically on the GPU
            ds_nn = self.layer2(output) ### this is doubles see above and remove

            """ conceptual part """
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            if temp < self.mint:
                psnow = p
                prain = torch.tensor(0, dtype=torch.float32)
            else:
                psnow = torch.tensor(0, dtype=torch.float32)
                prain = p


            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)

            # 2. melt -> in original done with snowbucket -> works
            if temp > self.maxt:
                if y[0,0] > 0:
                    melt = torch.min(y[0,0], self.ddf * (temp - self.maxt)).to(dtype=torch.float32) # try to create melt as torch tensor directly # check dtype
                else:
                    melt =  torch.tensor(0, dtype=torch.float32)
            else:
                melt =  torch.tensor(0, dtype=torch.float32)

            # GPU
            melt = melt.to(device)


            # 3. qsub qsurf -> in original done with soilbucket
            if y[0,1] < 0:
                qsub = torch.tensor(0, dtype=torch.float32)
                qsurf = torch.tensor(0, dtype=torch.float32)

                ######## added
                et = torch.tensor(0, dtype=torch.float32)

            elif y[0,1] > self.smax:
                qsub = self.qmax # might have to turn this into a tensor?
                qsurf = y[0,1] - self.smax # also turn into tensor

                ####### added
                et = pet

            else:
                qsub = self.qmax * torch.exp(-self.f * (self.smax - y[0,1]))
                qsurf = torch.tensor(0, dtype=torch.float32)

                #### added
                et = pet * (y[0,1] / self.smax)


            ###### qsub = qsub.to(device) # put back in if using GPU, else does not matter
            ###### qsurf = qsurf.to(device)

            # qsim
            qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            ###################################################



            # GPU
            ##### qsim = qsim.to(device)

            # dS
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)
            dS1 = prain + melt - et - qsub - qsurf # water storage
            dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ##### dS0 = dS0.to(device)
            ##### dS1 = dS1.to(device)

            ds_conceptual  = torch.cat([dS0, dS1], axis=0)
            # GPU
            ##### ds_conceptual = ds_conceptual.to(device)

            ds_combined = ds_nn + ds_conceptual ###
            # GPU
            ##### ds_combined = ds_combined.to(device)


            return ds_combined

        torchdiffeq.odeint(func, q0, t_solver, method='euler') # options=dict(step_size=4)

        return qsim


# In[ ]:


# Hybrid V1 with RK4
""" Final for now """
class Hybrid_V1RK4(nn.Module):
    """
    In this model the neural network is connected to both storages. IS IT REALLY? WHY NN ONLY 1 OUPT NODE?
    """
    def __init__(self, p, pet, temp, static_parameters):
        super().__init__()
        self.P = p
        self.PET = pet
        self.T = temp

        self.layer1 = nn.Linear(5, 16)
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(16, 16)
        self.activation1a = nn.LeakyReLU()
        self.layer2 = nn.Linear(16, 2) ############################

        nn.init.normal_(self.layer1.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        ###
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        ###
        nn.init.normal_(self.layer2.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)


        # set the static parameters
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension
        qsim = torch.tensor([], requires_grad=False) #


        #################################################
        print("t", t)
        print("t_solver", t_solver)
        print("qsim", qsim)

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)

        q0 = q0.to(device)
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]
            pet = self.PET[t]

            # GPU
            t = t.to(device)
            p = p.to(device)
            temp = temp.to(device)
            pet = pet.to(device)
            y = y.to(device)

            """ neural network """
            input = torch.cat([
                y[0,0].view(-1, 1),
                y[0,1].view(-1, 1),
                self.P[t].view(-1, 1),
                self.PET[t].view(-1, 1),
                self.T[t].view(-1, 1),
            ], dim=1)


            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)
            ###
            output = self.layer1a(output)
            output = self.activation1a(output)
            ###
            ds_nn = self.layer2(output)

            # GPU
            output = output.to(device)


            # I thinkt this should be automatically on the GPU
            ds_nn = self.layer2(output)

            """ conceptual part """
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            if temp < self.mint:
                psnow = p
                prain = torch.tensor(0, dtype=torch.float32)
            else:
                psnow = torch.tensor(0, dtype=torch.float32)
                prain = p


            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)

            # 2. melt -> in original done with snowbucket -> works
            if temp > self.maxt:
                if y[0,0] > 0:
                    melt = torch.min(y[0,0], self.ddf * (temp - self.maxt)).to(dtype=torch.float32) # try to create melt as torch tensor directly # check dtype
                else:
                    melt =  torch.tensor(0, dtype=torch.float32)
            else:
                melt =  torch.tensor(0, dtype=torch.float32)

            # GPU
            melt = melt.to(device)



            ##### ADDDDDed


            # 3. qsub qsurf -> in original done with soilbucket
            if y[0,1] < 0:
                qsub = torch.tensor(0, dtype=torch.float32)
                qsurf = torch.tensor(0, dtype=torch.float32)

                ######## added
                et = torch.tensor(0, dtype=torch.float32)

            elif y[0,1] > self.smax:
                qsub = self.qmax # might have to turn this into a tensor?
                qsurf = y[0,1] - self.smax # also turn into tensor

                ####### added
                et = pet

            else:
                qsub = self.qmax * torch.exp(-self.f * (self.smax - y[0,1]))
                qsurf = torch.tensor(0, dtype=torch.float32)

                #### added
                et = pet * (y[0,1] / self.smax)


            ###### qsub = qsub.to(device) # put back in if using GPU, else does not matter
            ###### qsurf = qsurf.to(device)

            # qsim
            qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ##### qsim = qsim.to(device)

            # dS
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)
            dS1 = prain + melt - et - qsub - qsurf # water storage
            dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ##### dS0 = dS0.to(device)
            ##### dS1 = dS1.to(device)

            ds_conceptual  = torch.cat([dS0, dS1], axis=0)
            # GPU
            ##### ds_conceptual = ds_conceptual.to(device)

            ds_combined = ds_nn + ds_conceptual ###
            # GPU
            ##### ds_combined = ds_combined.to(device)

            return ds_combined

        torchdiffeq.odeint(func, q0, t_solver, method='rk4') # options=dict(step_size=4)

        return qsim


# ###### Hybrid V1 Analysis Version

# In[ ]:


### another copy with even more changes to output more variables to use for pre-training M100
#### copy of above just to test something
# Hybrid V1 with Euler
""" Final for now """
class Hybrid_V1_analysis(nn.Module):
    """
    In this model the neural network is connected to both storages.
    """
    def __init__(self, p, pet, temp, static_parameters):
        super().__init__()
        self.P = p
        self.PET = pet
        self.T = temp

        self.layer1 = nn.Linear(5, 16) # input
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(16, 16)
        self.activation1a = nn.LeakyReLU()
        self.layer2 = nn.Linear(16, 2)



        nn.init.normal_(self.layer1.weight, mean=0, std=1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        ###
        nn.init.normal_(self.layer1a.weight, mean=0, std=1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        ###
        nn.init.normal_(self.layer2.weight, mean=0, std=1) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)


        # timespan
        self.timespan = self.P.shape[0] ### this can go, right???

        # set the static parameters
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension

        ### define the variables we want to output
        qsim = torch.tensor([], requires_grad=False)
        ds_nn_S0_ts = torch.tensor([], requires_grad=False)
        ds_nn_S1_ts = torch.tensor([], requires_grad=False)
        S0_ts = torch.tensor([], requires_grad=False)
        S1_ts = torch.tensor([], requires_grad=False)
        psnow_ts = torch.tensor([], requires_grad=False)
        prain_ts = torch.tensor([], requires_grad=False)
        melt_ts = torch.tensor([], requires_grad=False)
        et_ts = torch.tensor([], requires_grad=False)
        # qsim_ts we already have in the original



        # this we actually do not want anymore
        ds_nn_TS = torch.tensor([], requires_grad=False)



        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)

        q0 = q0.to(device)
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim
            nonlocal ds_nn_TS
            nonlocal ds_nn_S0_ts, ds_nn_S1_ts, S0_ts, S1_ts, psnow_ts, prain_ts, melt_ts, et_ts


            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]
            pet = self.PET[t]

            # GPU
            t = t.to(device)
            p = p.to(device)
            temp = temp.to(device)
            pet = pet.to(device)
            y = y.to(device)

            """ neural network """
            input = torch.cat([
                y[0,0].view(-1, 1),
                y[0,1].view(-1, 1),
                self.P[t].view(-1, 1),
                self.PET[t].view(-1, 1),
                self.T[t].view(-1, 1),
            ], dim=1)


            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)
            ###
            output = self.layer1a(output)
            output = self.activation1a(output)
            ###
            ds_nn = self.layer2(output)

            # GPU
            output = output.to(device)

            """ conceptual part """
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            if temp < self.mint:
                psnow = p
                prain = torch.tensor(0, dtype=torch.float32)
            else:
                psnow = torch.tensor(0, dtype=torch.float32)
                prain = p


            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)

            # 2. melt -> in original done with snowbucket -> works
            if temp > self.maxt:
                if y[0,0] > 0:
                    melt = torch.min(y[0,0], self.ddf * (temp - self.maxt)).to(dtype=torch.float32) # try to create melt as torch tensor directly # check dtype
                else:
                    melt =  torch.tensor(0, dtype=torch.float32)
            else:
                melt =  torch.tensor(0, dtype=torch.float32)

            # GPU
            melt = melt.to(device)

            # 3. qsub qsurf -> in original done with soilbucket
            if y[0,1] < 0:
                qsub = torch.tensor(0, dtype=torch.float32)
                qsurf = torch.tensor(0, dtype=torch.float32)

                ######## added
                et = torch.tensor(0, dtype=torch.float32)

            elif y[0,1] > self.smax:
                qsub = self.qmax # might have to turn this into a tensor?
                qsurf = y[0,1] - self.smax # also turn into tensor

                ####### added
                et = pet

            else:
                qsub = self.qmax * torch.exp(-self.f * (self.smax - y[0,1]))
                qsurf = torch.tensor(0, dtype=torch.float32)

                #### added
                et = pet * (y[0,1] / self.smax)


            ###### qsub = qsub.to(device) # put back in if using GPU, else does not matter
            ###### qsurf = qsurf.to(device)

            # qsim
            qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ##### qsim = qsim.to(device)

            # dS
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)
            dS1 = prain + melt - et - qsub - qsurf # water storage
            dS1 = torch.unsqueeze(dS1, 0)


            # GPU
            ##### dS0 = dS0.to(device)
            ##### dS1 = dS1.to(device)

            ds_conceptual  = torch.cat([dS0, dS1], axis=0)
            # GPU
            ##### ds_conceptual = ds_conceptual.to(device)

            ds_combined = ds_nn + ds_conceptual ###
            # GPU
            ##### ds_combined = ds_combined.to(device)



            ### can we return the storages as time series (should be similar to qsim)
            ds_nn_TS = torch.cat((ds_nn_TS, ds_nn.unsqueeze(0)), dim=0)
            # first save the storage changes of the NN in separate variables
            ds_nn_S0 = ds_nn[0,0]
            ds_nn_S1 = ds_nn[0,1]
            # now save all the arrays
            ds_nn_S0_ts = torch.cat((ds_nn_S0_ts, ds_nn_S0.unsqueeze(0)), dim=0)
            ds_nn_S1_ts = torch.cat((ds_nn_S1_ts, ds_nn_S1.unsqueeze(0)), dim=0)
            # storage 0 and 1
            y00 = y[0,0]
            y01 = y[0,1]
            S0_ts = torch.cat((S0_ts, y00.unsqueeze(0)), dim=0)
            S1_ts = torch.cat((S1_ts, y01.unsqueeze(0)), dim=0)
            # swow rain melt
            psnow_ts = torch.cat((psnow_ts, psnow.unsqueeze(0)), dim=0)
            prain_ts = torch.cat((prain_ts, prain.unsqueeze(0)), dim=0)
            melt_ts = torch.cat((melt_ts, melt.unsqueeze(0)), dim=0)
            et_ts = torch.cat((et_ts, et.unsqueeze(0)), dim=0)

            return ds_combined

        torchdiffeq.odeint(func, q0, t_solver, method='euler') # options=dict(step_size=4)

        return qsim, ds_nn_TS, ds_nn_S0_ts, ds_nn_S1_ts, S0_ts, S1_ts, psnow_ts, prain_ts, melt_ts, et_ts


# #### Hybrid V2 (LamaH-CE) [Euler, RK4]

# In[ ]:


# Hybrid V2 with Euler
""" Final for now """
class Hybrid_V2(nn.Module):
    """
    In this model the neural network is connected to both storages. IS IT REALLY? WHY NN ONLY 1 OUPT NODE?
    """
    def __init__(self, p, et, temp, static_parameters):
        super().__init__()
        self.P = p
        self.ET = et
        self.T = temp

        self.layer1 = nn.Linear(5, 16)
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(16, 16)
        self.activation1a = nn.LeakyReLU()
        self.layer2 = nn.Linear(16, 2) ############################

        nn.init.normal_(self.layer1.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        ###
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        ###
        nn.init.normal_(self.layer2.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)


        # set the static parameters
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension
        qsim = torch.tensor([], requires_grad=False) #

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)

        q0 = q0.to(device)
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]
            et = self.ET[t]

            # GPU
            t = t.to(device)
            p = p.to(device)
            temp = temp.to(device)
            et = et.to(device)
            y = y.to(device)

            """ neural network """
            input = torch.cat([
                y[0,0].view(-1, 1),
                y[0,1].view(-1, 1),
                self.P[t].view(-1, 1),
                self.ET[t].view(-1, 1),
                self.T[t].view(-1, 1),
            ], dim=1)


            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)
            ###
            output = self.layer1a(output)
            output = self.activation1a(output)
            ###
            ds_nn = self.layer2(output)

            # GPU
            output = output.to(device)


            # I thinkt this should be automatically on the GPU
            ds_nn = self.layer2(output)

            """ conceptual part """
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            if temp < self.mint:
                psnow = p
                prain = torch.tensor(0, dtype=torch.float32)
            else:
                psnow = torch.tensor(0, dtype=torch.float32)
                prain = p


            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)

            # 2. melt -> in original done with snowbucket -> works
            if temp > self.maxt:
                if y[0,0] > 0:
                    melt = torch.min(y[0,0], self.ddf * (temp - self.maxt)).to(dtype=torch.float32) # try to create melt as torch tensor directly # check dtype
                else:
                    melt =  torch.tensor(0, dtype=torch.float32)
            else:
                melt =  torch.tensor(0, dtype=torch.float32)

            # GPU
            melt = melt.to(device)


            # 3. qsub qsurf -> in original done with soilbucket
            if y[0,1] < 0:
                qsub = torch.tensor(0, dtype=torch.float32)
                qsurf = torch.tensor(0, dtype=torch.float32)
            elif y[0,1] > self.smax:
                qsub = self.qmax # might have to turn this into a tensor?
                qsurf = y[0,1] - self.smax # also turn into tensor

            else:
                qsub = self.qmax * torch.exp(-self.f * (self.smax - y[0,1]))
                qsurf = torch.tensor(0, dtype=torch.float32)


            ###### qsub = qsub.to(device) # put back in if using GPU, else does not matter
            ###### qsurf = qsurf.to(device)

            # qsim
            qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ##### qsim = qsim.to(device)

            # dS
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)
            dS1 = prain + melt - self.ET[t] - qsub - qsurf # water storage
            dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ##### dS0 = dS0.to(device)
            ##### dS1 = dS1.to(device)

            ds_conceptual  = torch.cat([dS0, dS1], axis=0)
            # GPU
            ##### ds_conceptual = ds_conceptual.to(device)

            ds_combined = ds_nn + ds_conceptual ###
            # GPU
            ##### ds_combined = ds_combined.to(device)

            return ds_combined

        torchdiffeq.odeint(func, q0, t_solver, method='euler') # options=dict(step_size=4)

        return qsim


# In[ ]:


# Hybrid V2 with RK4
""" Final for now """
class Hybrid_V2RK4(nn.Module):
    """
    In this model the neural network is connected to both storages. IS IT REALLY? WHY NN ONLY 1 OUPT NODE?
    """
    def __init__(self, p, et, temp, static_parameters):
        super().__init__()
        self.P = p
        self.ET = et
        self.T = temp

        self.layer1 = nn.Linear(5, 16)
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(16, 16)
        self.activation1a = nn.LeakyReLU()
        self.layer2 = nn.Linear(16, 2) ############################

        nn.init.normal_(self.layer1.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        ###
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        ###
        nn.init.normal_(self.layer2.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)

        # set the static parameters
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension

        qsim = torch.tensor([], requires_grad=False) #

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)

        q0 = q0.to(device)
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]
            et = self.ET[t]

            # GPU
            t = t.to(device)
            p = p.to(device)
            temp = temp.to(device)
            et = et.to(device)
            y = y.to(device)

            """ neural network """
            input = torch.cat([
                y[0,0].view(-1, 1),
                y[0,1].view(-1, 1),
                self.P[t].view(-1, 1),
                self.ET[t].view(-1, 1),
                self.T[t].view(-1, 1),
            ], dim=1)


            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)
            ###
            output = self.layer1a(output)
            output = self.activation1a(output)
            ###
            ds_nn = self.layer2(output)

            # GPU
            output = output.to(device)


            # I thinkt this should be automatically on the GPU
            ds_nn = self.layer2(output)

            """ conceptual part """
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            if temp < self.mint:
                psnow = p
                prain = torch.tensor(0, dtype=torch.float32)
            else:
                psnow = torch.tensor(0, dtype=torch.float32)
                prain = p


            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)

            # 2. melt -> in original done with snowbucket -> works
            if temp > self.maxt:
                if y[0,0] > 0:
                    melt = torch.min(y[0,0], self.ddf * (temp - self.maxt)).to(dtype=torch.float32) # try to create melt as torch tensor directly # check dtype
                else:
                    melt =  torch.tensor(0, dtype=torch.float32)
            else:
                melt =  torch.tensor(0, dtype=torch.float32)

            # GPU
            melt = melt.to(device)


            # 3. qsub qsurf -> in original done with soilbucket
            if y[0,1] < 0:
                qsub = torch.tensor(0, dtype=torch.float32)
                qsurf = torch.tensor(0, dtype=torch.float32)
            elif y[0,1] > self.smax:
                qsub = self.qmax # might have to turn this into a tensor?
                qsurf = y[0,1] - self.smax # also turn into tensor

            else:
                qsub = self.qmax * torch.exp(-self.f * (self.smax - y[0,1]))
                qsurf = torch.tensor(0, dtype=torch.float32)


            ###### qsub = qsub.to(device) # put back in if using GPU, else does not matter
            ###### qsurf = qsurf.to(device)

            # qsim
            qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ##### qsim = qsim.to(device)

            # dS
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)
            dS1 = prain + melt - self.ET[t] - qsub - qsurf # water storage
            dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ##### dS0 = dS0.to(device)
            ##### dS1 = dS1.to(device)

            ds_conceptual  = torch.cat([dS0, dS1], axis=0)
            # GPU
            ##### ds_conceptual = ds_conceptual.to(device)

            ds_combined = ds_nn + ds_conceptual ###
            # GPU
            ##### ds_combined = ds_combined.to(device)

            return ds_combined

        torchdiffeq.odeint(func, q0, t_solver, method='rk4') # options=dict(step_size=4)

        return qsim


# ###  Hybrid Model - SingleV1 and SingleV2
# To Do:
# - NN connected to only one storage
# 

# #### Hybrid V1 Single (E-OBS) [Euler, RK4]

# In[ ]:


# Hybrid V1 Single with Euler
class Hybrid_SingleV1(nn.Module):
    """
    This model connects the neural network only to the water storage
    """
    def __init__(self, p, pet, temp, static_parameters):
        super().__init__()
        self.P = p
        self.PET = pet
        self.T = temp

        self.layer1 = nn.Linear(4, 16)
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(16, 16)
        self.activation1a = nn.LeakyReLU()
        self.layer2 = nn.Linear(16, 1) ###

        nn.init.normal_(self.layer1.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        ###
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        ###
        nn.init.normal_(self.layer2.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)

        # timespan
        self.timespan = self.P.shape[0] ## can this go?

        # set the static parameters
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension


        #### this + the torch.cat later causes the problem
        qsim = torch.tensor([], requires_grad=False) ### True does not seem to need gradient tracking

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)
        q0 = q0.to(device)
        qsim = qsim.to(device)


        # why is melt even created as an array / tensor? do we need it as that?

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]
            pet = self.PET[t]

            # GPU
            t = t.to(device)
            p = p.to(device)
            temp = temp.to(device)
            pet = pet.to(device)
            y = y.to(device)


            """start of neural network"""
            input = torch.cat([
                # y[0,0].view(-1, 1), ###
                y[0,1].view(-1, 1),
                self.P[t].view(-1, 1),
                self.PET[t].view(-1, 1),
                self.T[t].view(-1, 1),
            ], dim=1)


            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)

            output = self.layer1a(output)
            output = self.activation1a(output)

            # GPU
            ### output = output.to(device)


            # I thinkt this should be automatically on the GPU
            ds_nn = self.layer2(output)



            """start of conceptual part ds"""
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            if temp < self.mint:
                psnow = p
                prain = torch.tensor(0, dtype=torch.float32)
            else:
                psnow = torch.tensor(0, dtype=torch.float32)
                prain = p

            # GPU
            ### psnow = psnow.to(device)
            ### prain = prain.to(device)


            # 2. melt -> in original done with snowbucket -> works
            if temp > self.maxt:
                if y[0,0] > 0:
                    melt = torch.min(y[0,0], self.ddf * (temp - self.maxt)).to(dtype=torch.float32) # try to create melt as torch tensor directly # check dtype
                else:
                    melt =  torch.tensor(0, dtype=torch.float32)

            else:
                melt =  torch.tensor(0, dtype=torch.float32)


            # GPU
            melt = melt.to(device)


            # 3. qsub qsurf -> in original done with soilbucket
            if y[0,1] < 0:
                qsub = torch.tensor(0, dtype=torch.float32)
                qsurf = torch.tensor(0, dtype=torch.float32)

                et = torch.tensor(0, dtype=torch.float32) ### added
            elif y[0,1] > self.smax:
                qsub = self.qmax # might have to turn this into a tensor?
                qsurf = y[0,1] - self.smax # also turn into tensor

                et = pet ### added

            else:
                qsub = self.qmax * torch.exp(-self.f * (self.smax - y[0,1]))
                qsurf = torch.tensor(0, dtype=torch.float32)

                et = pet * (y[0,1] / self.smax) ### added

            ### qsub = qsub.to(device)
            ### qsurf = qsurf.to(device)

            # qsim
            qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            ###qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ### qsim = qsim.to(device)


            # dS
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)
            dS1 = prain + melt - et - qsub - qsurf # water storage
            dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ### dS0 = dS0.to(device)
            #### dS1 = dS1.to(device)

            ### need to unsqueze dS0 and dS1 a second time?
            # dS0 = torch.unsqueeze(dS0, 0)
            # dS1 = torch.unsqueeze(dS1, 0)


            dS1_combined = dS1[0] + ds_nn[0]
            ds_combined = torch.cat([dS0, dS1_combined,], axis=0)

            # GPU
            ### dS1_combined = dS1_combined.to(device)
            ### ds_combined = ds_combined.to(device)

            return ds_combined

        # ODE solver
        torchdiffeq.odeint(func, q0, t_solver, method='euler')

        return qsim


# In[ ]:


# Hybrid V1 Single with RK4
class Hybrid_SingleV1RK4(nn.Module):
    """
    This model connects the neural network only to the water storage
    """
    def __init__(self, p, pet, temp, static_parameters):
        super().__init__()
        self.P = p
        self.PET = pet
        self.T = temp

        self.layer1 = nn.Linear(4, 16)
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(16, 16)
        self.activation1a = nn.LeakyReLU()
        self.layer2 = nn.Linear(16, 1) ###

        nn.init.normal_(self.layer1.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        ###
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        ###
        nn.init.normal_(self.layer2.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)

        # timespan
        self.timespan = self.P.shape[0] ## can this go?

        # set the static parameters
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension


        #### this + the torch.cat later causes the problem
        qsim = torch.tensor([], requires_grad=False) ### True does not seem to need gradient tracking

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)
        q0 = q0.to(device)
        qsim = qsim.to(device)


        # why is melt even created as an array / tensor? do we need it as that?

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]
            pet = self.PET[t]

            # GPU
            t = t.to(device)
            p = p.to(device)
            temp = temp.to(device)
            pet = pet.to(device)
            y = y.to(device)


            """start of neural network"""
            input = torch.cat([
                # y[0,0].view(-1, 1), ###
                y[0,1].view(-1, 1),
                self.P[t].view(-1, 1),
                self.PET[t].view(-1, 1),
                self.T[t].view(-1, 1),
            ], dim=1)


            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)

            output = self.layer1a(output)
            output = self.activation1a(output)

            # GPU
            ### output = output.to(device)


            # I thinkt this should be automatically on the GPU
            ds_nn = self.layer2(output)



            """start of conceptual part ds"""
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            if temp < self.mint:
                psnow = p
                prain = torch.tensor(0, dtype=torch.float32)
            else:
                psnow = torch.tensor(0, dtype=torch.float32)
                prain = p

            # GPU
            ### psnow = psnow.to(device)
            ### prain = prain.to(device)


            # 2. melt -> in original done with snowbucket -> works
            if temp > self.maxt:
                if y[0,0] > 0:
                    melt = torch.min(y[0,0], self.ddf * (temp - self.maxt)).to(dtype=torch.float32) # try to create melt as torch tensor directly # check dtype
                else:
                    melt =  torch.tensor(0, dtype=torch.float32)

            else:
                melt =  torch.tensor(0, dtype=torch.float32)


            # GPU
            melt = melt.to(device)


            # 3. qsub qsurf -> in original done with soilbucket
            if y[0,1] < 0:
                qsub = torch.tensor(0, dtype=torch.float32)
                qsurf = torch.tensor(0, dtype=torch.float32)

                et = torch.tensor(0, dtype=torch.float32) ### added
            elif y[0,1] > self.smax:
                qsub = self.qmax # might have to turn this into a tensor?
                qsurf = y[0,1] - self.smax # also turn into tensor

                et = pet ### added

            else:
                qsub = self.qmax * torch.exp(-self.f * (self.smax - y[0,1]))
                qsurf = torch.tensor(0, dtype=torch.float32)

                et = pet * (y[0,1] / self.smax) ### added

            ### qsub = qsub.to(device)
            ### qsurf = qsurf.to(device)

            # qsim
            qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            ###qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ### qsim = qsim.to(device)


            # dS
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)
            dS1 = prain + melt - et - qsub - qsurf # water storage
            dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ### dS0 = dS0.to(device)
            #### dS1 = dS1.to(device)

            ### need to unsqueze dS0 and dS1 a second time?
            # dS0 = torch.unsqueeze(dS0, 0)
            # dS1 = torch.unsqueeze(dS1, 0)


            dS1_combined = dS1[0] + ds_nn[0]
            ds_combined = torch.cat([dS0, dS1_combined,], axis=0)

            # GPU
            ### dS1_combined = dS1_combined.to(device)
            ### ds_combined = ds_combined.to(device)

            return ds_combined

        # ODE solver
        torchdiffeq.odeint(func, q0, t_solver, method='rk4')

        return qsim


# #### Hybrid V2 Single (LamaH-CE) [Euler, RK4]

# In[ ]:


# Hybrid V2 Single with Euler
class Hybrid_SingleV2(nn.Module):
    """
    This model connects the neural network only to the water storage
    """
    def __init__(self, p, et, temp, static_parameters):
        super().__init__()
        self.P = p
        self.ET = et
        self.T = temp

        self.layer1 = nn.Linear(4, 16)
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(16, 16)
        self.activation1a = nn.LeakyReLU()
        self.layer2 = nn.Linear(16, 1) ###

        nn.init.normal_(self.layer1.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        ###
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        ###
        nn.init.normal_(self.layer2.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)

        # timespan
        self.timespan = self.P.shape[0] ## can this go?

        # set the static parameters
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension


        #### this + the torch.cat later causes the problem
        qsim = torch.tensor([], requires_grad=False) ### True does not seem to need gradient tracking

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)
        q0 = q0.to(device)
        qsim = qsim.to(device)


        # why is melt even created as an array / tensor? do we need it as that?

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]
            et = self.ET[t]

            # GPU
            t = t.to(device)
            p = p.to(device)
            temp = temp.to(device)
            et = et.to(device)
            y = y.to(device)


            """start of neural network"""
            input = torch.cat([
                # y[0,0].view(-1, 1), ###
                y[0,1].view(-1, 1),
                self.P[t].view(-1, 1),
                self.ET[t].view(-1, 1),
                self.T[t].view(-1, 1),
            ], dim=1)


            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)

            output = self.layer1a(output)
            output = self.activation1a(output)

            # GPU
            ### output = output.to(device)


            # I thinkt this should be automatically on the GPU
            ds_nn = self.layer2(output)



            """start of conceptual part ds"""
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            if temp < self.mint:
                psnow = p
                prain = torch.tensor(0, dtype=torch.float32)
            else:
                psnow = torch.tensor(0, dtype=torch.float32)
                prain = p

            # GPU
            ### psnow = psnow.to(device)
            ### prain = prain.to(device)


            # 2. melt -> in original done with snowbucket -> works
            if temp > self.maxt:
                if y[0,0] > 0:
                    melt = torch.min(y[0,0], self.ddf * (temp - self.maxt)).to(dtype=torch.float32) # try to create melt as torch tensor directly # check dtype
                else:
                    melt =  torch.tensor(0, dtype=torch.float32)

            else:
                melt =  torch.tensor(0, dtype=torch.float32)


            # GPU
            melt = melt.to(device)


            # 3. qsub qsurf -> in original done with soilbucket
            if y[0,1] < 0:
                qsub = torch.tensor(0, dtype=torch.float32)
                qsurf = torch.tensor(0, dtype=torch.float32)
            elif y[0,1] > self.smax:
                qsub = self.qmax # might have to turn this into a tensor?
                qsurf = y[0,1] - self.smax # also turn into tensor

            else:
                qsub = self.qmax * torch.exp(-self.f * (self.smax - y[0,1]))
                qsurf = torch.tensor(0, dtype=torch.float32)

            ### qsub = qsub.to(device)
            ### qsurf = qsurf.to(device)

            # qsim
            qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            ###qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ### qsim = qsim.to(device)


            # dS
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)
            dS1 = prain + melt - et - qsub - qsurf # water storage
            dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ### dS0 = dS0.to(device)
            #### dS1 = dS1.to(device)

            ### need to unsqueze dS0 and dS1 a second time?
            # dS0 = torch.unsqueeze(dS0, 0)
            # dS1 = torch.unsqueeze(dS1, 0)


            dS1_combined = dS1[0] + ds_nn[0]
            ds_combined = torch.cat([dS0, dS1_combined,], axis=0)

            # GPU
            ### dS1_combined = dS1_combined.to(device)
            ### ds_combined = ds_combined.to(device)

            return ds_combined

        # ODE solver
        torchdiffeq.odeint(func, q0, t_solver, method='euler')

        return qsim


# In[ ]:


# Hybrid V2 Single with RK4
class Hybrid_SingleV2RK4(nn.Module):
    """
    This model connects the neural network only to the water storage
    """
    def __init__(self, p, et, temp, static_parameters):
        super().__init__()
        self.P = p
        self.ET = et
        self.T = temp

        self.layer1 = nn.Linear(4, 16)
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(16, 16)
        self.activation1a = nn.LeakyReLU()
        self.layer2 = nn.Linear(16, 1) ###

        nn.init.normal_(self.layer1.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        ###
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        ###
        nn.init.normal_(self.layer2.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)

        # timespan
        self.timespan = self.P.shape[0] ## can this go?

        # set the static parameters
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension


        #### this + the torch.cat later causes the problem
        qsim = torch.tensor([], requires_grad=False) ### True does not seem to need gradient tracking

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)
        q0 = q0.to(device)
        qsim = qsim.to(device)


        # why is melt even created as an array / tensor? do we need it as that?

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]
            et = self.ET[t]

            # GPU
            t = t.to(device)
            p = p.to(device)
            temp = temp.to(device)
            et = et.to(device)
            y = y.to(device)


            """start of neural network"""
            input = torch.cat([
                # y[0,0].view(-1, 1), ###
                y[0,1].view(-1, 1),
                self.P[t].view(-1, 1),
                self.ET[t].view(-1, 1),
                self.T[t].view(-1, 1),
            ], dim=1)


            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)

            output = self.layer1a(output)
            output = self.activation1a(output)

            # GPU
            ### output = output.to(device)


            # I thinkt this should be automatically on the GPU
            ds_nn = self.layer2(output)



            """start of conceptual part ds"""
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            if temp < self.mint:
                psnow = p
                prain = torch.tensor(0, dtype=torch.float32)
            else:
                psnow = torch.tensor(0, dtype=torch.float32)
                prain = p

            # GPU
            ### psnow = psnow.to(device)
            ### prain = prain.to(device)


            # 2. melt -> in original done with snowbucket -> works
            if temp > self.maxt:
                if y[0,0] > 0:
                    melt = torch.min(y[0,0], self.ddf * (temp - self.maxt)).to(dtype=torch.float32) # try to create melt as torch tensor directly # check dtype
                else:
                    melt =  torch.tensor(0, dtype=torch.float32)

            else:
                melt =  torch.tensor(0, dtype=torch.float32)


            # GPU
            melt = melt.to(device)


            # 3. qsub qsurf -> in original done with soilbucket
            if y[0,1] < 0:
                qsub = torch.tensor(0, dtype=torch.float32)
                qsurf = torch.tensor(0, dtype=torch.float32)
            elif y[0,1] > self.smax:
                qsub = self.qmax # might have to turn this into a tensor?
                qsurf = y[0,1] - self.smax # also turn into tensor

            else:
                qsub = self.qmax * torch.exp(-self.f * (self.smax - y[0,1]))
                qsurf = torch.tensor(0, dtype=torch.float32)

            ### qsub = qsub.to(device)
            ### qsurf = qsurf.to(device)

            # qsim
            qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            ###qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ### qsim = qsim.to(device)


            # dS
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)
            dS1 = prain + melt - et - qsub - qsurf # water storage
            dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ### dS0 = dS0.to(device)
            #### dS1 = dS1.to(device)

            ### need to unsqueze dS0 and dS1 a second time?
            # dS0 = torch.unsqueeze(dS0, 0)
            # dS1 = torch.unsqueeze(dS1, 0)


            dS1_combined = dS1[0] + ds_nn[0]
            ds_combined = torch.cat([dS0, dS1_combined,], axis=0)

            # GPU
            ### dS1_combined = dS1_combined.to(device)
            ### ds_combined = ds_combined.to(device)

            return ds_combined

        # ODE solver
        torchdiffeq.odeint(func, q0, t_solver, method='rk4')

        return qsim


# ### Hybrid Model - V1 Large and V2 Large
# Take additional inputs from the SAME dataset.
# 
# In this version we will add additional meteorological input forcings to the neural network. This means that the neural network receives some input forcings that the conceptual part of the model does not receive as input.
# 
# **IMPORTANT:** This of course also allows for the possibility to add observations from another dataset, make sure to be consistent here!

# #### Hybrid Model V1Large [Euler and RK4] 🍄 NEED TO Check if it works 🌵 🌵
# takes additional input forcings (from the same dataset)

# In[ ]:


# hybrid V1 Large with Euler
class Hybrid_V1_large(nn.Module):
    """
    This model connects the neural only to the water storage and it takes additional input variables.
    This model is designed to run with UNSCALED data.
    """

    # variabels 'tmax', 'tmean', 'tmin', 'seapress', 'humidity', 'windspeed', 'srad', 'albedo', 'pet', 'daylength', 'pev', 'prcp']
    def __init__(self, tmax, tmean, tmin, seapress, humidity, windspeed, srad, albedo, pet, daylength, pev, prcp, static_parameters):
        super().__init__()
        """
        self.P = p
        self.PET = pet
        self.T = temp
        """

        ### all variables
        self.tmax = tmax
        self.T = tmean # rename
        self.tmin = tmin
        self.seapress = seapress
        self.humidity = humidity
        self.windspeed = windspeed
        self.srad = srad
        self.albedo = albedo
        self.daylength = daylength
        self.pev = pev
        self.PET = pet # rename
        self.P = prcp # rename



        self.layer1 = nn.Linear(13, 20) ### variables + 1 for the water storage
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(20, 20)
        self.activation1a = nn.LeakyReLU()
        self.layer2 = nn.Linear(20, 1)

        nn.init.normal_(self.layer1.weight, mean=0, std=0.05) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.05) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        nn.init.normal_(self.layer2.weight, mean=0, std=0.05) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)

        # timespan
        self.timespan = self.P.shape[0]

        # set the static parameters
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension
        qsim = torch.tensor([1.0], requires_grad=False)
        # why is melt even created as an array / tensor? do we need it as that?

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)
        q0 = q0.to(device)
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)

            ### assign variables here or in __init__?
            ### this is what is usually done here but now we will refer to the instnace variables directly, is this good or bad or no difference?

            p = self.P[t]
            temp = self.T[t]
            pet = self.PET[t]


            """start of neural network"""
            input = torch.cat([
                y[0,1].view(-1, 1),
                # y[0,1].view(-1, 1),
                self.tmax[t].view(-1,1),
                self.T[t].view(-1,1),
                self.tmin[t].view(-1,1),
                self.seapress[t].view(-1,1),
                self.humidity[t].view(-1,1),
                self.windspeed[t].view(-1,1),
                self.srad[t].view(-1,1),
                self.albedo[t].view(-1,1),
                self.daylength[t].view(-1,1),
                self.pev[t].view(-1,1),
                self.P[t].view(-1, 1),
                self.PET[t].view(-1, 1)
            ], dim=1)

            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)
            ###
            output = self.layer1a(output)
            output = self.activation1a(output)
            ###
            ds_nn = self.layer2(output)

            # GPU
            outout = output.to(device)

            """start of conceptual part ds"""
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            ###if self.T[t] < self.mint:
            if temp < self.mint:
                psnow = p
                ###psnow = self.P[t]
                prain = torch.tensor(0, dtype=torch.float32)
            else:
                psnow = torch.tensor(0, dtype=torch.float32)
                prain = p
                ### prain = self.P[t]

            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)


            # 2. melt -> in original done with snowbucket -> works
            ### if self.T[t] > self.maxt:
            if temp > self.maxt:
                if y[0,0] > 0:
                    melt = torch.min(y[0,0], self.ddf * (temp - self.maxt)).to(dtype=torch.float32) # try to create melt as torch tensor directly # check dtype
                    ###melt = torch.min(y[0,0], self.ddf * (self.T[t] - self.maxt)).to(dtype=torch.float32)
                else:
                    melt =  torch.tensor(0, dtype=torch.float32)
            else:
                melt =  torch.tensor(0, dtype=torch.float32)


            # GPU
            melt = melt.to(device)

            # 3. qsub qsurf -> in original done with soilbucket
            if y[0,1] < 0:
                qsub = torch.tensor(0, dtype=torch.float32)
                qsurf = torch.tensor(0, dtype=torch.float32)

                ######## added
                et = torch.tensor(0, dtype=torch.float32)

            elif y[0,1] > self.smax:
                qsub = self.qmax # might have to turn this into a tensor?
                qsurf = y[0,1] - self.smax # also turn into tensor

                ####### added
                et = pet

            else:
                qsub = self.qmax * torch.exp(-self.f * (self.smax - y[0,1]))
                qsurf = torch.tensor(0, dtype=torch.float32)

                #### added
                et = pet * (y[0,1] / self.smax)


            # GPU
            ### qsub = qsub.to(device)
            ### qsurf = qsurf.to(device)

            # qsim
            qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ### qsim = qsim.to(device)

            # dS
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)
            dS1 = prain + melt - et - qsub - qsurf # water storage
            dS1 = torch.unsqueeze(dS1, 0)

            ### need to unsqueze dS0 and dS1 a second time?
            # dS0 = torch.unsqueeze(dS0, 0)
            # dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ### dS0 = dS0.to(device)
            ### dS1 = dS1.to(device)



            dS1_combined = dS1[0] + ds_nn[0]
            ds_combined = torch.cat([dS0, dS1_combined,], axis=0)

            return ds_combined

        # ODE solver
        torchdiffeq.odeint(func, q0, t_solver, method='euler')

        return qsim


# In[ ]:


# hybrid V1 Large with Euler
class Hybrid_V1RK4_large(nn.Module):
    """
    This model connects the neural only to the water storage and it takes additional input variables.
    This model is designed to run with UNSCALED data.
    """

    # variabels 'tmax', 'tmean', 'tmin', 'seapress', 'humidity', 'windspeed', 'srad', 'albedo', 'pet', 'daylength', 'pev', 'prcp']
    def __init__(self, tmax, tmean, tmin, seapress, humidity, windspeed, srad, albedo, pet, daylength, pev, prcp, static_parameters):
        super().__init__()
        """
        self.P = p
        self.PET = pet
        self.T = temp
        """

        ### all variables
        self.tmax = tmax
        self.T = tmean # rename
        self.tmin = tmin
        self.seapress = seapress
        self.humidity = humidity
        self.windspeed = windspeed
        self.srad = srad
        self.albedo = albedo
        self.daylength = daylength
        self.pev = pev
        self.PET = pet # rename
        self.P = prcp # rename



        self.layer1 = nn.Linear(13, 20) ### variables + 1 for the water storage
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(20, 20)
        self.activation1a = nn.LeakyReLU()
        self.layer2 = nn.Linear(20, 1)

        nn.init.normal_(self.layer1.weight, mean=0, std=0.05) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.05) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        nn.init.normal_(self.layer2.weight, mean=0, std=0.05) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)

        # timespan
        self.timespan = self.P.shape[0]

        # set the static parameters
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension
        qsim = torch.tensor([1.0], requires_grad=False)
        # why is melt even created as an array / tensor? do we need it as that?

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)
        q0 = q0.to(device)
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)

            ### assign variables here or in __init__?
            ### this is what is usually done here but now we will refer to the instnace variables directly, is this good or bad or no difference?

            p = self.P[t]
            temp = self.T[t]
            pet = self.PET[t]


            """start of neural network"""
            input = torch.cat([
                y[0,1].view(-1, 1),
                # y[0,1].view(-1, 1),
                self.tmax[t].view(-1,1),
                self.T[t].view(-1,1),
                self.tmin[t].view(-1,1),
                self.seapress[t].view(-1,1),
                self.humidity[t].view(-1,1),
                self.windspeed[t].view(-1,1),
                self.srad[t].view(-1,1),
                self.albedo[t].view(-1,1),
                self.daylength[t].view(-1,1),
                self.pev[t].view(-1,1),
                self.P[t].view(-1, 1),
                self.PET[t].view(-1, 1)
            ], dim=1)

            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)
            ###
            output = self.layer1a(output)
            output = self.activation1a(output)
            ###
            ds_nn = self.layer2(output)

            # GPU
            outout = output.to(device)

            """start of conceptual part ds"""
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            ###if self.T[t] < self.mint:
            if temp < self.mint:
                psnow = p
                ###psnow = self.P[t]
                prain = torch.tensor(0, dtype=torch.float32)
            else:
                psnow = torch.tensor(0, dtype=torch.float32)
                prain = p
                ### prain = self.P[t]

            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)


            # 2. melt -> in original done with snowbucket -> works
            ### if self.T[t] > self.maxt:
            if temp > self.maxt:
                if y[0,0] > 0:
                    melt = torch.min(y[0,0], self.ddf * (temp - self.maxt)).to(dtype=torch.float32) # try to create melt as torch tensor directly # check dtype
                    ###melt = torch.min(y[0,0], self.ddf * (self.T[t] - self.maxt)).to(dtype=torch.float32)
                else:
                    melt =  torch.tensor(0, dtype=torch.float32)
            else:
                melt =  torch.tensor(0, dtype=torch.float32)


            # GPU
            melt = melt.to(device)

            # 3. qsub qsurf -> in original done with soilbucket
            if y[0,1] < 0:
                qsub = torch.tensor(0, dtype=torch.float32)
                qsurf = torch.tensor(0, dtype=torch.float32)

                ######## added
                et = torch.tensor(0, dtype=torch.float32)

            elif y[0,1] > self.smax:
                qsub = self.qmax # might have to turn this into a tensor?
                qsurf = y[0,1] - self.smax # also turn into tensor

                ####### added
                et = pet

            else:
                qsub = self.qmax * torch.exp(-self.f * (self.smax - y[0,1]))
                qsurf = torch.tensor(0, dtype=torch.float32)

                #### added
                et = pet * (y[0,1] / self.smax)


            # GPU
            ### qsub = qsub.to(device)
            ### qsurf = qsurf.to(device)

            # qsim
            qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ### qsim = qsim.to(device)

            # dS
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)
            dS1 = prain + melt - et - qsub - qsurf # water storage
            dS1 = torch.unsqueeze(dS1, 0)

            ### need to unsqueze dS0 and dS1 a second time?
            # dS0 = torch.unsqueeze(dS0, 0)
            # dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ### dS0 = dS0.to(device)
            ### dS1 = dS1.to(device)



            dS1_combined = dS1[0] + ds_nn[0]
            ds_combined = torch.cat([dS0, dS1_combined,], axis=0)

            return ds_combined

        # ODE solver
        torchdiffeq.odeint(func, q0, t_solver, method='rk4')

        return qsim


# In[ ]:


# actually we want this smaller with just the same expansion as for the LSTM Large
# variables =  ['prcp', 'tmean', 'pet', 'srad', 'tmin', 'tmax']
# hybrid V1 Large with Euler
class Hybrid_V1RK4_large(nn.Module):
    """
    This model connects the neural only to the water storage and it takes additional input variables.
    This model is designed to run with UNSCALED data.
    """

    # variabels 'tmax', 'tmean', 'tmin', 'seapress', 'humidity', 'windspeed', 'srad', 'albedo', 'pet', 'daylength', 'pev', 'prcp']
    def __init__(self, tmax, tmean, tmin, srad, pet, prcp, static_parameters):
        super().__init__()
        """
        self.P = p
        self.PET = pet
        self.T = temp
        """

        ### all variables
        self.tmax = tmax
        self.T = tmean # rename
        self.tmin = tmin
        self.srad = srad
        self.PET = pet # rename
        self.P = prcp # rename



        self.layer1 = nn.Linear(7, 16) ### variables + 1 for the water storage
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(16, 16)
        self.activation1a = nn.LeakyReLU()
        self.layer2 = nn.Linear(16, 1)

        nn.init.normal_(self.layer1.weight, mean=0, std=0.05) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.05) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        nn.init.normal_(self.layer2.weight, mean=0, std=0.05) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)

        # timespan
        self.timespan = self.P.shape[0]

        # set the static parameters
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension
        qsim = torch.tensor([1.0], requires_grad=False)
        # why is melt even created as an array / tensor? do we need it as that?

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)
        q0 = q0.to(device)
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)

            ### assign variables here or in __init__?
            ### this is what is usually done here but now we will refer to the instnace variables directly, is this good or bad or no difference?

            p = self.P[t]
            temp = self.T[t]
            pet = self.PET[t]


            """start of neural network"""
            input = torch.cat([
                y[0,1].view(-1, 1),
                # y[0,1].view(-1, 1),
                self.tmax[t].view(-1,1),
                self.T[t].view(-1,1),
                self.tmin[t].view(-1,1),
                self.srad[t].view(-1,1),
                self.P[t].view(-1, 1),
                self.PET[t].view(-1, 1)
            ], dim=1)

            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)
            ###
            output = self.layer1a(output)
            output = self.activation1a(output)
            ###
            ds_nn = self.layer2(output)

            # GPU
            outout = output.to(device)

            """start of conceptual part ds"""
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            ###if self.T[t] < self.mint:
            if temp < self.mint:
                psnow = p
                ###psnow = self.P[t]
                prain = torch.tensor(0, dtype=torch.float32)
            else:
                psnow = torch.tensor(0, dtype=torch.float32)
                prain = p
                ### prain = self.P[t]

            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)


            # 2. melt -> in original done with snowbucket -> works
            ### if self.T[t] > self.maxt:
            if temp > self.maxt:
                if y[0,0] > 0:
                    melt = torch.min(y[0,0], self.ddf * (temp - self.maxt)).to(dtype=torch.float32) # try to create melt as torch tensor directly # check dtype
                    ###melt = torch.min(y[0,0], self.ddf * (self.T[t] - self.maxt)).to(dtype=torch.float32)
                else:
                    melt =  torch.tensor(0, dtype=torch.float32)
            else:
                melt =  torch.tensor(0, dtype=torch.float32)


            # GPU
            melt = melt.to(device)

            # 3. qsub qsurf -> in original done with soilbucket
            if y[0,1] < 0:
                qsub = torch.tensor(0, dtype=torch.float32)
                qsurf = torch.tensor(0, dtype=torch.float32)

                ######## added
                et = torch.tensor(0, dtype=torch.float32)

            elif y[0,1] > self.smax:
                qsub = self.qmax # might have to turn this into a tensor?
                qsurf = y[0,1] - self.smax # also turn into tensor

                ####### added
                et = pet

            else:
                qsub = self.qmax * torch.exp(-self.f * (self.smax - y[0,1]))
                qsurf = torch.tensor(0, dtype=torch.float32)

                #### added
                et = pet * (y[0,1] / self.smax)


            # GPU
            ### qsub = qsub.to(device)
            ### qsurf = qsurf.to(device)

            # qsim
            qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ### qsim = qsim.to(device)

            # dS
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)
            dS1 = prain + melt - et - qsub - qsurf # water storage
            dS1 = torch.unsqueeze(dS1, 0)

            ### need to unsqueze dS0 and dS1 a second time?
            # dS0 = torch.unsqueeze(dS0, 0)
            # dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ### dS0 = dS0.to(device)
            ### dS1 = dS1.to(device)



            dS1_combined = dS1[0] + ds_nn[0]
            ds_combined = torch.cat([dS0, dS1_combined,], axis=0)

            return ds_combined

        # ODE solver
        torchdiffeq.odeint(func, q0, t_solver, method='rk4')

        return qsim


# #### Hybrid Model V2 Large  [ Euler, no RK4 at this point]
# takes additional input forcings (from the same dataset)

# In[ ]:


# hybrid V2 Large with Euler
class Hybrid_V2_large(nn.Module):
    """
    This model connects the neural only to the water storage and it takes additional input variables.
    This model is designed to run with UNSCALED data.
    """
    def __init__(self, tmax, tmean, tmin, tdpmax, tdpmean, tdpmin, windu, windv, albedo, swe, sradmean, tradmean, et, prcp, static_parameters):
        super().__init__()
        """
        self.P = p
        self.ET = et
        self.T = temp
        """

        ### all variables
        self.tmax = tmax
        self.T = tmean # rename
        self.tmin = tmin
        self.tdpmax = tdpmax
        self.tdpmean = tdpmean
        self.tdpmin = tdpmin
        self.windu = windu
        self.windv = windv
        self.albedo = albedo
        self.swe = swe
        self.sradmean = sradmean
        self.tradmean = tradmean
        self.ET = et # rename
        self.P = prcp # rename



        self.layer1 = nn.Linear(15, 20) ### variables + 1 for the water storage
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(20, 20)
        self.activation1a = nn.LeakyReLU()
        self.layer2 = nn.Linear(20, 1)

        nn.init.normal_(self.layer1.weight, mean=0, std=0.05) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.05) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        nn.init.normal_(self.layer2.weight, mean=0, std=0.05) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)

        # timespan
        self.timespan = self.P.shape[0]

        # set the static parameters
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension
        qsim = torch.tensor([1.0], requires_grad=False)
        # why is melt even created as an array / tensor? do we need it as that?

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)
        q0 = q0.to(device)
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)

            ### assign variables here or in __init__?
            ### this is what is usually done here but now we will refer to the instnace variables directly, is this good or bad or no difference?

            p = self.P[t]
            temp = self.T[t]
            et = self.ET[t]


            """start of neural network"""
            input = torch.cat([
                y[0,1].view(-1, 1),
                # y[0,1].view(-1, 1),
                self.tmax[t].view(-1,1),
                self.T[t].view(-1,1),
                self.tmin[t].view(-1,1),
                self.tdpmax[t].view(-1,1),
                self.tdpmean[t].view(-1,1),
                self.tdpmin[t].view(-1,1),
                self.windu[t].view(-1,1),
                self.windv[t].view(-1,1),
                self.albedo[t].view(-1,1),
                self.swe[t].view(-1,1),
                self.sradmean[t].view(-1,1),
                self.tradmean[t].view(-1,1),
                self.P[t].view(-1, 1),
                self.ET[t].view(-1, 1)
            ], dim=1)

            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)
            ###
            output = self.layer1a(output)
            output = self.activation1a(output)
            ###
            ds_nn = self.layer2(output)

            # GPU
            outout = output.to(device)

            """start of conceptual part ds"""
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            ###if self.T[t] < self.mint:
            if temp < self.mint:
                psnow = p
                ###psnow = self.P[t]
                prain = torch.tensor(0, dtype=torch.float32)
            else:
                psnow = torch.tensor(0, dtype=torch.float32)
                prain = p
                ### prain = self.P[t]

            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)


            # 2. melt -> in original done with snowbucket -> works
            ### if self.T[t] > self.maxt:
            if temp > self.maxt:
                if y[0,0] > 0:
                    melt = torch.min(y[0,0], self.ddf * (temp - self.maxt)).to(dtype=torch.float32) # try to create melt as torch tensor directly # check dtype
                    ###melt = torch.min(y[0,0], self.ddf * (self.T[t] - self.maxt)).to(dtype=torch.float32)
                else:
                    melt =  torch.tensor(0, dtype=torch.float32)
            else:
                melt =  torch.tensor(0, dtype=torch.float32)


            # GPU
            melt = melt.to(device)

            # 3. qsub qsurf -> in original done with soilbucket
            if y[0,1] < 0:
                qsub = torch.tensor(0, dtype=torch.float32)
                qsurf = torch.tensor(0, dtype=torch.float32)
            elif y[0,1] > self.smax:
                qsub = self.qmax # might have to turn this into a tensor?
                qsurf = y[0,1] - self.smax # also turn into tensor

            else:
                qsub = self.qmax * torch.exp(-self.f * (self.smax - y[0,1]))
                qsurf = torch.tensor(0, dtype=torch.float32)


            # GPU
            ### qsub = qsub.to(device)
            ### qsurf = qsurf.to(device)

            # qsim
            qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ### qsim = qsim.to(device)

            # dS
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)
            dS1 = prain + melt - self.ET[t] - qsub - qsurf # water storage
            dS1 = torch.unsqueeze(dS1, 0)

            ### need to unsqueze dS0 and dS1 a second time?
            # dS0 = torch.unsqueeze(dS0, 0)
            # dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ### dS0 = dS0.to(device)
            ### dS1 = dS1.to(device)



            dS1_combined = dS1[0] + ds_nn[0]
            ds_combined = torch.cat([dS0, dS1_combined,], axis=0)

            return ds_combined

        # ODE solver
        torchdiffeq.odeint(func, q0, t_solver, method='euler')

        return qsim


# ### V1 Duo and V2 Duo 🍄
#  NEEED TO FINALISE THIS AND CHECK IF we want to use this
#  - Add V1 (E-OBS) version if we want to use it!!!

# #### 1.2.4.1 Hybrid Model V1 Duo - This takes as additional input data from another dataset (SCALED)
# 
# -> Add only if we want to use this

# #### 1.2.4.2 Hybrid Model V2 Duo - This takes as additional input data from another dataset (SCALED) # Under construction
# It is important that this is a very different model compared to the one above.

# In[ ]:


### under construction ###

# Hybrid V2 Duo
class Hybrid_V2_duo(nn.Module):
    """
    This model connects the neural only to the water storage and it takes additional input variables.
    """
    def __init__(self, p, et, temp, p_sc, et_sc, temp_sc, static_parameters):
        super().__init__()
        self.P = p
        self.ET = et
        self.T = temp

        ###
        self.temp_sc = temp_sc
        self.p_sc = p_sc
        self.et_sc = et_sc
        ###

        self.layer1 = nn.Linear(7, 20) ###
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(20, 20)
        self.activation1a = nn.LeakyReLU()
        self.layer2 = nn.Linear(20, 1)

        nn.init.normal_(self.layer1.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        nn.init.normal_(self.layer2.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)

        # timespan
        self.timespan = self.P.shape[0]

        # set the static parameters
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension
        qsim = torch.tensor([1.0], requires_grad=True)
        # why is melt even created as an array / tensor? do we need it as that?

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)
        q0 = q0.to(device)
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]
            et = self.ET[t]
            ### necessary???
            temp_sc = self.temp_sc[t]
            p_sc = self.p_sc[t]
            et_sc = self.et_sc[t]
            ###



            """start of neural network"""
            input = torch.cat([
                y[0,1].view(-1, 1),
                # y[0,1].view(-1, 1),
                self.P[t].view(-1, 1),
                self.ET[t].view(-1, 1),
                self.T[t].view(-1, 1),
                ###
                self.temp_sc[t].view(-1,1),
                self.p_sc[t].view(-1,1),
                self.et_sc[t].view(-1,1)
                ###
            ], dim=1)

            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)
            ###
            output = self.layer1a(output)
            output = self.activation1a(output)
            ###
            ds_nn = self.layer2(output)

            # GPU
            outout = output.to(device)

            """start of conceptual part ds"""
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            if temp < self.mint:
                psnow = p
                prain = torch.tensor(0, dtype=torch.float32)
            else:
                psnow = torch.tensor(0, dtype=torch.float32)
                prain = p


            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)


            # 2. melt -> in original done with snowbucket -> works
            if temp > self.maxt:
                if y[0,0] > 0:
                    melt = torch.min(y[0,0], self.ddf * (temp - self.maxt)).to(dtype=torch.float32) # try to create melt as torch tensor directly # check dtype
                else:
                    melt =  torch.tensor(0, dtype=torch.float32)
            else:
                melt =  torch.tensor(0, dtype=torch.float32)


            # GPU
            melt = melt.to(device)

            # 3. qsub qsurf -> in original done with soilbucket
            if y[0,1] < 0:
                qsub = torch.tensor(0, dtype=torch.float32)
                qsurf = torch.tensor(0, dtype=torch.float32)
            elif y[0,1] > self.smax:
                qsub = self.qmax # might have to turn this into a tensor?
                qsurf = y[0,1] - self.smax # also turn into tensor

            else:
                qsub = self.qmax * torch.exp(-self.f * (self.smax - y[0,1]))
                qsurf = torch.tensor(0, dtype=torch.float32)


            # GPU
            qsub = qsub.to(device)
            qsurf = qsurf.to(device)

            # qsim
            qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            qsim = qsim.to(device)

            # dS
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)
            dS1 = prain + melt - self.ET[t] - qsub - qsurf # water storage
            dS1 = torch.unsqueeze(dS1, 0)

            ### need to unsqueze dS0 and dS1 a second time?
            # dS0 = torch.unsqueeze(dS0, 0)
            # dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            dS0 = dS0.to(device)
            dS1 = dS1.to(device)



            dS1_combined = dS1[0] + ds_nn[0]
            ds_combined = torch.cat([dS0, dS1_combined,], axis=0)

            return ds_combined

        # ODE solver
        torchdiffeq.odeint(func, q0, t_solver, method='euler')

        return qsim


# #### 1.2.4.3 Now we want to achieve that the coinceptual part gets only the unscaled and the NN gets only the scaled 🍄 Are we going to use this?

# In[ ]:


### under construction ###

# hybrid version 3b
class Hybrid_V3b(nn.Module):
    """
    This model connects the neural only to the water storage and it takes additional input variables.
    """
    def __init__(self, p, et, temp, p_sc, et_sc, temp_sc, static_parameters):
        super().__init__()
        self.P = p
        self.ET = et
        self.T = temp

        ###
        self.temp_sc = temp_sc
        self.p_sc = p_sc
        self.et_sc = et_sc
        ###

        self.layer1 = nn.Linear(4, 10) ###
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(10, 10)
        self.activation1a = nn.LeakyReLU()
        self.layer2 = nn.Linear(10, 1)

        """

        nn.init.normal_(self.layer1.weight, mean=0, std=0.25) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.25) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        nn.init.normal_(self.layer2.weight, mean=0, std=0.25) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)

        """

        # timespan
        self.timespan = self.P.shape[0]

        # set the static parameters
        self.f = static_parameters[0]
        self.smax = static_parameters[1]
        self.qmax = static_parameters[2]
        self.ddf = static_parameters[3]
        self.mint = static_parameters[4]
        self.maxt = static_parameters[5]

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension
        qsim = torch.tensor([1.0], requires_grad=True)
        # why is melt even created as an array / tensor? do we need it as that?

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)
        q0 = q0.to(device)
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]
            et = self.ET[t]
            ### necessary???
            temp_sc = self.temp_sc[t]
            p_sc = self.p_sc[t]
            et_sc = self.et_sc[t]
            ###



            """start of neural network"""
            input = torch.cat([
                y[0,1].view(-1, 1),
                # y[0,1].view(-1, 1),
                ### self.P[t].view(-1, 1),
                ### self.ET[t].view(-1, 1),
                ### self.T[t].view(-1, 1),
                ###
                self.temp_sc[t].view(-1,1),
                self.p_sc[t].view(-1,1),
                self.et_sc[t].view(-1,1)
                ###
            ], dim=1)

            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)
            ###
            output = self.layer1a(output)
            output = self.activation1a(output)
            ###
            ds_nn = self.layer2(output)

            # GPU
            outout = output.to(device)

            """start of conceptual part ds"""
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            if temp < self.mint:
                psnow = p
                prain = torch.tensor(0, dtype=torch.float32)
            else:
                psnow = torch.tensor(0, dtype=torch.float32)
                prain = p


            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)


            # 2. melt -> in original done with snowbucket -> works
            if temp > self.maxt:
                if y[0,0] > 0:
                    melt = torch.min(y[0,0], self.ddf * (temp - self.maxt)).to(dtype=torch.float32) # try to create melt as torch tensor directly # check dtype
                else:
                    melt =  torch.tensor(0, dtype=torch.float32)
            else:
                melt =  torch.tensor(0, dtype=torch.float32)


            # GPU
            melt = melt.to(device)

            # 3. qsub qsurf -> in original done with soilbucket
            if y[0,1] < 0:
                qsub = torch.tensor(0, dtype=torch.float32)
                qsurf = torch.tensor(0, dtype=torch.float32)
            elif y[0,1] > self.smax:
                qsub = self.qmax # might have to turn this into a tensor?
                qsurf = y[0,1] - self.smax # also turn into tensor

            else:
                qsub = self.qmax * torch.exp(-self.f * (self.smax - y[0,1]))
                qsurf = torch.tensor(0, dtype=torch.float32)


            # GPU
            qsub = qsub.to(device)
            qsurf = qsurf.to(device)

            # qsim
            qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            qsim = qsim.to(device)

            # dS
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)
            dS1 = prain + melt - self.ET[t] - qsub - qsurf # water storage
            dS1 = torch.unsqueeze(dS1, 0)

            ### need to unsqueze dS0 and dS1 a second time?
            # dS0 = torch.unsqueeze(dS0, 0)
            # dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            dS0 = dS0.to(device)
            dS1 = dS1.to(device)



            dS1_combined = dS1[0] + ds_nn[0]
            ds_combined = torch.cat([dS0, dS1_combined,], axis=0)

            return ds_combined

        # ODE solver
        torchdiffeq.odeint(func, q0, t_solver, method='euler')

        return qsim


# ### Hybrid Model Version HybridV1M100 and HybridV2M100 (2024)
# 
# 
# This is in its most important aspects analohous to Hoeges M100.
# 

# #### V1M100 Euler and RK4

# In[ ]:


# For now this is just a copy from V1
class Hybrid_V1M100(nn.Module):
    """
    In this model the neural network is connected to both storages. IS IT REALLY? WHY NN ONLY 1 OUPT NODE?
    """
    def __init__(self, p, pet, temp):
        super().__init__()
        self.P = p
        self.PET = pet
        self.T = temp

        ### increased nodes per layer to 20
        self.layer1 = nn.Linear(4, 16) ### 4 inputs instead of 5
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(16, 16)
        self.activation1a = nn.LeakyReLU()
        #self.layer1b = nn.Linear(20, 20)
        #self.activation1b = nn.LeakyReLU()
        self.layer2 = nn.Linear(16, 5) ###5 output nodes instead of 2



        nn.init.normal_(self.layer1.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        ###
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        ###
        nn.init.normal_(self.layer2.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)

        # timespan
        self.timespan = self.P.shape[0] ### this can go, right??

        # no need for static parameters

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension
        qsim = torch.tensor([], requires_grad=False) #

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)

        q0 = q0.to(device)
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]

            pet = self.PET[t] ### pet is now w NN output
            ### pet = ds_nn[0,3] ### this may not be the right location to do this

            # GPU
            t = t.to(device)
            p = p.to(device)
            temp = temp.to(device)
            pet = pet.to(device)
            y = y.to(device)

            """ neural network """
            input = torch.cat([
                y[0,0].view(-1, 1),
                y[0,1].view(-1, 1),
                self.P[t].view(-1, 1),
                #self.PET[t].view(-1, 1), ### remove this
                self.T[t].view(-1, 1),
            ], dim=1)


            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)
            ###
            output = self.layer1a(output)
            output = self.activation1a(output)
            ###
            ds_nn = self.layer2(output)

            # GPU
            output = output.to(device)


            # I thinkt this should be automatically on the GPU
            ds_nn = self.layer2(output) ### this is doubles see above and remove

            """ conceptual part """
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            ### replaced by NN
            psnow = ds_nn[0,0]
            prain = ds_nn[0,2]


            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)

            # 2. melt -> in original done with snowbucket -> works
            ### replaced by NN
            melt = ds_nn[0,1]

            # GPU
            melt = melt.to(device)


            # 3. qsub qsurf -> in original done with soilbucket
            ### replaced by NN

            ### NN
            et = ds_nn[0,3]
            qsim_temp = ds_nn[0,4]

            # qsim
            ### replaced by NN
            ### qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ##### qsim = qsim.to(device)

            # dS
            ### I think this is kept anyways
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)

            ### changed here as with NN there is no qsub and qsurf
            ### dS1 = prain + melt - et - qsub - qsurf # water storage
            dS1 = prain + melt - qsim_temp
            ### dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ##### dS0 = dS0.to(device)
            ##### dS1 = dS1.to(device)

            ds_conceptual  = torch.cat([dS0, dS1], axis=0)

            # GPU
            ##### ds_conceptual = ds_conceptual.to(device)

            ### remove NN
            ### ds_combined = ds_nn + ds_conceptual ###


            # GPU
            ##### ds_combined = ds_combined.to(device)

            ### return ds_combined
            return ds_conceptual

        torchdiffeq.odeint(func, q0, t_solver, method='euler') # options=dict(step_size=4)

        return qsim


# In[ ]:


# with RK4
class Hybrid_V1M100RK4(nn.Module):
    """
    In this model the neural network is connected to both storages. IS IT REALLY? WHY NN ONLY 1 OUPT NODE?
    """
    def __init__(self, p, pet, temp):
        super().__init__()
        self.P = p
        self.PET = pet
        self.T = temp

        ### increased nodes per layer to 20
        self.layer1 = nn.Linear(4, 16) ### 4 inputs instead of 5
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(16, 16)
        self.activation1a = nn.LeakyReLU()
        #self.layer1b = nn.Linear(20, 20)
        #self.activation1b = nn.LeakyReLU()
        self.layer2 = nn.Linear(16, 5) ###5 output nodes instead of 2

        """
        nn.init.normal_(self.layer1.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        ###
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        ###
        nn.init.normal_(self.layer2.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)
        """

        # timespan
        self.timespan = self.P.shape[0] ### this can go, right???

        # no need for static parameters

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension
        qsim = torch.tensor([], requires_grad=False) #

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)

        q0 = q0.to(device)
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]

            pet = self.PET[t] ### pet is now w NN output
            ### pet = ds_nn[0,3] ### this may not be the right location to do this

            # GPU
            t = t.to(device)
            p = p.to(device)
            temp = temp.to(device)
            pet = pet.to(device)
            y = y.to(device)

            """ neural network """
            input = torch.cat([
                y[0,0].view(-1, 1),
                y[0,1].view(-1, 1),
                self.P[t].view(-1, 1),
                #self.PET[t].view(-1, 1), ### remove this
                self.T[t].view(-1, 1),
            ], dim=1)


            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)
            ###
            output = self.layer1a(output)
            output = self.activation1a(output)
            ###
            ds_nn = self.layer2(output)

            # GPU
            output = output.to(device)


            # I thinkt this should be automatically on the GPU
            ds_nn = self.layer2(output) ### this is doubles see above and remove

            """ conceptual part """
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            ### replaced by NN
            psnow = ds_nn[0,0]
            prain = ds_nn[0,2]

            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)

            # 2. melt -> in original done with snowbucket -> works
            ### replaced by NN
            melt = ds_nn[0,1]

            # GPU
            melt = melt.to(device)


            # 3. qsub qsurf -> in original done with soilbucket
            ### replaced by NN

            ### NN
            et = ds_nn[0,3]
            qsim_temp = ds_nn[0,4]

            # qsim
            ### replaced by NN
            ### qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ##### qsim = qsim.to(device)

            # dS
            ### I think this is kept anyways
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)

            ### changed here as with NN there is no qsub and qsurf
            ### dS1 = prain + melt - et - qsub - qsurf # water storage
            dS1 = prain + melt - qsim_temp
            ### dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ##### dS0 = dS0.to(device)
            ##### dS1 = dS1.to(device)

            ds_conceptual  = torch.cat([dS0, dS1], axis=0)

            # GPU
            ##### ds_conceptual = ds_conceptual.to(device)

            ### remove NN
            ### ds_combined = ds_nn + ds_conceptual ###


            # GPU
            ##### ds_combined = ds_combined.to(device)

            ### return ds_combined
            return ds_conceptual

        torchdiffeq.odeint(func, q0, t_solver, method='rk4') # options=dict(step_size=4)

        return qsim


# In[ ]:


# changes
# with RK4
class Hybrid_V1M100RK4(nn.Module):
    """
    In this model the neural network is connected to both storages. IS IT REALLY? WHY NN ONLY 1 OUPT NODE?
    """
    def __init__(self, p, temp):
        super().__init__()
        self.P = p
        #self.PET = pet
        self.T = temp

        ### increased nodes per layer to 20
        self.layer1 = nn.Linear(4, 16) ### 4 inputs instead of 5
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(16, 16)
        self.activation1a = nn.LeakyReLU()
        #self.layer1b = nn.Linear(20, 20)
        #self.activation1b = nn.LeakyReLU()
        self.layer2 = nn.Linear(16, 5) ###5 output nodes instead of 2

        """
        nn.init.normal_(self.layer1.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        ###
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        ###
        nn.init.normal_(self.layer2.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)
        """

        # timespan
        self.timespan = self.P.shape[0] ### this can go, right???

        # no need for static parameters

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension
        qsim = torch.tensor([], requires_grad=False) #

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)

        q0 = q0.to(device)
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]

            #pet = self.PET[t] ### pet is now w NN output
            ### pet = ds_nn[0,3] ### this may not be the right location to do this

            # GPU
            t = t.to(device)
            p = p.to(device)
            temp = temp.to(device)
            #pet = pet.to(device)
            y = y.to(device)

            """ neural network """
            input = torch.cat([
                y[0,0].view(-1, 1),
                y[0,1].view(-1, 1),
                self.P[t].view(-1, 1),
                #self.PET[t].view(-1, 1), ### remove this
                self.T[t].view(-1, 1),
            ], dim=1)


            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)
            ###
            output = self.layer1a(output)
            output = self.activation1a(output)
            ###
            ds_nn = self.layer2(output)

            # GPU
            output = output.to(device)


            # I thinkt this should be automatically on the GPU
            ds_nn = self.layer2(output) ### this is doubles see above and remove

            """ conceptual part """
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            ### replaced by NN
            psnow = ds_nn[0,0]
            prain = ds_nn[0,2]

            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)

            # 2. melt -> in original done with snowbucket -> works
            ### replaced by NN
            melt = ds_nn[0,1]

            # GPU
            melt = melt.to(device)


            # 3. qsub qsurf -> in original done with soilbucket
            ### replaced by NN

            ### NN
            et = ds_nn[0,3]
            qsim_temp = ds_nn[0,4]

            # qsim
            ### replaced by NN
            ### qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ##### qsim = qsim.to(device)

            # dS
            ### I think this is kept anyways
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)

            ### changed here as with NN there is no qsub and qsurf
            ### dS1 = prain + melt - et - qsub - qsurf # water storage
            dS1 = prain + melt - qsim_temp
            ### dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ##### dS0 = dS0.to(device)
            ##### dS1 = dS1.to(device)

            ds_conceptual  = torch.cat([dS0, dS1], axis=0)

            # GPU
            ##### ds_conceptual = ds_conceptual.to(device)

            ### remove NN
            ### ds_combined = ds_nn + ds_conceptual ###


            # GPU
            ##### ds_combined = ds_combined.to(device)

            ### return ds_combined
            return ds_conceptual

        torchdiffeq.odeint(func, q0, t_solver, method='rk4') # options=dict(step_size=4)

        return qsim


# In[ ]:


# larger NN
# changes
# with RK4
class Hybrid_V1M100RK4(nn.Module):
    """
    In this model the neural network is connected to both storages. IS IT REALLY? WHY NN ONLY 1 OUPT NODE?
    """
    def __init__(self, p, temp):
        super().__init__()
        self.P = p
        #self.PET = pet
        self.T = temp

        ### increased nodes per layer to 20
        self.layer1 = nn.Linear(4, 32) ### 4 inputs instead of 5
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(32, 32)
        self.activation1a = nn.LeakyReLU()
        self.layer1b = nn.Linear(32, 32)
        self.activation1b = nn.LeakyReLU()
        self.layer2 = nn.Linear(32, 5) ###5 output nodes instead of 2

        """
        nn.init.normal_(self.layer1.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        ###
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        ###
        nn.init.normal_(self.layer2.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)
        """

        # timespan
        self.timespan = self.P.shape[0] ### this can go, right???

        # no need for static parameters

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension
        qsim = torch.tensor([], requires_grad=False) #

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)

        q0 = q0.to(device)
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]

            #pet = self.PET[t] ### pet is now w NN output
            ### pet = ds_nn[0,3] ### this may not be the right location to do this

            # GPU
            t = t.to(device)
            p = p.to(device)
            temp = temp.to(device)
            #pet = pet.to(device)
            y = y.to(device)

            """ neural network """
            input = torch.cat([
                y[0,0].view(-1, 1),
                y[0,1].view(-1, 1),
                self.P[t].view(-1, 1),
                #self.PET[t].view(-1, 1), ### remove this
                self.T[t].view(-1, 1),
            ], dim=1)


            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)
            ###
            output = self.layer1a(output)
            output = self.activation1a(output)
            ###
            ds_nn = self.layer2(output)

            # GPU
            output = output.to(device)


            # I thinkt this should be automatically on the GPU
            ds_nn = self.layer2(output) ### this is doubles see above and remove

            """ conceptual part """
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            ### replaced by NN
            psnow = ds_nn[0,0]
            prain = ds_nn[0,2]

            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)

            # 2. melt -> in original done with snowbucket -> works
            ### replaced by NN
            melt = ds_nn[0,1]

            # GPU
            melt = melt.to(device)


            # 3. qsub qsurf -> in original done with soilbucket
            ### replaced by NN

            ### NN
            et = ds_nn[0,3]
            qsim_temp = ds_nn[0,4]

            # qsim
            ### replaced by NN
            ### qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ##### qsim = qsim.to(device)

            # dS
            ### I think this is kept anyways
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)

            ### changed here as with NN there is no qsub and qsurf
            ### dS1 = prain + melt - et - qsub - qsurf # water storage
            dS1 = prain + melt - qsim_temp
            ### dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ##### dS0 = dS0.to(device)
            ##### dS1 = dS1.to(device)

            ds_conceptual  = torch.cat([dS0, dS1], axis=0)

            # GPU
            ##### ds_conceptual = ds_conceptual.to(device)

            ### remove NN
            ### ds_combined = ds_nn + ds_conceptual ###


            # GPU
            ##### ds_combined = ds_combined.to(device)

            ### return ds_combined
            return ds_conceptual

        torchdiffeq.odeint(func, q0, t_solver, method='rk4') # options=dict(step_size=4)

        return qsim


# #### V2M100 Euler and RK4

# In[ ]:


# For now this is just a copy from V1
class Hybrid_V2M100(nn.Module):
    """
    In this model the neural network is connected to both storages. IS IT REALLY? WHY NN ONLY 1 OUPT NODE?
    """
    def __init__(self, p, et, temp):
        super().__init__()
        self.P = p
        self.ET = et
        self.T = temp

        ### increased nodes per layer to 20
        self.layer1 = nn.Linear(4, 16) ### 4 inputs instead of 5
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(16, 16)
        self.activation1a = nn.LeakyReLU()
        #self.layer1b = nn.Linear(20, 20)
        #self.activation1b = nn.LeakyReLU()
        self.layer2 = nn.Linear(16, 5) ###5 output nodes instead of 2

        nn.init.normal_(self.layer1.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        ###
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        ###
        nn.init.normal_(self.layer2.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)

        # timespan
        self.timespan = self.P.shape[0] ### this can go, right??

        # no need for static parameters

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension
        qsim = torch.tensor([], requires_grad=False) #

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)

        q0 = q0.to(device)
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]

            et = self.ET[t] ### pet is now w NN output
            ### pet = ds_nn[0,3] ### this may not be the right location to do this

            # GPU
            t = t.to(device)
            p = p.to(device)
            temp = temp.to(device)
            et = et.to(device)
            y = y.to(device)

            """ neural network """
            input = torch.cat([
                y[0,0].view(-1, 1),
                y[0,1].view(-1, 1),
                self.P[t].view(-1, 1),
                #self.ET[t].view(-1, 1), ### remove this
                self.T[t].view(-1, 1),
            ], dim=1)


            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)
            ###
            output = self.layer1a(output)
            output = self.activation1a(output)
            ###
            ds_nn = self.layer2(output)

            # GPU
            output = output.to(device)


            # I thinkt this should be automatically on the GPU
            ds_nn = self.layer2(output) ### this is doubles see above and remove

            """ conceptual part """
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            ### replaced by NN
            psnow = ds_nn[0,0]
            prain = ds_nn[0,2]


            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)

            # 2. melt -> in original done with snowbucket -> works
            ### replaced by NN
            melt = ds_nn[0,1]

            # GPU
            melt = melt.to(device)


            # 3. qsub qsurf -> in original done with soilbucket
            ### replaced by NN

            ### NN
            et = ds_nn[0,3] # even though we have et given, we have it here as an output of the NN for better comparability
            qsim_temp = ds_nn[0,4]

            # qsim
            ### replaced by NN
            ### qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ##### qsim = qsim.to(device)

            # dS
            ### I think this is kept anyways
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)

            ### changed here as with NN there is no qsub and qsurf
            ### dS1 = prain + melt - et - qsub - qsurf # water storage
            dS1 = prain + melt - qsim_temp
            ### dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ##### dS0 = dS0.to(device)
            ##### dS1 = dS1.to(device)

            ds_conceptual  = torch.cat([dS0, dS1], axis=0)

            # GPU
            ##### ds_conceptual = ds_conceptual.to(device)

            ### remove NN
            ### ds_combined = ds_nn + ds_conceptual ###


            # GPU
            ##### ds_combined = ds_combined.to(device)

            ### return ds_combined
            return ds_conceptual

        torchdiffeq.odeint(func, q0, t_solver, method='euler') # options=dict(step_size=4)

        return qsim


# In[ ]:


# For now this is just a copy from V1
class Hybrid_V2M100RK4(nn.Module):
    """
    In this model the neural network is connected to both storages. IS IT REALLY? WHY NN ONLY 1 OUPT NODE?
    """
    def __init__(self, p, et, temp):
        super().__init__()
        self.P = p
        self.ET = et
        self.T = temp

        ### increased nodes per layer to 20
        self.layer1 = nn.Linear(4, 16) ### 4 inputs instead of 5
        self.activation1 = nn.Tanh()
        self.layer1a = nn.Linear(16, 16)
        self.activation1a = nn.LeakyReLU()
        #self.layer1b = nn.Linear(20, 20)
        #self.activation1b = nn.LeakyReLU()
        self.layer2 = nn.Linear(16, 5) ###5 output nodes instead of 2

        nn.init.normal_(self.layer1.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1.bias, val=0)
        ###
        nn.init.normal_(self.layer1a.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer1a.bias, val=0)
        ###
        nn.init.normal_(self.layer2.weight, mean=0, std=0.1) # 0.1 -> 0.01
        nn.init.constant_(self.layer2.bias, val=0)

        # timespan
        self.timespan = self.P.shape[0] ### this can go, right??

        # no need for static parameters

    def forward(self, t, q0):
        t_solver = t # not the most elegant solution but it works, this is done as t is changed below and then does not work to pass it to the ODE solver
        t = t.view(-1)  # dimension
        qsim = torch.tensor([], requires_grad=False) #

        # GPU
        t = t.to(device)
        t_solver = t_solver.to(device)

        q0 = q0.to(device)
        qsim = qsim.to(device)

        def func(t, y):
            nonlocal qsim

            t = t.to(torch.long)
            p = self.P[t]
            temp = self.T[t]

            et = self.ET[t] ### pet is now w NN output
            ### pet = ds_nn[0,3] ### this may not be the right location to do this

            # GPU
            t = t.to(device)
            p = p.to(device)
            temp = temp.to(device)
            et = et.to(device)
            y = y.to(device)

            """ neural network """
            input = torch.cat([
                y[0,0].view(-1, 1),
                y[0,1].view(-1, 1),
                self.P[t].view(-1, 1),
                #self.ET[t].view(-1, 1), ### remove this
                self.T[t].view(-1, 1),
            ], dim=1)


            # GPU
            input = input.to(device)

            output = self.layer1(input)
            output = self.activation1(output)
            ###
            output = self.layer1a(output)
            output = self.activation1a(output)
            ###
            ds_nn = self.layer2(output)

            # GPU
            output = output.to(device)


            # I thinkt this should be automatically on the GPU
            ds_nn = self.layer2(output) ### this is doubles see above and remove

            """ conceptual part """
            # 1. prain psnow -> in original done with rainsnowpartition -> works
            ### replaced by NN
            psnow = ds_nn[0,0]
            prain = ds_nn[0,2]


            # GPU
            psnow = psnow.to(device)
            prain = prain.to(device)

            # 2. melt -> in original done with snowbucket -> works
            ### replaced by NN
            melt = ds_nn[0,1]

            # GPU
            melt = melt.to(device)


            # 3. qsub qsurf -> in original done with soilbucket
            ### replaced by NN

            ### NN
            et = ds_nn[0,3] # even though we have et given, we have it here as an output of the NN for better comparability
            qsim_temp = ds_nn[0,4]

            # qsim
            ### replaced by NN
            ### qsim_temp = qsub + qsurf
            qsim_temp = torch.unsqueeze(qsim_temp, 0)

            # GPU
            qsim_temp = qsim_temp.to(device)

            qsim = torch.cat((qsim, qsim_temp), dim=0)

            # GPU
            ##### qsim = qsim.to(device)

            # dS
            ### I think this is kept anyways
            dS0 = psnow - melt # snow storage
            dS0 = torch.unsqueeze(dS0, 0)

            ### changed here as with NN there is no qsub and qsurf
            ### dS1 = prain + melt - et - qsub - qsurf # water storage
            dS1 = prain + melt - qsim_temp
            ### dS1 = torch.unsqueeze(dS1, 0)

            # GPU
            ##### dS0 = dS0.to(device)
            ##### dS1 = dS1.to(device)

            ds_conceptual  = torch.cat([dS0, dS1], axis=0)

            # GPU
            ##### ds_conceptual = ds_conceptual.to(device)

            ### remove NN
            ### ds_combined = ds_nn + ds_conceptual ###


            # GPU
            ##### ds_combined = ds_combined.to(device)

            ### return ds_combined
            return ds_conceptual

        torchdiffeq.odeint(func, q0, t_solver, method='rk4') # options=dict(step_size=4)

        return qsim


# ## 1.4 LSTM Model

# In[ ]:


# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)  # New layer
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(hidden_size, output_size)
        self.leakyrelu = nn.LeakyReLU() # to prevent outputs below 0 ?

    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout(lstm_out1)
        lstm_out2, _ = self.lstm2(lstm_out1)  # Pass the output of the first layer to the second layer
        lstm_out2 = self.dropout(lstm_out2)
        y_pred = self.linear(lstm_out2[:, -1, :])
        y_pred = self.leakyrelu(y_pred)
        return y_pred


# # 2. Functions

# ## 2.1 General functions
# 

# ### 2.1.1 Run model and return running time

# In[ ]:


def run_model(model):
    start_time = time.time()
    model = model.simulate()
    end_time = time.time()

    execution_time = end_time - start_time
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)

    print("Execution time: {} minutes {} seconds".format(minutes, seconds))
    return model


# ## 2.2 Data functions

# ### 2.2.1 Turn dataframe columns into torch tensor
# 

# In[ ]:


def df_to_tensors(df, name_suffix=""):
    tensor_dict = {}
    for column in df.columns:
        if column != 'index':  # skip index
            new_column_name = f"{column}_{name_suffix}" if name_suffix else column
            tensor = torch.tensor(df[column].values, dtype=torch.float32)
            tensor_dict[new_column_name] = tensor

    return tensor_dict


# ## 2.3 Parameter functions

# In[ ]:


def load_parameters(folder_path, ID, method):
    filename_pattern = f"static_parameters_ID_{ID}_method_{method}.npy"
    matching_files = [file for file in os.listdir(folder_path) if filename_pattern in file]
    if len(matching_files) == 0:
        print("No matching files found.")
        return None
    selected_file = matching_files[0]
    file_path = os.path.join(folder_path, selected_file)
    parameters = np.load(file_path)
    return parameters


# ## 2.4 Other

# In[ ]:


# for down sampling, tensor A is the "reference" so tensor B will be adjusted in size to tensor A
def interpolate_downsample(tensor_A, tensor_B):
    step_size = (len(tensor_B) - 1) / (len(tensor_A) - 1)

    downsampled_B = torch.empty(len(tensor_A))


    for i in range(len(tensor_A)):
        index_in_B = i * step_size
        lower_index = int(index_in_B)
        upper_index = min(lower_index + 1, len(tensor_B) - 1)
        alpha = index_in_B - lower_index

        downsampled_B[i] = (1 - alpha) * tensor_B[lower_index] + alpha * tensor_B[upper_index]

    return downsampled_B


# # 3. Data
# 

# Functions currently in script:
# 
# 
# *   load_data
# *   prep_data
# *   loss_functions

# ### 3.0 Select random catchment IDs from the clusters 💀 (This is not needed anymore as we have selected representative catchments for each cluster)
# Select only from the catchment in IDs_red, which includes only the catchments that have gap-free runoff timeseries from 1981 - to 2017. This should be 246 catchments. Note that the clustering was performed with all catchments so we are selecting a subset of the catchments now.
# 

# In[ ]:


unique_clusters = catchment_attributes_red['cluster_10vars_8cl'].unique()

# dictionary
cluster_id_dict = {cluster: catchment_attributes_red[catchment_attributes_red['cluster_10vars_8cl'] == cluster]['ID'].to_numpy() for cluster in unique_clusters}


# In[ ]:


# arrays with IDs of the respective clusters
cluster_0 = cluster_id_dict[0]
cluster_1 = cluster_id_dict[1]
cluster_2 = cluster_id_dict[2]
cluster_3 = cluster_id_dict[3]
cluster_4 = cluster_id_dict[4]
cluster_5 = cluster_id_dict[5]
#cluster_6 = cluster_id_dict[6] # this one is not used as it contains only one catchment and this one is very large! If using the subset the cluster will be empty as the catchments runoff timeseries contains gaps
cluster_7 = cluster_id_dict[7]


# In[ ]:


# seed for reproducability
seed_value = 12345
np.random.seed(seed_value)

# dict
cluster_dict = {
    0: cluster_0,
    1: cluster_1,
    2: cluster_2,
    3: cluster_3,
    4: cluster_4,
    5: cluster_5,
    7: cluster_7
}

# empty array to store the random catchment IDs
random_catchments = []

# print the random IDs
for cluster_value, cluster_ids in cluster_dict.items():
    random_catchment = np.random.choice(cluster_ids)
    random_catchments.append(random_catchment)
    print(f"Random value for catchment cluster {cluster_value}: {random_catchment}")


# In[ ]:


# arrays for the respective clusters that contain all catchment IDs of that cluster
cluster_0


# In[ ]:


# compute the number of catchments in each cluster (when using the subset of complete runoff timeseries)
print("Cluster 0:", len(cluster_0))
print("Cluster 1:", len(cluster_1))
print("Cluster 2:", len(cluster_2))
print("Cluster 3:", len(cluster_3))
print("Cluster 4:", len(cluster_4))
print("Cluster 5:", len(cluster_5))
#print("Cluster 6:", len(cluster_6))
print("Cluster 7:", len(cluster_7))


# ### Create calibration periods corresponding to training, validation and testing periods

# **Note:**
# 
# - The reference date is now also set to 1981-10-01, so same as start_cal and start_train, this means no shift for the hydrological year is needed anymore as we will just load data from that point in time
# 
# - The calculation of the periods below does not include the last day, therefore we add +1 day

# In[ ]:


# adjusted for hydrological year
reference_date = dt.datetime.strptime(reference, "%Y-%m-%d") # this is now set to match start_train and start_cal, hence has no impact in this setup

# train
start_train_date = dt.datetime.strptime(start_train, "%Y-%m-%d")
start_train_short_date = dt.datetime.strptime(start_train_short, "%Y-%m-%d")
stop_train_date = dt.datetime.strptime(stop_train, "%Y-%m-%d")

# cal
start_cal_date = dt.datetime.strptime(start_cal, "%Y-%m-%d")
stop_cal_date = dt.datetime.strptime(stop_cal, "%Y-%m-%d")

# val
start_val_date = dt.datetime.strptime(start_val, "%Y-%m-%d")
stop_val_date = dt.datetime.strptime(stop_val, "%Y-%m-%d")

# test
start_test_date = dt.datetime.strptime(start_test, "%Y-%m-%d")
stop_test_date = dt.datetime.strptime(stop_test, "%Y-%m-%d")

"""
# Calculate the number of days between the two dates
cal_period = np.arange((start_train_date - reference_date).days, (stop_cal_date - start_cal_date).days + 1)

# for the hybrid model training we create a short calibration periods so that the training then also only takes place after the calibration interval
train_period = np.arange((start_train_date - reference_date).days, (stop_train_date - start_train_date).days + 1)

train_period_short = np.arange((start_train_short_date - start_train_date).days, (stop_train_date - start_train_date).days + 1)
val_period = np.arange((start_val_date - start_train_date).days, (stop_val_date - start_train_date).days + 1)
test_period = np.arange((start_test_date - start_train_date).days, (stop_test_date - start_train_date).days + 1)

### changed
full_period = np.arange((start_train_date - reference_date).days, (stop_test_date - start_cal_date).days + 1)
"""

# Calculate the number of days between the two dates
cal_period = np.arange((start_train_date - reference_date).days, (stop_cal_date - start_cal_date).days)

# for the hybrid model training we create a short calibration periods so that the training then also only takes place after the calibration interval
train_period = np.arange((start_train_date - reference_date).days, (stop_train_date - start_train_date).days)

train_period_short = np.arange((start_train_short_date - start_train_date).days, (stop_train_date - start_train_date).days)
val_period = np.arange((start_val_date - start_train_date).days, (stop_val_date - start_train_date).days)
test_period = np.arange((start_test_date - start_train_date).days, (stop_test_date - start_train_date).days)

### changed
full_period = np.arange((start_train_date - reference_date).days, (stop_test_date - start_cal_date).days)

print(cal_period)
print(train_period_short)
print(train_period)
print(val_period)
print(test_period)
print(full_period) # starting from the hydrolgoical year (1981-10-01) to 2017-09-30


# \#### ⏰ Maybe they should be type "long" instead! (Clean up here)

# In[ ]:


# turn those periods into tensors
cal_period_int = torch.from_numpy(cal_period).int()
train_period_short_int = torch.from_numpy(train_period_short).int()
train_period_int = torch.from_numpy(train_period).int()
val_period_int = torch.from_numpy(val_period).int()
test_period_int = torch.from_numpy(test_period).int()
full_period_int = torch.from_numpy(full_period).int()


# In[ ]:


# turn those periods into tensors
cal_period = torch.from_numpy(cal_period).float()
train_period_short = torch.from_numpy(train_period_short).float()
train_period = torch.from_numpy(train_period).float()
val_period = torch.from_numpy(val_period).float()
test_period = torch.from_numpy(test_period).float()
full_period = torch.from_numpy(full_period).float()


# # 4. Model Runs
# Note: calibration of conceptual model was performed using the notebook "my_exp_hydro_patil.ipynb"

# ## 4.1 Conceptual V1 - (E-OBS)
# Remember, V1 takes (p, pet, temp) - EOBS
# V2 takes (p, et, temp) - LamaH
# 
# 4.1.1: ConceptualV1Cal
# 
# 4.1.2: ConceptualV1 Single Run
# 
# 

# In[ ]:


results_dfs = []

for id_num in IDs_red:
    print("ID:", id_num)

    # define initial storages
    initial_storages = torch.tensor([0.0, 0.0])

    # load df cal
    if environment == "colab":
        df_cal = load_data_new(path_vars_eobs_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_eobs, start_date=start_cal, stop_date=stop_cal, dataset="eobs", qobs_mm=True)
    if environment == "local":
        df_cal = load_data_new(path_vars_eobs_local, path_qobs_lamah_local, path_catchment_attrs_colab, ID=id_num, variables=variables_eobs, start_date=start_cal, stop_date=stop_cal, dataset="eobs", qobs_mm=True)

    # create tensors from the dataframe
    tensors_cal = df_to_tensors(df_cal, name_suffix="cal")

    # Processing tensors
    for name, tensor in tensors_cal.items():
        # Moving to GPU
        tensor = tensor.to(device)
        tensor = tensor.to(dtype=torch.float32)
        tensor.requires_grad = False  # inplace operation
        locals()[name] = tensor

    # load df test
    if environment == "colab":
        df_test = load_data_new(path_vars_eobs_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_eobs, start_date=start_test, stop_date=stop_test, dataset="eobs", qobs_mm=True)
    if environment == "local":
        df_cal = load_data_new(path_vars_eobs_local, path_qobs_lamah_local, path_catchment_attrs_colab, ID=id_num, variables=variables_eobs, start_date=start_test, stop_date=stop_test, dataset="eobs", qobs_mm=True)

    # create tensors from the dataframe
    tensors_test = df_to_tensors(df_test, name_suffix="test")

    # Processing tensors
    for name, tensor in tensors_test.items():
        # Moving to GPU
        tensor = tensor.to(device)
        tensor = tensor.to(dtype=torch.float32)
        tensor.requires_grad = False  # inplace operation
        locals()[name] = tensor

    # load the parameters for the model
    if environment == "colab":
        path = os.path.join(path_conceptual_v1_colab, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)
    if environment == "local":
        path = os.path.join(path_conceptual_v1_local, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)

    # instantiate the models calibration period
    model_euler_cal = ConceptualV1EULER(prcp_cal.cpu(), pet_cal.cpu(), tmean_cal.cpu(), qobs_mm_cal.cpu(), calibrated_parameters, initial_storages)
    model_rk4_cal = ConceptualV1RK4(prcp_cal.cpu(), pet_cal.cpu(), tmean_cal.cpu(), qobs_mm_cal.cpu(), calibrated_parameters, initial_storages)

    # run the models calibration period
    qsim_euler_cal = run_model(model_euler_cal)
    qsim_rk4_cal = run_model(model_rk4_cal)

    # instantiate the models test period
    model_euler_test = ConceptualV1EULER(prcp_test.cpu(), pet_test.cpu(), tmean_test.cpu(), qobs_mm_test.cpu(), calibrated_parameters, initial_storages)
    model_rk4_test = ConceptualV1RK4(prcp_test.cpu(), pet_test.cpu(), tmean_test.cpu(), qobs_mm_test.cpu(), calibrated_parameters, initial_storages)

    # run the models test period
    qsim_euler_test = run_model(model_euler_test)
    qsim_rk4_test = run_model(model_rk4_test)

    # save the qsim timeseries for cal and test
    if environment == "colab":
        path_euler_cal = os.path.join(path_conceptual_v1_colab, "discharge/cal/euler/", f"qsim_cal_euler_ID_{id_num}.csv")
        path_euler_test = os.path.join(path_conceptual_v1_colab, "discharge/test/euler/", f"qsim_test_euler_ID_{id_num}.csv")
        path_rk4_cal = os.path.join(path_conceptual_v1_colab, "discharge/cal/rk4/", f"qsim_cal_rk4_ID_{id_num}.csv")
        path_rk4_test = os.path.join(path_conceptual_v1_colab, "discharge/test/rk4/", f"qsim_test_rk4_ID_{id_num}.csv")
    if environment == "local":
        path_euler_cal = os.path.join(path_conceptual_v1_local, "discharge/cal/euler/", f"qsim_cal_euler_ID_{id_num}.csv")
        path_euler_test = os.path.join(path_conceptual_v1_local, "discharge/test/euler/", f"qsim_test_euler_ID_{id_num}.csv")
        path_rk4_cal = os.path.join(path_conceptual_v1_local, "discharge/cal/rk4/", f"qsim_cal_rk4_ID_{id_num}.csv")
        path_rk4_test = os.path.join(path_conceptual_v1_local, "discharge/test/rk4/", f"qsim_test_rk4_ID_{id_num}.csv")

    np.savetxt(path_euler_cal, qsim_euler_cal, delimiter=",")
    np.savetxt(path_euler_test, qsim_euler_test, delimiter=",")
    np.savetxt(path_rk4_cal, qsim_rk4_cal, delimiter=",")
    np.savetxt(path_rk4_test, qsim_rk4_test, delimiter=",")

    # save the observation timeseries
    if environment == "colab":
        path_qobs_mm_cal = os.path.join(path_conceptual_v1_colab, "discharge/cal/observations/", f"qobs_mm_cal_ID_{id_num}.csv")
        path_qobs_mm_test = os.path.join(path_conceptual_v1_colab, "discharge/test/observations/", f"qobs_mm_test_ID_{id_num}.csv")

    if environment == "local":
        path_qobs_mm_cal = os.path.join(path_conceptual_v1_local, "discharge/cal/observations/", f"qobs_mm_cal_ID_{id_num}.csv")
        path_qobs_mm_test = os.path.join(path_conceptual_v1_local, "discharge/test/observations/", f"qobs_mm_test_ID_{id_num}.csv")

    np.savetxt(path_qobs_mm_cal, qobs_mm_cal, delimiter=",")
    np.savetxt(path_qobs_mm_test, qobs_mm_test, delimiter=",")


    # calculate evaluation metrics BUT leave 365 days for warm-up
    nse_loss = NSELoss()
    mse_loss = MSELoss()
    kge_loss = KGELoss()

    # loss calibration
    nse_loss_euler_cal = nse_loss(qsim_euler_cal[365:], qobs_mm_cal[365:].cpu())
    mse_loss_euler_cal = mse_loss(qsim_euler_cal[365:], qobs_mm_cal[365:].cpu())
    kge_loss_euler_cal = kge_loss(qsim_euler_cal[365:], qobs_mm_cal[365:].cpu())

    nse_loss_rk4_cal = nse_loss(qsim_rk4_cal[365:], qobs_mm_cal[365:].cpu())
    mse_loss_rk4_cal = mse_loss(qsim_rk4_cal[365:], qobs_mm_cal[365:].cpu())
    kge_loss_rk4_cal = kge_loss(qsim_rk4_cal[365:], qobs_mm_cal[365:].cpu())
    print(f"Calibration losses (RK4), NSE: {nse_loss_rk4_cal}, MSE: {mse_loss_rk4_cal}, KGE: {kge_loss_rk4_cal}")

    # loss test
    nse_loss_euler_test = nse_loss(qsim_euler_test[365:], qobs_mm_test[365:].cpu())
    mse_loss_euler_test = mse_loss(qsim_euler_test[365:], qobs_mm_test[365:].cpu())
    kge_loss_euler_test = kge_loss(qsim_euler_test[365:], qobs_mm_test[365:].cpu())

    nse_loss_rk4_test = nse_loss(qsim_rk4_test[365:], qobs_mm_test[365:].cpu())
    mse_loss_rk4_test = mse_loss(qsim_rk4_test[365:], qobs_mm_test[365:].cpu())
    kge_loss_rk4_test = kge_loss(qsim_rk4_test[365:], qobs_mm_test[365:].cpu())
    print(f"Test losses (RK4), NSE: {nse_loss_rk4_test}, MSE: {mse_loss_rk4_test}, KGE: {kge_loss_rk4_test}")

    # save evaluation metrics (ideally to one large file )
    row = {
        "id_num": id_num,
        "start calibration": start_cal,
        "stop calibration": stop_cal,
        "start test": start_test,
        "stop test": stop_test,
        "nse_loss_euler_cal": nse_loss_euler_cal,
        "mse_loss_euler_cal": mse_loss_euler_cal,
        "kge_loss_euler_cal": kge_loss_euler_cal,
        "nse_loss_rk4_cal": nse_loss_rk4_cal,
        "mse_loss_rk4_cal": mse_loss_rk4_cal,
        "kge_loss_rk4_cal": kge_loss_rk4_cal,
        "nse_loss_euler_test": nse_loss_euler_test,
        "mse_loss_euler_test": mse_loss_euler_test,
        "kge_loss_euler_test": kge_loss_euler_test,
        "nse_loss_rk4_test": nse_loss_rk4_test,
        "mse_loss_rk4_test": mse_loss_rk4_test,
        "kge_loss_rk4_test": kge_loss_rk4_test
    }

    # Create a DataFrame for the current iteration
    iteration_df = pd.DataFrame([row])

    # Append the iteration DataFrame to the list
    results_dfs.append(iteration_df)

# Concatenate all DataFrames into one
results_df = pd.concat(results_dfs, ignore_index=True)


# Save the results DataFrame to a CSV file
if environment == "colab":
    path_results = os.path.join(path_conceptual_v1_colab, "evaluation_metrics/", f"conceptual_v1_results.csv")

if environment == "local":
    path_results = os.path.join(path_conceptual_v1_local, "evaluation_metrics/", f"conceptual_v1_results.csv")

results_df.to_csv(path_results, index=False)


# ## 4.2 Conceptual V2 - LamaH

# In[ ]:


results_dfs = []

for id_num in IDs_red:
    print("ID:", id_num)

    # define initial storages
    initial_storages = torch.tensor([0.0, 0.0])

    # load df cal
    if environment == "colab":
        df_cal = load_data_new(path_vars_lamah_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_lamah, start_date=start_cal, stop_date=stop_cal, qobs_mm=True)
    if environment == "local":
        df_cal = load_data_new(path_vars_lamah_local, path_qobs_lamah_local, path_catchment_attrs_colab, ID=id_num, variables=variables_lamah, start_date=start_cal, stop_date=stop_cal, qobs_mm=True)

    # create tensors from the dataframe
    tensors_cal = df_to_tensors(df_cal, name_suffix="cal")

    # Processing tensors
    for name, tensor in tensors_cal.items():
        # Moving to GPU
        tensor = tensor.to(device)
        tensor = tensor.to(dtype=torch.float32)
        tensor.requires_grad = False  # inplace operation
        locals()[name] = tensor

    # load df test
    if environment == "colab":
        df_test = load_data_new(path_vars_lamah_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_lamah, start_date=start_test, stop_date=stop_test, qobs_mm=True)
    if environment == "local":
        df_cal = load_data_new(path_vars_lamah_local, path_qobs_lamah_local, path_catchment_attrs_colab, ID=id_num, variables=variables_lamah, start_date=start_test, stop_date=stop_test, qobs_mm=True)

    # create tensors from the dataframe
    tensors_test = df_to_tensors(df_test, name_suffix="test")

    # Processing tensors
    for name, tensor in tensors_test.items():
        # Moving to GPU
        tensor = tensor.to(device)
        tensor = tensor.to(dtype=torch.float32)
        tensor.requires_grad = False  # inplace operation
        locals()[name] = tensor

    # load the parameters for the model
    if environment == "colab":
        path = os.path.join(path_conceptual_v2_colab, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)
    if environment == "local":
        path = os.path.join(path_conceptual_v2_local, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)

    # instantiate the models calibration period
    model_euler_cal = ConceptualV2EULER(prcp_cal.cpu(), et_cal.cpu(), tmean_cal.cpu(), qobs_mm_cal.cpu(), calibrated_parameters, initial_storages)
    model_rk4_cal = ConceptualV2RK4(prcp_cal.cpu(), et_cal.cpu(), tmean_cal.cpu(), qobs_mm_cal.cpu(), calibrated_parameters, initial_storages)

    # run the models calibration period
    qsim_euler_cal = run_model(model_euler_cal)
    qsim_rk4_cal = run_model(model_rk4_cal)

    # instantiate the models test period
    model_euler_test = ConceptualV2EULER(prcp_test.cpu(), et_test.cpu(), tmean_test.cpu(), qobs_mm_test.cpu(), calibrated_parameters, initial_storages)
    model_rk4_test = ConceptualV2RK4(prcp_test.cpu(), et_test.cpu(), tmean_test.cpu(), qobs_mm_test.cpu(), calibrated_parameters, initial_storages)

    # run the models test period
    qsim_euler_test = run_model(model_euler_test)
    qsim_rk4_test = run_model(model_rk4_test)

    # save the qsim timeseries for cal and test
    if environment == "colab":
        path_euler_cal = os.path.join(path_conceptual_v2_colab, "discharge/cal/euler/", f"qsim_cal_euler_ID_{id_num}.csv")
        path_euler_test = os.path.join(path_conceptual_v2_colab, "discharge/test/euler/", f"qsim_test_euler_ID_{id_num}.csv")
        path_rk4_cal = os.path.join(path_conceptual_v2_colab, "discharge/cal/rk4/", f"qsim_cal_rk4_ID_{id_num}.csv")
        path_rk4_test = os.path.join(path_conceptual_v2_colab, "discharge/test/rk4/", f"qsim_test_rk4_ID_{id_num}.csv")
    if environment == "local":
        path_euler_cal = os.path.join(path_conceptual_v2_local, "discharge/cal/euler/", f"qsim_cal_euler_ID_{id_num}.csv")
        path_euler_test = os.path.join(path_conceptual_v2_local, "discharge/test/euler/", f"qsim_test_euler_ID_{id_num}.csv")
        path_rk4_cal = os.path.join(path_conceptual_v2_local, "discharge/cal/rk4/", f"qsim_cal_rk4_ID_{id_num}.csv")
        path_rk4_test = os.path.join(path_conceptual_v2_local, "discharge/test/rk4/", f"qsim_test_rk4_ID_{id_num}.csv")

    np.savetxt(path_euler_cal, qsim_euler_cal, delimiter=",")
    np.savetxt(path_euler_test, qsim_euler_test, delimiter=",")
    np.savetxt(path_rk4_cal, qsim_rk4_cal, delimiter=",")
    np.savetxt(path_rk4_test, qsim_rk4_test, delimiter=",")

    # save the observation timeseries
    if environment == "colab":
        path_qobs_mm_cal = os.path.join(path_conceptual_v2_colab, "discharge/cal/observations/", f"qobs_mm_cal_ID_{id_num}.csv")
        path_qobs_mm_test = os.path.join(path_conceptual_v2_colab, "discharge/test/observations/", f"qobs_mm_test_ID_{id_num}.csv")

    if environment == "local":
        path_qobs_mm_cal = os.path.join(path_conceptual_v2_local, "discharge/cal/observations/", f"qobs_mm_cal_ID_{id_num}.csv")
        path_qobs_mm_test = os.path.join(path_conceptual_v2_local, "discharge/test/observations/", f"qobs_mm_test_ID_{id_num}.csv")

    np.savetxt(path_qobs_mm_cal, qobs_mm_cal, delimiter=",")
    np.savetxt(path_qobs_mm_test, qobs_mm_test, delimiter=",")


    # calculate evaluation metrics BUT leave 365 days for warm-up
    nse_loss = NSELoss()
    mse_loss = MSELoss()
    kge_loss = KGELoss()

    # loss calibration
    nse_loss_euler_cal = nse_loss(qsim_euler_cal[365:], qobs_mm_cal[365:].cpu())
    mse_loss_euler_cal = mse_loss(qsim_euler_cal[365:], qobs_mm_cal[365:].cpu())
    kge_loss_euler_cal = kge_loss(qsim_euler_cal[365:], qobs_mm_cal[365:].cpu())

    nse_loss_rk4_cal = nse_loss(qsim_rk4_cal[365:], qobs_mm_cal[365:].cpu())
    mse_loss_rk4_cal = mse_loss(qsim_rk4_cal[365:], qobs_mm_cal[365:].cpu())
    kge_loss_rk4_cal = kge_loss(qsim_rk4_cal[365:], qobs_mm_cal[365:].cpu())
    print(f"Calibration losses (RK4), NSE: {nse_loss_rk4_cal}, MSE: {mse_loss_rk4_cal}, KGE: {kge_loss_rk4_cal}")

    # loss test
    nse_loss_euler_test = nse_loss(qsim_euler_test[365:], qobs_mm_test[365:].cpu())
    mse_loss_euler_test = mse_loss(qsim_euler_test[365:], qobs_mm_test[365:].cpu())
    kge_loss_euler_test = kge_loss(qsim_euler_test[365:], qobs_mm_test[365:].cpu())

    nse_loss_rk4_test = nse_loss(qsim_rk4_test[365:], qobs_mm_test[365:].cpu())
    mse_loss_rk4_test = mse_loss(qsim_rk4_test[365:], qobs_mm_test[365:].cpu())
    kge_loss_rk4_test = kge_loss(qsim_rk4_test[365:], qobs_mm_test[365:].cpu())
    print(f"Test losses (RK4), NSE: {nse_loss_rk4_test}, MSE: {mse_loss_rk4_test}, KGE: {kge_loss_rk4_test}")

    # save evaluation metrics (ideally to one large file )
    row = {
        "id_num": id_num,
        "start calibration": start_cal,
        "stop calibration": stop_cal,
        "start test": start_test,
        "stop test": stop_test,
        "nse_loss_euler_cal": nse_loss_euler_cal,
        "mse_loss_euler_cal": mse_loss_euler_cal,
        "kge_loss_euler_cal": kge_loss_euler_cal,
        "nse_loss_rk4_cal": nse_loss_rk4_cal,
        "mse_loss_rk4_cal": mse_loss_rk4_cal,
        "kge_loss_rk4_cal": kge_loss_rk4_cal,
        "nse_loss_euler_test": nse_loss_euler_test,
        "mse_loss_euler_test": mse_loss_euler_test,
        "kge_loss_euler_test": kge_loss_euler_test,
        "nse_loss_rk4_test": nse_loss_rk4_test,
        "mse_loss_rk4_test": mse_loss_rk4_test,
        "kge_loss_rk4_test": kge_loss_rk4_test
    }

    # Create a DataFrame for the current iteration
    iteration_df = pd.DataFrame([row])

    # Append the iteration DataFrame to the list
    results_dfs.append(iteration_df)

# Concatenate all DataFrames into one
results_df = pd.concat(results_dfs, ignore_index=True)


# Save the results DataFrame to a CSV file
if environment == "colab":
    path_results = os.path.join(path_conceptual_v2_colab, "evaluation_metrics/", f"conceptual_v2_results.csv")

if environment == "local":
    path_results = os.path.join(path_conceptual_v2_local, "evaluation_metrics/", f"conceptual_v2_results.csv")

results_df.to_csv(path_results, index=False)


# ## 4.3 Hybrid Models

# ### 4.3.1 Hybrid V1 - E-OBS
# 
# 
# 

# #### Run on the 6 Representative Catchments 
# 
# 

# In[ ]:


# user input
num_epochs = 50
solver = "euler" # or "rk4"
loss_func = "MSE" # "MSE" or "RMSE" or if used "NSE", "KGE"
start_lr = 0.01 # 0.01


# In[ ]:


# representative catchment IDs for E-OBS
IDs_red = [241, 215, 581, 21, 277, 797, 432] # ID 572 is closest to overall median 


# In[ ]:


# TRAINING THE 6 REPRESENTATIVE CATCHMNETS
# loop for everything
results_dfs = []

# loop that rules it all
for id_num in IDs_red:
    print("ID:", id_num)

    loss_func = loss_func

    # define initial storages
    S0_hybrid_v1 = torch.tensor([0.0, 0.0])
    S0_hybrid_v1 = torch.unsqueeze(S0_hybrid_v1, 0) # reshape
    # push to GPU
    S0_hybrid_v1 = S0_hybrid_v1.to(device)

    # load the parameters for the model
    if environment == "colab":
        path = os.path.join(path_conceptual_v1_colab, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)
    if environment == "local":
        path = os.path.join(path_conceptual_v1_local, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)

    # load df hybrid # IMPORTANT: full period from start train to stop test!
    if environment == "colab":
        df_hybrid = load_data_new(path_vars_eobs_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_eobs, start_date=start_train, stop_date=stop_test, dataset="eobs", qobs_mm=True)
    if environment == "local":
        df_hybrid = load_data_new(path_vars_eobs_local, path_qobs_lamah_local, path_catchment_attrs_colab, ID=id_num, variables=variables_eobs, start_date=start_train, stop_date=stop_test,  dataset="eobs", qobs_mm=True)

    # create tensors from the dataframe
    tensors_hybrid = df_to_tensors(df_hybrid, name_suffix="hybrid")

    # Processing tensors
    for name, tensor in tensors_hybrid.items():
        # Moving to GPU
        tensor = tensor.to(device)
        tensor = tensor.to(dtype=torch.float32)
        tensor.requires_grad = False  # inplace operation
        locals()[name] = tensor

    # instantiate model
    hybridmodel_v1 = Hybrid_V1(prcp_hybrid, pet_hybrid, tmean_hybrid, calibrated_parameters) ### note vor V2 we need et here instead!
    # hybridmodel_v1.to(device) # put back if using GPU

    # loss function
    if loss_func == "MSE":
        criterion = nn.MSELoss()
        print("Using MSE loss func")
    elif loss_func == "RMSE":
        criterion = RMSLELoss()
        print("Using RMSE loss func")
    elif loss_func == "KGE":
        criterion = KGELossMOD()
        print("Using KGE loss func")
    else:
        print("no valid loss func defined")

    # Define the optimizer, which optimzer should be used? Adam? SGD?
    optimizer = optim.Adam(hybridmodel_v1.parameters(), lr=start_lr) # what is a good lr to start with?


    # learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=7, verbose=True)

    torch.autograd.set_detect_anomaly(True)

    train_losses_v1 = []
    val_losses_v1 = []

    best_loss = 50000000
    best_model_state_hybrid_v1 = None ### what was this for again?

    # Train the model
    for epoch in range(num_epochs):

        # params
        params_hybrid_v1 = hybridmodel_v1.state_dict()

        # forward pass training data
        # qsim_train_hybrid_v1 = hybridmodel_v1.forward(train_period, S0_hybrid_v1)
        qsim_train_hybrid_v1 = hybridmodel_v1.forward(train_period, S0_hybrid_v1)

        # forward pass on validation data
        with torch.no_grad():
            #qsim_val_hybrid_v1 = hybridmodel_v1.forward(val_period, S0_hybrid_v1)
            qsim_val_hybrid_v1 = hybridmodel_v1.forward(val_period, S0_hybrid_v1)


        # Added warmup period here
        # [-1] for qobs_mm is a fix here as qsim is one element too short because of the way it is calculated and appended
        train_loss = criterion(qsim_train_hybrid_v1[365:], qobs_mm_hybrid[train_period_int[365:-1]]) # make sure to NOT train on normalized observations!
        val_loss = criterion(qsim_val_hybrid_v1[365:], qobs_mm_hybrid[val_period_int[365:-1]]) # make sure to NOT train on normalized observations!

        # append train and validation loss
        train_losses_v1.append(train_loss.item())
        val_losses_v1.append(val_loss.item())


        # backward pass
        optimizer.zero_grad()
        train_loss.backward() #retain_graph=True

        print(hybridmodel_v1.layer1.weight.grad)

        optimizer.step()

        # update lr schedule
        scheduler.step(train_loss) # train loss

        # update and save best model (based on validation loss)
        if val_loss.item() < best_loss:
            print("Accessed loop")
            best_loss = val_loss.item()
            best_model_state_hybrid_v1 = copy.deepcopy(hybridmodel_v1.state_dict())


        print(f"Epoch {epoch}: Train loss = {train_loss.item()} Validation loss = {val_loss.item()}")


    # save trained model parameters
    if environment=='local':
        torch.save(best_model_state_hybrid_v1, f'models/hybrid_v1_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{start_lr}.pth')
    if environment=='colab':
        torch.save(best_model_state_hybrid_v1, f'/content/drive/MyDrive/msc_thesis/python/notebooks/colab/HybridV1/trained_parameters/hybrid_v1_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{start_lr}_epoch_{num_epochs}.pth')


# In[ ]:


# user input to generate the results
num_epochs = 50
solver = "euler" # or "rk4"
loss_func = "MSE" # "MSE" or "RMSE" or if used "NSE", "KGE"
start_lr = 0.01 # 0.01, 0.005


# In[ ]:


# representative catchemnts E-OBS (V1)
IDs_red = [241, 215, 581, 21, 277, 797, 432, 572] # added 572 here so that we can skip the code above


# In[ ]:


# RUN AND SAVE THE MODELS WITH THE TRAINED PARAMETERS
results_hybrid_dfs = []

# loop that rules it all
for id_num in IDs_red:
    print("ID:", id_num)

    # define initial storages
    S0_hybrid_v1 = torch.tensor([0.0, 0.0])
    S0_hybrid_v1 = torch.unsqueeze(S0_hybrid_v1, 0) # reshape

    # load the data
    # load df hybrid # IMPORTANT: full period from start train to stop test!
    if environment == "colab":
        df_hybrid = load_data_new(path_vars_eobs_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_eobs, start_date=start_train, stop_date=stop_test, dataset="eobs", qobs_mm=True)
    if environment == "local":
        df_hybrid = load_data_new(path_vars_eobs_local, path_qobs_lamah_local, path_catchment_attrs_colab, ID=id_num, variables=variables_eobs, start_date=start_train, stop_date=stop_test,  dataset="eobs", qobs_mm=True)


    # create tensors from the dataframe
    tensors_hybrid = df_to_tensors(df_hybrid, name_suffix="hybrid")
    # Processing tensors
    for name, tensor in tensors_hybrid.items():
        # Moving to GPU
        tensor = tensor.to(device)
        tensor = tensor.to(dtype=torch.float32)
        tensor.requires_grad = False  # inplace operation
        locals()[name] = tensor


    # load the parameters for the conceptual model part
    if environment == "colab":
        path = os.path.join(path_conceptual_v1_colab, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)
    if environment == "local":
        path = os.path.join(path_conceptual_v1_local, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)

    # instantiate the model
    hybridmodel_v1 = Hybrid_V1(prcp_hybrid, pet_hybrid, tmean_hybrid, calibrated_parameters) ### note vor V2 we need et here instead!


    # load the model with the trained weights and biases ### add local if needed
    if environment == "colab":
        path_nn = os.path.join(path_hybrid_v1_colab, "trained_parameters", f"hybrid_v1_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{start_lr}_epoch_{num_epochs}.pth")
        hybridmodel_v1.load_state_dict(torch.load(path_nn))

    # training
    with torch.no_grad():
        qsim_train_hybrid_v1  = hybridmodel_v1.forward(train_period, S0_hybrid_v1)

    # validation
    with torch.no_grad():
        qsim_val_hybrid_v1 = hybridmodel_v1.forward(val_period, S0_hybrid_v1)

    # testing
    with torch.no_grad():
        qsim_test_hybrid_v1 = hybridmodel_v1.forward(test_period, S0_hybrid_v1)

    # save training, validation and testing
    if environment == "colab":
        path_euler_training = os.path.join(path_hybrid_v1_colab, "discharge/training/euler/", f"qsim_hybrid_training_euler_ID_{id_num}_loss_func_{loss_func}_epoch_{num_epochs}_solver_{solver}_lr_{start_lr}.csv")
        path_euler_validation = os.path.join(path_hybrid_v1_colab, "discharge/validation/euler/", f"qsim_hybrid_validation_euler_ID_{id_num}_loss_func_{loss_func}_epoch_{num_epochs}_solver_{solver}_lr_{start_lr}.csv")
        path_euler_testing = os.path.join(path_hybrid_v1_colab, "discharge/testing/euler/", f"qsim_hybrid_testing_euler_ID_{id_num}_loss_func_{loss_func}_epoch_{num_epochs}_solver_{solver}_lr_{start_lr}.csv")

    if environment == "local":
        ### add if needed
        """
        path_euler_cal = os.path.join(path_hybrid_local, "discharge/cal/euler/", f"qsim_cal_euler_ID_{id_num}.csv")
        path_euler_test = os.path.join(path_hybrid_local, "discharge/test/euler/", f"qsim_test_euler_ID_{id_num}.csv")
        """

    # training, validation, testing
    np.savetxt(path_euler_training, qsim_train_hybrid_v1, delimiter=",")
    np.savetxt(path_euler_validation, qsim_val_hybrid_v1, delimiter=",")
    np.savetxt(path_euler_testing, qsim_test_hybrid_v1, delimiter=",")

    # divide the observation timeseries into training, validation and testing
    qobs_mm_training = qobs_mm_hybrid[train_period_int]
    qobs_mm_validation = qobs_mm_hybrid[val_period_int]
    qobs_mm_testing = qobs_mm_hybrid[test_period_int]

    # save the observation timeseries
    if environment == "colab":
        path_qobs_mm_training = os.path.join(path_hybrid_v1_colab, "discharge/training/observations/", f"qobs_mm_train_ID_{id_num}.csv")
        path_qobs_mm_validation = os.path.join(path_hybrid_v1_colab, "discharge/validation/observations/", f"qobs_mm_train_ID_{id_num}.csv")
        path_qobs_mm_testing = os.path.join(path_hybrid_v1_colab, "discharge/testing/observations/", f"qobs_mm_train_ID_{id_num}.csv")

    if environment == "local":
        ### add if needed
        """
        path_qobs_mm_cal = os.path.join(path_conceptual_v2_local, "discharge/cal/observations/", f"qobs_mm_cal_ID_{id_num}.csv")
        path_qobs_mm_test = os.path.join(path_conceptual_v2_local, "discharge/test/observations/", f"qobs_mm_test_ID_{id_num}.csv")
        """

    # training, validation, testing
    np.savetxt(path_qobs_mm_training, qobs_mm_training, delimiter=",") # [:-1]
    np.savetxt(path_qobs_mm_validation, qobs_mm_validation, delimiter=",")
    np.savetxt(path_qobs_mm_testing, qobs_mm_testing, delimiter=",")


    # calculate evaluation metrics BUT leave 365 days for warm-up
    nse_loss = NSELoss()
    mse_loss = MSELoss()
    kge_loss = KGELoss()

    # loss training
    nse_loss_euler_training = nse_loss(qsim_train_hybrid_v1[365:], qobs_mm_training[365:-1].cpu())
    mse_loss_euler_training = mse_loss(qsim_train_hybrid_v1[365:], qobs_mm_training[365:-1].cpu())
    kge_loss_euler_training = kge_loss(qsim_train_hybrid_v1[365:], qobs_mm_training[365:-1].cpu())

    # loss validation
    nse_loss_euler_validation = nse_loss(qsim_val_hybrid_v1[365:], qobs_mm_validation[365:-1].cpu())
    mse_loss_euler_validation = mse_loss(qsim_val_hybrid_v1[365:], qobs_mm_validation[365:-1].cpu())
    kge_loss_euler_validation = kge_loss(qsim_val_hybrid_v1[365:], qobs_mm_validation[365:-1].cpu())

    # loss test
    nse_loss_euler_testing = nse_loss(qsim_test_hybrid_v1[365:], qobs_mm_testing[365:-1].cpu())
    mse_loss_euler_testing = mse_loss(qsim_test_hybrid_v1[365:], qobs_mm_testing[365:-1].cpu())
    kge_loss_euler_testing = kge_loss(qsim_test_hybrid_v1[365:], qobs_mm_testing[365:-1].cpu())

    # save evaluation metrics (ideally to one large file )
    row = {
        "id_num": id_num,
        "start calibration": start_cal,
        "stop calibration": stop_cal,
        "start test": start_test,
        "stop test": stop_test,
        "nse_loss_euler_train": nse_loss_euler_training,
        "mse_loss_euler_train": mse_loss_euler_training,
        "kge_loss_euler_train": kge_loss_euler_training,
        "nse_loss_euler_val": nse_loss_euler_validation,
        "mse_loss_euler_val": mse_loss_euler_validation,
        "kge_loss_euler_val": kge_loss_euler_validation,
        "nse_loss_euler_test": nse_loss_euler_testing,
        "mse_loss_euler_test": mse_loss_euler_testing,
        "kge_loss_euler_test": kge_loss_euler_testing
    }

    # Create a DataFrame for the current iteration
    iteration_df = pd.DataFrame([row])

    # Append the iteration DataFrame to the list
    results_hybrid_dfs.append(iteration_df)

# Concatenate all DataFrames into one
results_hybrid_df = pd.concat(results_hybrid_dfs, ignore_index=True)


# Save the results DataFrame to a CSV file
if environment == "colab":
    path_results = os.path.join(path_hybrid_v1_colab, "evaluation_metrics/", f"hybrid_v1_results_solver_{solver}_loss_{loss_func}_epoch_{num_epochs}_lr_{start_lr}.csv")

if environment == "local":
    path_results = os.path.join(path_hybrid_v1_local, "evaluation_metrics/", f"hybrid_v1_results_cluster.csv")

results_hybrid_df.to_csv(path_results, index=False)


# In[ ]:


# RUN AND SAVE THE MODELS WITH THE TRAINED PARAMETERS
results_hybrid_dfs = []

# loop that rules it all
for id_num in IDs_red:
    print("ID:", id_num)

    # define initial storages
    S0_hybrid_v1 = torch.tensor([0.0, 0.0])
    S0_hybrid_v1 = torch.unsqueeze(S0_hybrid_v1, 0) # reshape

    # load the data
    # load df hybrid # IMPORTANT: full period from start train to stop test!
    if environment == "colab":
        df_hybrid = load_data_new(path_vars_eobs_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_eobs, start_date=start_train, stop_date=stop_test, dataset="eobs", qobs_mm=True)
    if environment == "local":
        df_hybrid = load_data_new(path_vars_eobs_local, path_qobs_lamah_local, path_catchment_attrs_colab, ID=id_num, variables=variables_eobs, start_date=start_train, stop_date=stop_test,  dataset="eobs", qobs_mm=True)


    # create tensors from the dataframe
    tensors_hybrid = df_to_tensors(df_hybrid, name_suffix="hybrid")
    # Processing tensors
    for name, tensor in tensors_hybrid.items():
        # Moving to GPU
        tensor = tensor.to(device)
        tensor = tensor.to(dtype=torch.float32)
        tensor.requires_grad = False  # inplace operation
        locals()[name] = tensor


    # load the parameters for the conceptual model part
    if environment == "colab":
        path = os.path.join(path_conceptual_v1_colab, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)
    if environment == "local":
        path = os.path.join(path_conceptual_v1_local, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)

    # instantiate the model
    hybridmodel_v1 = Hybrid_V1(prcp_hybrid, pet_hybrid, tmean_hybrid, calibrated_parameters) ### note vor V2 we need et here instead!


    # load the model with the trained weights and biases ### add local if needed
    if environment == "colab":
        path_nn = os.path.join(path_hybrid_v1_colab, "trained_parameters", f"hybrid_v1_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{start_lr}_epoch_{num_epochs}.pth")
        hybridmodel_v1.load_state_dict(torch.load(path_nn))

    # training
    with torch.no_grad():
        qsim_train_hybrid_v1  = hybridmodel_v1.forward(train_period, S0_hybrid_v1)

    # validation
    with torch.no_grad():
        qsim_val_hybrid_v1 = hybridmodel_v1.forward(val_period, S0_hybrid_v1)

    # testing
    with torch.no_grad():
        qsim_test_hybrid_v1 = hybridmodel_v1.forward(test_period, S0_hybrid_v1)


# #### Hybrid V1 RK4 Solver

# In[ ]:


# user input
num_epochs = 25 ### ?
solver = "rk4" # "euler" or "rk4"
loss_func = "MSE" # or "RMSE" or if used "NSE", "KGE"
start_lr = 0.01 # 0.01


# In[ ]:


# specify id
IDs_red = [241, 215, 581, 21, 277, 797, 432, 572]


# In[ ]:


#### delete again

# user input
num_epochs = 2 ### ?
solver = "rk4" # "euler" or "rk4"
loss_func = "MSE" # or "RMSE" or if used "NSE", "KGE"
start_lr = 0.01 # 0.01
IDs_red = [2]


# In[ ]:


# TRAINING THE 6 REPRESENTATIVE CATCHMNETS
# loop for everything
results_dfs = []

# loop that rules it all
#for id_num in IDs_red:
for id_num in IDs_red:
    print("ID:", id_num)

    loss_func = loss_func

    # define initial storages
    S0_hybrid_v1 = torch.tensor([0.0, 0.0])
    S0_hybrid_v1 = torch.unsqueeze(S0_hybrid_v1, 0) # reshape
    # push to GPU
    S0_hybrid_v1 = S0_hybrid_v1.to(device)

    # load the parameters for the model
    if environment == "colab":
        path = os.path.join(path_conceptual_v1_colab, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)
    if environment == "local":
        path = os.path.join(path_conceptual_v1_local, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)

    # load df hybrid # IMPORTANT: full period from start train to stop test!
    if environment == "colab":
        df_hybrid = load_data_new(path_vars_eobs_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_eobs, start_date=start_train, stop_date=stop_test, dataset="eobs", qobs_mm=True)
    if environment == "local":
        df_hybrid = load_data_new(path_vars_eobs_local, path_qobs_lamah_local, path_catchment_attrs_colab, ID=id_num, variables=variables_eobs, start_date=start_train, stop_date=stop_test,  dataset="eobs", qobs_mm=True)

    # create tensors from the dataframe
    tensors_hybrid = df_to_tensors(df_hybrid, name_suffix="hybrid")

    # Processing tensors
    for name, tensor in tensors_hybrid.items():
        # Moving to GPU
        tensor = tensor.to(device)
        tensor = tensor.to(dtype=torch.float32)
        tensor.requires_grad = False  # inplace operation
        locals()[name] = tensor

    # instantiate model
    ###
    hybridmodel_v1 = Hybrid_V1RK4(prcp_hybrid, pet_hybrid, tmean_hybrid, calibrated_parameters) ### note vor V2 we need et here instead!
    # hybridmodel_v1.to(device) # put back if using GPU

    # loss function
    if loss_func == "MSE":
        criterion = nn.MSELoss()
        print("Using MSE loss func")
    elif loss_func == "RMSE":
        criterion = RMSLELoss()
        print("Using RMSE loss func")
    elif loss_func == "KGE":
        criterion = KGELossMOD()
        print("Using KGE loss func")
    else:
        print("no valid loss func defined")

    # Define the optimizer, which optimzer should be used? Adam? SGD?
    optimizer = optim.Adam(hybridmodel_v1.parameters(), lr=start_lr) # what is a good lr to start with?


    # learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    torch.autograd.set_detect_anomaly(True)

    train_losses_v1 = []
    val_losses_v1 = []

    best_loss = 50000000
    best_model_state_hybrid_v1 = None ### what was this for again?

    # Train the model
    for epoch in range(num_epochs):

        # params
        params_hybrid_v1 = hybridmodel_v1.state_dict()

        # forward pass training data
        qsim_train_hybrid_v1 = hybridmodel_v1.forward(train_period, S0_hybrid_v1)

        # forward pass on validation data
        with torch.no_grad():
            qsim_val_hybrid_v1 = hybridmodel_v1.forward(val_period, S0_hybrid_v1)


        #### remove the last time setp of qobs
        qobs_train = qobs_mm_hybrid[train_period_int[:-1]]
        qobs_val = qobs_mm_hybrid[val_period_int[:-1]]


        ############################### downsampling here
        qsim_train_hybrid_v1_itp = interpolate_downsample(qobs_train, qsim_train_hybrid_v1)
        qsim_val_hybrid_v1_itp = interpolate_downsample(qobs_val, qsim_val_hybrid_v1)



        # Added warmup period here
        # [-1] for qobs_mm is a fix here as qsim is one element too short because of the way it is calculated and appended
        ##### train_loss = criterion(qsim_train_hybrid_v1[365:], qobs_mm_hybrid[train_period_int[365:-1]]) # make sure to NOT train on normalized observations!
        ##### val_loss = criterion(qsim_val_hybrid_v1[365:], qobs_mm_hybrid[val_period_int[365:-1]]) # make sure to NOT train on normalized observations!

        train_loss = criterion(qsim_train_hybrid_v1_itp[365:], qobs_train[365:]) # make sure to NOT train on normalized observations!
        val_loss = criterion(qsim_val_hybrid_v1_itp[365:], qobs_val[365:]) # make sure to NOT train on normalized observations!

        # append train and validation loss
        train_losses_v1.append(train_loss.item())
        val_losses_v1.append(val_loss.item())


        # backward pass
        optimizer.zero_grad()
        train_loss.backward() #retain_graph=True

        print(hybridmodel_v1.layer1.weight.grad)

        optimizer.step()

        # update lr schedule
        scheduler.step(train_loss) # train loss

        # update and save best model (based on validation loss)
        if val_loss.item() < best_loss:
            print("Accessed loop")
            best_loss = val_loss.item()
            best_model_state_hybrid_v1 = copy.deepcopy(hybridmodel_v1.state_dict())


        print(f"Epoch {epoch}: Train loss = {train_loss.item()} Validation loss = {val_loss.item()}")


    # save trained model parameters
    if environment=='local':
        torch.save(best_model_state_hybrid_v1, f'models/hybrid_v1_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{start_lr}.pth')
    if environment=='colab':
        torch.save(best_model_state_hybrid_v1, f'/content/drive/MyDrive/msc_thesis/python/notebooks/colab/HybridV1/trained_parameters/hybrid_v1_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{start_lr}_epoch_{num_epochs}.pth')


# In[ ]:


# user input to run the results
num_epochs = 25 ### ?
solver = "rk4" # "euler" or "rk4"
loss_func = "MSE" # or "RMSE" or if used "NSE", "KGE"
start_lr = 0.01 # 0.01


# In[ ]:


# check id_num
IDs_red


# In[ ]:


# RUN AND SAVE THE MODELS WITH THE TRAINED PARAMETERS
results_hybrid_dfs = []

# loop that rules it all
#for id_num in IDs_red:
for id_num in IDs_red:
    print("ID:", id_num)

    # define initial storages
    S0_hybrid_v1 = torch.tensor([0.0, 0.0])
    S0_hybrid_v1 = torch.unsqueeze(S0_hybrid_v1, 0) # reshape

    # load the data
    # load df hybrid # IMPORTANT: full period from start train to stop test!
    if environment == "colab":
        df_hybrid = load_data_new(path_vars_eobs_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_eobs, start_date=start_train, stop_date=stop_test, dataset="eobs", qobs_mm=True)
    if environment == "local":
        df_hybrid = load_data_new(path_vars_eobs_local, path_qobs_lamah_local, path_catchment_attrs_colab, ID=id_num, variables=variables_eobs, start_date=start_train, stop_date=stop_test,  dataset="eobs", qobs_mm=True)


    # create tensors from the dataframe
    tensors_hybrid = df_to_tensors(df_hybrid, name_suffix="hybrid")
    # Processing tensors
    for name, tensor in tensors_hybrid.items():
        # Moving to GPU
        tensor = tensor.to(device)
        tensor = tensor.to(dtype=torch.float32)
        tensor.requires_grad = False  # inplace operation
        locals()[name] = tensor


    # load the parameters for the conceptual model part
    if environment == "colab":
        path = os.path.join(path_conceptual_v1_colab, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)
    if environment == "local":
        path = os.path.join(path_conceptual_v1_local, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)

    # instantiate the model
    ###
    hybridmodel_v1 = Hybrid_V1RK4(prcp_hybrid, pet_hybrid, tmean_hybrid, calibrated_parameters) ### note vor V2 we need et here instead!


    # load the model with the trained weights and biases ### add local if needed
    if environment == "colab":
        path_nn = os.path.join(path_hybrid_v1_colab, "trained_parameters", f"hybrid_v1_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{start_lr}_epoch_{num_epochs}.pth")
        hybridmodel_v1.load_state_dict(torch.load(path_nn))

    # training
    with torch.no_grad():
        qsim_train_hybrid_v1 = hybridmodel_v1.forward(train_period, S0_hybrid_v1)

    # validation
    with torch.no_grad():
        qsim_val_hybrid_v1 = hybridmodel_v1.forward(val_period, S0_hybrid_v1)

    # testing
    with torch.no_grad():
        qsim_test_hybrid_v1 = hybridmodel_v1.forward(test_period, S0_hybrid_v1)


    ### save the observations
    # divide the observation timeseries into training, validation and testing this is already done in the code block above for the training so not needed
    qobs_mm_training = qobs_mm_hybrid[train_period_int[:-1]]
    qobs_mm_validation = qobs_mm_hybrid[val_period_int[:-1]]
    qobs_mm_testing = qobs_mm_hybrid[test_period_int[:-1]]

    # save the observation timeseries
    if environment == "colab":
        path_qobs_mm_training = os.path.join(path_hybrid_v1_colab, "discharge/training/observations/", f"qobs_mm_train_ID_{id_num}.csv")
        path_qobs_mm_validation = os.path.join(path_hybrid_v1_colab, "discharge/validation/observations/", f"qobs_mm_train_ID_{id_num}.csv")
        path_qobs_mm_testing = os.path.join(path_hybrid_v1_colab, "discharge/testing/observations/", f"qobs_mm_train_ID_{id_num}.csv")

    if environment == "local":
        ### add if needed
        """
        path_qobs_mm_cal = os.path.join(path_conceptual_v2_local, "discharge/cal/observations/", f"qobs_mm_cal_ID_{id_num}.csv")
        path_qobs_mm_test = os.path.join(path_conceptual_v2_local, "discharge/test/observations/", f"qobs_mm_test_ID_{id_num}.csv")
        """

    # training, validation, testing
    np.savetxt(path_qobs_mm_training, qobs_mm_training, delimiter=",") # [:-1]
    np.savetxt(path_qobs_mm_validation, qobs_mm_validation, delimiter=",")
    np.savetxt(path_qobs_mm_testing, qobs_mm_testing, delimiter=",")


    ### now interpolate the train, val and test predictions to make them the same shape as observations and predictions using the euler solver
    qsim_train_hybrid_v1_itp = interpolate_downsample(qobs_mm_training, qsim_train_hybrid_v1)
    qsim_val_hybrid_v1_itp = interpolate_downsample(qobs_mm_validation, qsim_val_hybrid_v1)
    qsim_test_hybrid_v1_itp = interpolate_downsample(qobs_mm_testing, qsim_test_hybrid_v1)


    ### save the predictions
    # save training, validation and testing
    if environment == "colab":
        path_euler_training = os.path.join(path_hybrid_v1_colab, "discharge/training/rk4/", f"qsim_hybrid_training_euler_ID_{id_num}_loss_func_{loss_func}_epoch_{num_epochs}_solver_{solver}_lr_{start_lr}.csv")
        path_euler_validation = os.path.join(path_hybrid_v1_colab, "discharge/validation/rk4/", f"qsim_hybrid_validation_euler_ID_{id_num}_loss_func_{loss_func}_epoch_{num_epochs}_solver_{solver}_lr_{start_lr}.csv")
        path_euler_testing = os.path.join(path_hybrid_v1_colab, "discharge/testing/rk4/", f"qsim_hybrid_testing_euler_ID_{id_num}_loss_func_{loss_func}_epoch_{num_epochs}_solver_{solver}_lr_{start_lr}.csv")

    if environment == "local":
        ### add if needed
        """
        path_euler_cal = os.path.join(path_hybrid_local, "discharge/cal/euler/", f"qsim_cal_euler_ID_{id_num}.csv")
        path_euler_test = os.path.join(path_hybrid_local, "discharge/test/euler/", f"qsim_test_euler_ID_{id_num}.csv")
        """

    # training, validation, testing
    np.savetxt(path_euler_training, qsim_train_hybrid_v1_itp, delimiter=",")
    np.savetxt(path_euler_validation, qsim_val_hybrid_v1_itp, delimiter=",")
    np.savetxt(path_euler_testing, qsim_test_hybrid_v1_itp, delimiter=",")




    # calculate evaluation metrics BUT leave 365 days for warm-up
    nse_loss = NSELoss()
    mse_loss = MSELoss()
    kge_loss = KGELoss()

    # loss training
    nse_loss_rk4_training = nse_loss(qsim_train_hybrid_v1_itp[365:], qobs_mm_training[365:].cpu())
    mse_loss_rk4_training = mse_loss(qsim_train_hybrid_v1_itp[365:], qobs_mm_training[365:].cpu())
    kge_loss_rk4_training = kge_loss(qsim_train_hybrid_v1_itp[365:], qobs_mm_training[365:].cpu())

    # loss validation
    nse_loss_rk4_validation = nse_loss(qsim_val_hybrid_v1_itp[365:], qobs_mm_validation[365:].cpu())
    mse_loss_rk4_validation = mse_loss(qsim_val_hybrid_v1_itp[365:], qobs_mm_validation[365:].cpu())
    kge_loss_rk4_validation = kge_loss(qsim_val_hybrid_v1_itp[365:], qobs_mm_validation[365:].cpu())

    # loss test
    nse_loss_rk4_testing = nse_loss(qsim_test_hybrid_v1_itp[365:], qobs_mm_testing[365:].cpu())
    mse_loss_rk4_testing = mse_loss(qsim_test_hybrid_v1_itp[365:], qobs_mm_testing[365:].cpu())
    kge_loss_rk4_testing = kge_loss(qsim_test_hybrid_v1_itp[365:], qobs_mm_testing[365:].cpu())

    # save evaluation metrics (ideally to one large file )
    row = {
        "id_num": id_num,
        "start calibration": start_cal,
        "stop calibration": stop_cal,
        "start test": start_test,
        "stop test": stop_test,
        "nse_loss_rk4_train": nse_loss_rk4_training,
        "mse_loss_rk4_train": mse_loss_rk4_training,
        "kge_loss_rk4_train": kge_loss_rk4_training,
        "nse_loss_rk4_val": nse_loss_rk4_validation,
        "mse_loss_rk4_val": mse_loss_rk4_validation,
        "kge_loss_rk4_val": kge_loss_rk4_validation,
        "nse_loss_rk4_test": nse_loss_rk4_testing,
        "mse_loss_rk4_test": mse_loss_rk4_testing,
        "kge_loss_rk4_test": kge_loss_rk4_testing}

    # Create a DataFrame for the current iteration
    iteration_df = pd.DataFrame([row])

    # Append the iteration DataFrame to the list
    results_hybrid_dfs.append(iteration_df)

# Concatenate all DataFrames into one
results_hybrid_df = pd.concat(results_hybrid_dfs, ignore_index=True)


# Save the results DataFrame to a CSV file
if environment == "colab":
    path_results = os.path.join(path_hybrid_v1_colab, "evaluation_metrics/", f"hybrid_v1_results_{solver}_loss_{loss_func}_epoch_{num_epochs}_lr_{start_lr}.csv")

if environment == "local":
    path_results = os.path.join(path_hybrid_v1_local, "evaluation_metrics/", f"hybrid_v1_results_ID_{id_num}_solver_{solver}.csv")

results_hybrid_df.to_csv(path_results, index=False)


# #### Hybrid V1 RK4 with additional input forcings

# In[ ]:


# user input
num_epochs = 25
solver = "rk4" # "euler" or "rk4"
loss_func = "MSE" # or "RMSE" or if used "NSE", "KGE"
start_lr = 0.01 # 0.01


# In[ ]:


# specify id
id_num = 215


# In[ ]:


#variables_hybrid_v1_large = ['tmax', 'tmean', 'tmin', 'seapress', 'humidity', 'windspeed', 'srad', 'albedo', 'pet', 'daylength', 'pev', 'prcp']
#variables_eobs = ['prcp', 'tmean', 'tmin', 'tmax', 'seapress', 'humidity', 'windspeed', 'srad', 'albedo', 'pet', 'daylength', 'pev'] # also has "DOY"
variables_hybrid_v1_large = ['prcp', 'tmean', 'pet', 'srad', 'tmin', 'tmax']


# In[ ]:


# load data
# load df hybrid # IMPORTANT: full period from start train to stop test!
if environment == "colab":
    df_hybrid = load_data_new(path_vars_eobs_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_hybrid_v1_large, start_date=start_train, stop_date=stop_test, dataset="eobs", qobs_mm=True)
if environment == "local":
    df_hybrid = load_data_new(path_vars_eobs_local, path_qobs_lamah_local, path_catchment_attrs_colab, ID=id_num, variables=variables_hybrid_v1_large, start_date=start_train, stop_date=stop_test,  dataset="eobs", qobs_mm=True)



# In[ ]:


##################### this is just to test Hybrid V3, keep only if it works
results_dfs = []

# loop that rules it all
# for id_num in IDs_red:
if id_num == 215:
    print("ID:", id_num)

    # define initial storages
    S0_hybrid_v1rk4_large = torch.tensor([0.0, 0.0])
    S0_hybrid_v1rk4_large = torch.unsqueeze(S0_hybrid_v1rk4_large, 0) # reshape
    # push to GPU
    S0_hybrid_v1rk4_large = S0_hybrid_v1rk4_large.to(device)

    # load the parameters for the model
    if environment == "colab":
        path = os.path.join(path_conceptual_v1_colab, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)
    if environment == "local":
        path = os.path.join(path_conceptual_v1_local, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)

    # load df hybrid # IMPORTANT: full period from start train to stop test!
    if environment == "colab":
        df_hybrid = load_data_new(path_vars_eobs_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_hybrid_v1_large, start_date=start_train, stop_date=stop_test, dataset="eobs", qobs_mm=True)
    if environment == "local":
        df_hybrid = load_data_new(path_vars_eobs_local, path_qobs_lamah_local, path_catchment_attrs_colab, ID=id_num, variables=variables_hybrid_v1_large, start_date=start_train, stop_date=stop_test,  dataset="eobs", qobs_mm=True)

    # create tensors from the dataframe
    tensors_hybrid = df_to_tensors(df_hybrid, name_suffix="hybrid")

    # Processing tensors
    for name, tensor in tensors_hybrid.items():
        # Moving to GPU
        tensor = tensor.to(device)
        tensor = tensor.to(dtype=torch.float32)
        tensor.requires_grad = False  # inplace operation
        locals()[name] = tensor

    # instantiate model
    # variables 'tmax', 'tmean', 'tmin', 'seapress', 'humidity', 'windspeed', 'srad', 'albedo', 'pet', 'daylength', 'pev', 'prcp']
    hybridmodel_v1rk4_large = Hybrid_V1RK4_large(tmax_hybrid, tmean_hybrid, tmin_hybrid, srad_hybrid, pet_hybrid, prcp_hybrid, calibrated_parameters)
    ### hybridmodel_v2 = Hybrid_V4(prcp_hybrid, et_hybrid, tmean_hybrid, calibrated_parameters) ### note vor V2 we need et here instead!
    # hybridmodel_v1.to(device) # put back if using GPU

    # loss function
    if loss_func == "MSE":
        criterion = nn.MSELoss()
        print("Using MSE loss func")
    elif loss_func == "RMSE":
        criterion = RMSLELoss()
        print("Using RMSE loss func")
    elif loss_func == "KGE":
        criterion = KGELossMOD()
        print("Using KGE loss func")
    else:
        print("no valid loss func defined")

    # Define the optimizer, which optimzer should be used? Adam? SGD?
    optimizer = optim.Adam(hybridmodel_v1rk4_large.parameters(), lr=start_lr) # what is a good lr to start with?


    # learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=7, verbose=True)

    torch.autograd.set_detect_anomaly(True)

    train_losses_v1rk4_large = []
    val_losses_v1rk4_large = []

    best_loss = 50000000
    best_model_state_hybrid_v1rk4_large = None ### what was this for again?

    # Train the model
    for epoch in range(num_epochs):

        # params
        params_hybrid_v1rk4_large = hybridmodel_v1rk4_large.state_dict()

        # forward pass training data
        qsim_train_hybrid_v1rk4_large = hybridmodel_v1rk4_large.forward(train_period, S0_hybrid_v1rk4_large)

        # forward pass on validation data
        with torch.no_grad():
            qsim_val_hybrid_v1rk4_large = hybridmodel_v1rk4_large.forward(val_period, S0_hybrid_v1rk4_large)

        #### remove the last time setp of qobs
        qobs_train = qobs_mm_hybrid[train_period_int[:]]
        qobs_val = qobs_mm_hybrid[val_period_int[:]]


        ############################### downsampling here
        qsim_train_hybrid_v1rk4_large_itp = interpolate_downsample(qobs_train, qsim_train_hybrid_v1rk4_large)
        qsim_val_hybrid_v1rk4_large_itp = interpolate_downsample(qobs_val, qsim_val_hybrid_v1rk4_large)



        # Added warmup period here
        # [-1] for qobs_mm is a fix here as qsim is one element too short because of the way it is calculated and appended
        ##### train_loss = criterion(qsim_train_hybrid_v4[365:], qobs_mm_hybrid[train_period_int[365:-1]]) # make sure to NOT train on normalized observations!
        ##### val_loss = criterion(qsim_val_hybrid_v4[365:], qobs_mm_hybrid[val_period_int[365:-1]]) # make sure to NOT train on normalized observations!

        train_loss = criterion(qsim_train_hybrid_v1rk4_large_itp[365:], qobs_mm_hybrid[train_period_int[365:]]) # make sure to NOT train on normalized observations!
        val_loss = criterion(qsim_val_hybrid_v1rk4_large_itp[365:], qobs_mm_hybrid[val_period_int[365:]]) # make sure to NOT train on normalized observations!

        # append train and validation loss
        train_losses_v1rk4_large.append(train_loss.item())
        val_losses_v1rk4_large.append(val_loss.item())


        # backward pass
        optimizer.zero_grad()
        train_loss.backward() #retain_graph=True

        print(hybridmodel_v1rk4_large.layer1.weight.grad)

        optimizer.step()

        # update lr schedule
        scheduler.step(train_loss) # train loss

        # update and save best model (based on validation loss)
        if val_loss.item() < best_loss:
            print("Accessed loop")
            best_loss = val_loss.item()
            best_model_state_hybrid_v1rk4_large = copy.deepcopy(hybridmodel_v1rk4_large.state_dict())


        print(f"Epoch {epoch}: Train loss = {train_loss.item()} Validation loss = {val_loss.item()}")


    # save trained model parameters
    if environment=='local':
        torch.save(best_model_state_hybrid_v1rk4_large, f'models/hybrid_v1_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{lr_start}.pth')
    if environment=='colab':
        torch.save(best_model_state_hybrid_v1rk4_large, f'/content/drive/MyDrive/msc_thesis/python/notebooks/colab/HybridV1Large/trained_parameters/hybrid_v1rk4_large_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{start_lr}.pth')


# ### Hybrid V2 - LamaH-CE

# Representative catchment IDs for LamaH-CE
# IDs = 334, 743, 439, 24, 330, 75, 383
# 
# **Catchment that is closest to the overall median KGE (Test) of the conceptual V2 model: 79**

# #### Run on the 6 representative catchments

# In[ ]:


# user input
num_epochs = 50
solver = "euler" # or "rk4"
loss_func = "MSE" # or "RMSE" or if used "NSE", "KGE"
start_lr = 0.01 # 0.01, 0.005


# In[ ]:


# representative catchment IDs for LamaH-CE
IDs_red = [334, 743, 439, 24, 330, 75, 383] # 79 closest to overall median 


# In[ ]:


# TRAINING THE 6 REPRESENTATIVE CATCHMENTs
# loop for everything
results_dfs = []

# loop that rules it all
for id_num in IDs_red:
    print("ID:", id_num)

    # define initial storages
    S0_hybrid_v2 = torch.tensor([0.0, 0.0])
    S0_hybrid_v2 = torch.unsqueeze(S0_hybrid_v2, 0) # reshape
    # push to GPU
    S0_hybrid_v2 = S0_hybrid_v2.to(device)

    # load the parameters for the model
    if environment == "colab":
        path = os.path.join(path_conceptual_v2_colab, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)
    if environment == "local":
        path = os.path.join(path_conceptual_v2_local, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)

    # load df hybrid # IMPORTANT: full period from start train to stop test!
    if environment == "colab":
        df_hybrid = load_data_new(path_vars_lamah_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_lamah, start_date=start_train, stop_date=stop_test, dataset="lamah", qobs_mm=True)
    if environment == "local":
        df_hybrid = load_data_new(path_vars_lamah_local, path_qobs_lamah_local, path_catchment_attrs_colab, ID=id_num, variables=variables_lamah, start_date=start_train, stop_date=stop_test,  dataset="lamah", qobs_mm=True)

    # create tensors from the dataframe
    tensors_hybrid = df_to_tensors(df_hybrid, name_suffix="hybrid")

    # Processing tensors
    for name, tensor in tensors_hybrid.items():
        # Moving to GPU
        tensor = tensor.to(device)
        tensor = tensor.to(dtype=torch.float32)
        tensor.requires_grad = False  # inplace operation
        locals()[name] = tensor

    # instantiate model
    hybridmodel_v2 = Hybrid_V2(prcp_hybrid, et_hybrid, tmean_hybrid, calibrated_parameters) ### note vor V2 we need et here instead!
    # hybridmodel_v1.to(device) # put back if using GPU

    # loss function
    if loss_func == "MSE":
        criterion = nn.MSELoss()
        print("Using MSE loss func")
    elif loss_func == "RMSE":
        criterion = RMSLELoss()
        print("Using RMSE loss func")
    elif loss_func == "KGE":
        criterion = KGELossMOD()
        print("Using KGE loss func")
    else:
        print("no valid loss func defined")

    # Define the optimizer, which optimzer should be used? Adam? SGD?
    optimizer = optim.Adam(hybridmodel_v2.parameters(), lr=start_lr) # what is a good lr to start with?


    # learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    torch.autograd.set_detect_anomaly(True)

    train_losses_v2 = []
    val_losses_v2 = []

    best_loss = 50000000
    best_model_state_hybrid_v2 = None ### what was this for again?

    # Train the model
    for epoch in range(num_epochs):

        # params
        params_hybrid_v2 = hybridmodel_v2.state_dict()

        # forward pass training data
        qsim_train_hybrid_v2 = hybridmodel_v2.forward(train_period, S0_hybrid_v2)

        # forward pass on validation data
        with torch.no_grad():
            qsim_val_hybrid_v2 = hybridmodel_v2.forward(val_period, S0_hybrid_v2)


        # Added warmup period here
        # [-1] for qobs_mm is a fix here as qsim is one element too short because of the way it is calculated and appended
        train_loss = criterion(qsim_train_hybrid_v2[365:], qobs_mm_hybrid[train_period_int[365:-1]]) # make sure to NOT train on normalized observations!
        val_loss = criterion(qsim_val_hybrid_v2[365:], qobs_mm_hybrid[val_period_int[365:-1]]) # make sure to NOT train on normalized observations!

        # append train and validation loss
        train_losses_v2.append(train_loss.item())
        val_losses_v2.append(val_loss.item())


        # backward pass
        optimizer.zero_grad()
        train_loss.backward() #retain_graph=True

        print(hybridmodel_v2.layer1.weight.grad)

        optimizer.step()

        # update lr schedule
        scheduler.step(train_loss) # train loss

        # update and save best model (based on validation loss)
        if val_loss.item() < best_loss:
            print("Accessed loop")
            best_loss = val_loss.item()
            best_model_state_hybrid_v2 = copy.deepcopy(hybridmodel_v2.state_dict())


        print(f"Epoch {epoch}: Train loss = {train_loss.item()} Validation loss = {val_loss.item()}")


    # save trained model parameters
    if environment=='local':
        torch.save(best_model_state_hybrid_v2, f'models/hybrid_v1_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{lr_start}.pth')
    if environment=='colab':
        torch.save(best_model_state_hybrid_v2, f'/content/drive/MyDrive/msc_thesis/python/notebooks/colab/HybridV2/trained_parameters/hybrid_v2_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{start_lr}_epoch_{num_epochs}.pth')


# In[ ]:


# user input to run the model outputs
num_epochs = 50
solver = "euler" # or "rk4"
loss_func = "MSE" # or "RMSE" or if used "NSE", "KGE"
start_lr = 0.005 # 0.01, 0.005


# In[ ]:


# representative catchment IDs for LamaH-CE
IDs_red = [334, 743, 439, 24, 330, 75, 383, 79]


# In[ ]:


# now all in one
results_hybrid_dfs = []

# loop that rules it all
for id_num in IDs_red:
    print("ID:", id_num)

    # define initial storages
    S0_hybrid_v2 = torch.tensor([0.0, 0.0])
    S0_hybrid_v2 = torch.unsqueeze(S0_hybrid_v2, 0) # reshape

    # load the data
    # load df hybrid # IMPORTANT: full period from start train to stop test!
    if environment == "colab":
        df_hybrid = load_data_new(path_vars_lamah_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_lamah, start_date=start_train, stop_date=stop_test, dataset="lamah", qobs_mm=True)
    if environment == "local":
        df_hybrid = load_data_new(path_vars_lamah_local, path_qobs_lamah_local, path_catchment_attrs_colab, ID=id_num, variables=variables_lamah, start_date=start_train, stop_date=stop_test,  dataset="lamah", qobs_mm=True)


    # create tensors from the dataframe
    tensors_hybrid = df_to_tensors(df_hybrid, name_suffix="hybrid")
    # Processing tensors
    for name, tensor in tensors_hybrid.items():
        # Moving to GPU
        tensor = tensor.to(device)
        tensor = tensor.to(dtype=torch.float32)
        tensor.requires_grad = False  # inplace operation
        locals()[name] = tensor


    # load the parameters for the conceptual model part
    if environment == "colab":
        path = os.path.join(path_conceptual_v2_colab, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)
    if environment == "local":
        path = os.path.join(path_conceptual_v2_local, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)

    # instantiate the model
    hybridmodel_v2 = Hybrid_V2(prcp_hybrid, et_hybrid, tmean_hybrid, calibrated_parameters) ### note vor V2 we need et here instead!


    # load the model with the trained weights and biases ### add local if needed
    if environment == "colab":
        path_nn = os.path.join(path_hybrid_v2_colab, "trained_parameters", f"hybrid_v2_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{start_lr}_epoch_{num_epochs}.pth")
        hybridmodel_v2.load_state_dict(torch.load(path_nn))

    # training
    with torch.no_grad():
        qsim_train_hybrid_v2 = hybridmodel_v2.forward(train_period, S0_hybrid_v2)

    # validation
    with torch.no_grad():
        qsim_val_hybrid_v2 = hybridmodel_v2.forward(val_period, S0_hybrid_v2)

    # testing
    with torch.no_grad():
        qsim_test_hybrid_v2 = hybridmodel_v2.forward(test_period, S0_hybrid_v2)

    # save training, validation and testing
    if environment == "colab":
        path_euler_training = os.path.join(path_hybrid_v2_colab, "discharge/training/euler/", f"qsim_hybrid_training_euler_ID_{id_num}_loss_func_{loss_func}_epoch_{num_epochs}_solver_{solver}_lr_{start_lr}.csv")
        path_euler_validation = os.path.join(path_hybrid_v2_colab, "discharge/validation/euler/", f"qsim_hybrid_validation_euler_ID_{id_num}_loss_func_{loss_func}_epoch_{num_epochs}_solver_{solver}_lr_{start_lr}.csv")
        path_euler_testing = os.path.join(path_hybrid_v2_colab, "discharge/testing/euler/", f"qsim_hybrid_testing_euler_ID_{id_num}_loss_func_{loss_func}_epoch_{num_epochs}_solver_{solver}_lr_{start_lr}.csv")

    if environment == "local":
        ### add if needed
        """
        path_euler_cal = os.path.join(path_hybrid_local, "discharge/cal/euler/", f"qsim_cal_euler_ID_{id_num}.csv")
        path_euler_test = os.path.join(path_hybrid_local, "discharge/test/euler/", f"qsim_test_euler_ID_{id_num}.csv")
        """

    # training, validation, testing
    np.savetxt(path_euler_training, qsim_train_hybrid_v2, delimiter=",")
    np.savetxt(path_euler_validation, qsim_val_hybrid_v2, delimiter=",")
    np.savetxt(path_euler_testing, qsim_test_hybrid_v2, delimiter=",")

    # divide the observation timeseries into training, validation and testing
    qobs_mm_training = qobs_mm_hybrid[train_period_int]
    qobs_mm_validation = qobs_mm_hybrid[val_period_int]
    qobs_mm_testing = qobs_mm_hybrid[test_period_int]

    # save the observation timeseries
    if environment == "colab":
        path_qobs_mm_training = os.path.join(path_hybrid_v2_colab, "discharge/training/observations/", f"qobs_mm_train_ID_{id_num}.csv")
        path_qobs_mm_validation = os.path.join(path_hybrid_v2_colab, "discharge/validation/observations/", f"qobs_mm_train_ID_{id_num}.csv")
        path_qobs_mm_testing = os.path.join(path_hybrid_v2_colab, "discharge/testing/observations/", f"qobs_mm_train_ID_{id_num}.csv")

    if environment == "local":
        ### add if needed
        """
        path_qobs_mm_cal = os.path.join(path_conceptual_v2_local, "discharge/cal/observations/", f"qobs_mm_cal_ID_{id_num}.csv")
        path_qobs_mm_test = os.path.join(path_conceptual_v2_local, "discharge/test/observations/", f"qobs_mm_test_ID_{id_num}.csv")
        """

    # training, validation, testing
    np.savetxt(path_qobs_mm_training, qobs_mm_training, delimiter=",") # [:-1]
    np.savetxt(path_qobs_mm_validation, qobs_mm_validation, delimiter=",")
    np.savetxt(path_qobs_mm_testing, qobs_mm_testing, delimiter=",")


    # calculate evaluation metrics BUT leave 365 days for warm-up
    nse_loss = NSELoss()
    mse_loss = MSELoss()
    kge_loss = KGELoss()

    # loss training
    nse_loss_euler_training = nse_loss(qsim_train_hybrid_v2[365:], qobs_mm_training[365:-1].cpu())
    mse_loss_euler_training = mse_loss(qsim_train_hybrid_v2[365:], qobs_mm_training[365:-1].cpu())
    kge_loss_euler_training = kge_loss(qsim_train_hybrid_v2[365:], qobs_mm_training[365:-1].cpu())

    # loss validation
    nse_loss_euler_validation = nse_loss(qsim_val_hybrid_v2[365:], qobs_mm_validation[365:-1].cpu())
    mse_loss_euler_validation = mse_loss(qsim_val_hybrid_v2[365:], qobs_mm_validation[365:-1].cpu())
    kge_loss_euler_validation = kge_loss(qsim_val_hybrid_v2[365:], qobs_mm_validation[365:-1].cpu())

    # loss test
    nse_loss_euler_testing = nse_loss(qsim_test_hybrid_v2[365:], qobs_mm_testing[365:-1].cpu())
    mse_loss_euler_testing = mse_loss(qsim_test_hybrid_v2[365:], qobs_mm_testing[365:-1].cpu())
    kge_loss_euler_testing = kge_loss(qsim_test_hybrid_v2[365:], qobs_mm_testing[365:-1].cpu())

    # save evaluation metrics (ideally to one large file )
    row = {
        "id_num": id_num,
        "start calibration": start_cal,
        "stop calibration": stop_cal,
        "start test": start_test,
        "stop test": stop_test,
        "nse_loss_euler_train": nse_loss_euler_training,
        "mse_loss_euler_train": mse_loss_euler_training,
        "kge_loss_euler_train": kge_loss_euler_training,
        "nse_loss_euler_val": nse_loss_euler_validation,
        "mse_loss_euler_val": mse_loss_euler_validation,
        "kge_loss_euler_val": kge_loss_euler_validation,
        "nse_loss_euler_test": nse_loss_euler_testing,
        "mse_loss_euler_test": mse_loss_euler_testing,
        "kge_loss_euler_test": kge_loss_euler_testing
    }

    # Create a DataFrame for the current iteration
    iteration_df = pd.DataFrame([row])

    # Append the iteration DataFrame to the list
    results_hybrid_dfs.append(iteration_df)

# Concatenate all DataFrames into one
results_hybrid_df = pd.concat(results_hybrid_dfs, ignore_index=True)


# Save the results DataFrame to a CSV file
if environment == "colab":
    path_results = os.path.join(path_hybrid_v2_colab, "evaluation_metrics/", f"hybrid_v2_results_solver_{solver}_loss_{loss_func}_epoch_{num_epochs}_lr_{start_lr}.csv")

if environment == "local":
    path_results = os.path.join(path_hybrid_v2_local, "evaluation_metrics/", f"hybrid_v2_results_cluster.csv")

results_hybrid_df.to_csv(path_results, index=False)


# #### Hybrid V2 RK4 Solver

# In[ ]:


# user input
num_epochs = 25
solver = "rk4" # "euler" or "rk4"
loss_func = "MSE" # or "RMSE" or if used "NSE", "KGE"
start_lr = 0.01 # 0.01


# In[ ]:


# representative catchment IDs for LamaH-CE
IDs_red = [334, 743, 439, 24, 330, 75, 383, 79]


# In[ ]:


# TRAINING THE 1 REPRESENTATIVE CATCHMENT1
# loop for everything
results_dfs = []

# loop that rules it all
for id_num in IDs_red:
    print("ID:", id_num)

    # define initial storages
    S0_hybrid_v2 = torch.tensor([0.0, 0.0])
    S0_hybrid_v2 = torch.unsqueeze(S0_hybrid_v2, 0) # reshape
    # push to GPU
    S0_hybrid_v2 = S0_hybrid_v2.to(device)

    # load the parameters for the model
    if environment == "colab":
        path = os.path.join(path_conceptual_v2_colab, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)
    if environment == "local":
        path = os.path.join(path_conceptual_v2_local, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)

    # load df hybrid # IMPORTANT: full period from start train to stop test!
    if environment == "colab":
        df_hybrid = load_data_new(path_vars_lamah_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_lamah, start_date=start_train, stop_date=stop_test, dataset="lamah", qobs_mm=True)
    if environment == "local":
        df_hybrid = load_data_new(path_vars_lamah_local, path_qobs_lamah_local, path_catchment_attrs_colab, ID=id_num, variables=variables_lamah, start_date=start_train, stop_date=stop_test,  dataset="lamah", qobs_mm=True)

    # create tensors from the dataframe
    tensors_hybrid = df_to_tensors(df_hybrid, name_suffix="hybrid")

    # Processing tensors
    for name, tensor in tensors_hybrid.items():
        # Moving to GPU
        tensor = tensor.to(device)
        tensor = tensor.to(dtype=torch.float32)
        tensor.requires_grad = False  # inplace operation
        locals()[name] = tensor

    # instantiate model
    hybridmodel_v2 = Hybrid_V2RK4(prcp_hybrid, et_hybrid, tmean_hybrid, calibrated_parameters) ### note vor V2 we need et here instead!
    # hybridmodel_v1.to(device) # put back if using GPU

    # loss function
    if loss_func == "MSE":
        criterion = nn.MSELoss()
        print("Using MSE loss func")
    elif loss_func == "RMSE":
        criterion = RMSLELoss()
        print("Using RMSE loss func")
    elif loss_func == "KGE":
        criterion = KGELossMOD()
        print("Using KGE loss func")
    else:
        print("no valid loss func defined")

    # Define the optimizer, which optimzer should be used? Adam? SGD?
    optimizer = optim.Adam(hybridmodel_v2.parameters(), lr=start_lr) # what is a good lr to start with?


    # learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    torch.autograd.set_detect_anomaly(True)

    train_losses_v2 = []
    val_losses_v2 = []

    best_loss = 50000000
    best_model_state_hybrid_v2 = None ### what was this for again?

    # Train the model
    for epoch in range(num_epochs):

        # params
        params_hybrid_v2 = hybridmodel_v2.state_dict()

        # forward pass training data
        qsim_train_hybrid_v2 = hybridmodel_v2.forward(train_period, S0_hybrid_v2)

        # forward pass on validation data
        with torch.no_grad():
            qsim_val_hybrid_v2 = hybridmodel_v2.forward(val_period, S0_hybrid_v2)




        #### remove the last time setp of qobs
        qobs_train = qobs_mm_hybrid[train_period_int[:-1]]
        qobs_val = qobs_mm_hybrid[val_period_int[:-1]]


        ############################### downsampling here
        qsim_train_hybrid_v2_itp = interpolate_downsample(qobs_train, qsim_train_hybrid_v2)
        qsim_val_hybrid_v2_itp = interpolate_downsample(qobs_val, qsim_val_hybrid_v2)



        # Added warmup period here
        # [-1] for qobs_mm is a fix here as qsim is one element too short because of the way it is calculated and appended
        ##### train_loss = criterion(qsim_train_hybrid_v1[365:], qobs_mm_hybrid[train_period_int[365:-1]]) # make sure to NOT train on normalized observations!
        ##### val_loss = criterion(qsim_val_hybrid_v1[365:], qobs_mm_hybrid[val_period_int[365:-1]]) # make sure to NOT train on normalized observations!

        train_loss = criterion(qsim_train_hybrid_v2_itp[365:], qobs_train[365:]) # make sure to NOT train on normalized observations!
        val_loss = criterion(qsim_val_hybrid_v2_itp[365:], qobs_val[365:]) # make sure to NOT train on normalized observations!



        # Added warmup period here
        # [-1] for qobs_mm is a fix here as qsim is one element too short because of the way it is calculated and appended
        train_loss = criterion(qsim_train_hybrid_v2_itp[365:], qobs_mm_hybrid[train_period_int[365:-1]]) # make sure to NOT train on normalized observations!
        val_loss = criterion(qsim_val_hybrid_v2_itp[365:], qobs_mm_hybrid[val_period_int[365:-1]]) # make sure to NOT train on normalized observations!

        # append train and validation loss
        train_losses_v2.append(train_loss.item())
        val_losses_v2.append(val_loss.item())


        # backward pass
        optimizer.zero_grad()
        train_loss.backward() #retain_graph=True

        print(hybridmodel_v2.layer1.weight.grad)

        optimizer.step()

        # update lr schedule
        scheduler.step(train_loss) # train loss

        # update and save best model (based on validation loss)
        if val_loss.item() < best_loss:
            print("Accessed loop")
            best_loss = val_loss.item()
            best_model_state_hybrid_v2 = copy.deepcopy(hybridmodel_v2.state_dict())


        print(f"Epoch {epoch}: Train loss = {train_loss.item()} Validation loss = {val_loss.item()}")


    # save trained model parameters
    if environment=='local':
        torch.save(best_model_state_hybrid_v2, f'models/hybrid_v1_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{lr_start}.pth')
    if environment=='colab':
        torch.save(best_model_state_hybrid_v2, f'/content/drive/MyDrive/msc_thesis/python/notebooks/colab/HybridV2/trained_parameters/hybrid_v2_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{start_lr}_epoch_{num_epochs}.pth')


# In[ ]:


# user input
num_epochs = 25
solver = "rk4" # "euler" or "rk4"
loss_func = "MSE" # or "RMSE" or if used "NSE", "KGE"
start_lr = 0.01 # 0.01


# In[ ]:


# representative catchment IDs for LamaH-CE
IDs_red = [334, 743, 439, 24, 330, 75, 383, 79]


# In[ ]:


# now all in one
results_hybrid_dfs = []

# loop that rules it all
for id_num in IDs_red:
    print("ID:", id_num)

    # define initial storages
    S0_hybrid_v2 = torch.tensor([0.0, 0.0])
    S0_hybrid_v2 = torch.unsqueeze(S0_hybrid_v2, 0) # reshape

    # load the data
    # load df hybrid # IMPORTANT: full period from start train to stop test!
    if environment == "colab":
        df_hybrid = load_data_new(path_vars_lamah_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_lamah, start_date=start_train, stop_date=stop_test, dataset="lamah", qobs_mm=True)
    if environment == "local":
        df_hybrid = load_data_new(path_vars_lamah_local, path_qobs_lamah_local, path_catchment_attrs_colab, ID=id_num, variables=variables_lamah, start_date=start_train, stop_date=stop_test,  dataset="lamah", qobs_mm=True)


    # create tensors from the dataframe
    tensors_hybrid = df_to_tensors(df_hybrid, name_suffix="hybrid")
    # Processing tensors
    for name, tensor in tensors_hybrid.items():
        # Moving to GPU
        tensor = tensor.to(device)
        tensor = tensor.to(dtype=torch.float32)
        tensor.requires_grad = False  # inplace operation
        locals()[name] = tensor


    # load the parameters for the conceptual model part
    if environment == "colab":
        path = os.path.join(path_conceptual_v2_colab, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)
    if environment == "local":
        path = os.path.join(path_conceptual_v2_local, "calibrated_parameters/1981_2001/", f"calibrated_parameters_ID_{id_num}_method_PSO_swarm_size_20.npy")
        calibrated_parameters = np.load(path)

    # instantiate the model
    hybridmodel_v2 = Hybrid_V2RK4(prcp_hybrid, et_hybrid, tmean_hybrid, calibrated_parameters) ### note vor V2 we need et here instead!


    # load the model with the trained weights and biases ### add local if needed
    if environment == "colab":
        path_nn = os.path.join(path_hybrid_v2_colab, "trained_parameters", f"hybrid_v2_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{start_lr}_epoch_{num_epochs}.pth")
        hybridmodel_v2.load_state_dict(torch.load(path_nn))

    # training
    with torch.no_grad():
        qsim_train_hybrid_v2 = hybridmodel_v2.forward(train_period, S0_hybrid_v2)

    # validation
    with torch.no_grad():
        qsim_val_hybrid_v2 = hybridmodel_v2.forward(val_period, S0_hybrid_v2)

    # testing
    with torch.no_grad():
        qsim_test_hybrid_v2 = hybridmodel_v2.forward(test_period, S0_hybrid_v2)

    ### save observations
    # divide the observation timeseries into training, validation and testing
    qobs_mm_training = qobs_mm_hybrid[train_period_int[:-1]]
    qobs_mm_validation = qobs_mm_hybrid[val_period_int[:-1]]
    qobs_mm_testing = qobs_mm_hybrid[test_period_int[:-1]]

    # save the observation timeseries
    if environment == "colab":
        path_qobs_mm_training = os.path.join(path_hybrid_v2_colab, "discharge/training/observations/", f"qobs_mm_train_ID_{id_num}.csv")
        path_qobs_mm_validation = os.path.join(path_hybrid_v2_colab, "discharge/validation/observations/", f"qobs_mm_train_ID_{id_num}.csv")
        path_qobs_mm_testing = os.path.join(path_hybrid_v2_colab, "discharge/testing/observations/", f"qobs_mm_train_ID_{id_num}.csv")

    if environment == "local":
        ### add if needed
        """
        path_qobs_mm_cal = os.path.join(path_conceptual_v2_local, "discharge/cal/observations/", f"qobs_mm_cal_ID_{id_num}.csv")
        path_qobs_mm_test = os.path.join(path_conceptual_v2_local, "discharge/test/observations/", f"qobs_mm_test_ID_{id_num}.csv")
        """

    # training, validation, testing
    np.savetxt(path_qobs_mm_training, qobs_mm_training, delimiter=",") # [:-1]
    np.savetxt(path_qobs_mm_validation, qobs_mm_validation, delimiter=",")
    np.savetxt(path_qobs_mm_testing, qobs_mm_testing, delimiter=",")


    ### interpolate
    ### now interpolate the train, val and test predictions to make them the same shape as observations and predictions using the euler solver
    qsim_train_hybrid_v2_itp = interpolate_downsample(qobs_mm_training, qsim_train_hybrid_v2)
    qsim_val_hybrid_v2_itp = interpolate_downsample(qobs_mm_validation, qsim_val_hybrid_v2)
    qsim_test_hybrid_v2_itp = interpolate_downsample(qobs_mm_testing, qsim_test_hybrid_v2)


    ### save predictions
    # save training, validation and testing
    if environment == "colab":
        path_euler_training = os.path.join(path_hybrid_v2_colab, "discharge/training/euler/", f"qsim_hybrid_training_euler_ID_{id_num}_loss_func_{loss_func}_epoch_{num_epochs}_solver_{solver}_lr_{start_lr}.csv")
        path_euler_validation = os.path.join(path_hybrid_v2_colab, "discharge/validation/euler/", f"qsim_hybrid_validation_euler_ID_{id_num}_loss_func_{loss_func}_epoch_{num_epochs}_solver_{solver}_lr_{start_lr}.csv")
        path_euler_testing = os.path.join(path_hybrid_v2_colab, "discharge/testing/euler/", f"qsim_hybrid_testing_euler_ID_{id_num}_loss_func_{loss_func}_epoch_{num_epochs}_solver_{solver}_lr_{start_lr}.csv")

    if environment == "local":
        ### add if needed
        """
        path_euler_cal = os.path.join(path_hybrid_local, "discharge/cal/euler/", f"qsim_cal_euler_ID_{id_num}.csv")
        path_euler_test = os.path.join(path_hybrid_local, "discharge/test/euler/", f"qsim_test_euler_ID_{id_num}.csv")
        """

    # training, validation, testing
    np.savetxt(path_euler_training, qsim_train_hybrid_v2_itp, delimiter=",")
    np.savetxt(path_euler_validation, qsim_val_hybrid_v2_itp, delimiter=",")
    np.savetxt(path_euler_testing, qsim_test_hybrid_v2_itp, delimiter=",")


    # calculate evaluation metrics BUT leave 365 days for warm-up
    nse_loss = NSELoss()
    mse_loss = MSELoss()
    kge_loss = KGELoss()

    # loss training
    nse_loss_rk4_training = nse_loss(qsim_train_hybrid_v2_itp[365:], qobs_mm_training[365:].cpu())
    mse_loss_rk4_training = mse_loss(qsim_train_hybrid_v2_itp[365:], qobs_mm_training[365:].cpu())
    kge_loss_rk4_training = kge_loss(qsim_train_hybrid_v2_itp[365:], qobs_mm_training[365:].cpu())

    # loss validation
    nse_loss_rk4_validation = nse_loss(qsim_val_hybrid_v2_itp[365:], qobs_mm_validation[365:].cpu())
    mse_loss_rk4_validation = mse_loss(qsim_val_hybrid_v2_itp[365:], qobs_mm_validation[365:].cpu())
    kge_loss_rk4_validation = kge_loss(qsim_val_hybrid_v2_itp[365:], qobs_mm_validation[365:].cpu())

    # loss test
    nse_loss_rk4_testing = nse_loss(qsim_test_hybrid_v2_itp[365:], qobs_mm_testing[365:].cpu())
    mse_loss_rk4_testing = mse_loss(qsim_test_hybrid_v2_itp[365:], qobs_mm_testing[365:].cpu())
    kge_loss_rk4_testing = kge_loss(qsim_test_hybrid_v2_itp[365:], qobs_mm_testing[365:].cpu())

    # save evaluation metrics (ideally to one large file )
    row = {
        "id_num": id_num,
        "start calibration": start_cal,
        "stop calibration": stop_cal,
        "start test": start_test,
        "stop test": stop_test,
        "nse_loss_rk4_train": nse_loss_rk4_training,
        "mse_loss_rk4_train": mse_loss_rk4_training,
        "kge_loss_rk4_train": kge_loss_rk4_training,
        "nse_loss_rk4_val": nse_loss_rk4_validation,
        "mse_loss_rk4_val": mse_loss_rk4_validation,
        "kge_loss_rk4_val": kge_loss_rk4_validation,
        "nse_loss_rk4_test": nse_loss_rk4_testing,
        "mse_loss_rk4_test": mse_loss_rk4_testing,
        "kge_loss_rk4_test": kge_loss_rk4_testing
    }

    # Create a DataFrame for the current iteration
    iteration_df = pd.DataFrame([row])

    # Append the iteration DataFrame to the list
    results_hybrid_dfs.append(iteration_df)

# Concatenate all DataFrames into one
results_hybrid_df = pd.concat(results_hybrid_dfs, ignore_index=True)


# Save the results DataFrame to a CSV file
if environment == "colab":
    path_results = os.path.join(path_hybrid_v2_colab, "evaluation_metrics/", f"hybrid_v2_results_solver_{solver}_loss_{loss_func}_epoch_{num_epochs}_lr_{start_lr}.csv")

if environment == "local":
    path_results = os.path.join(path_hybrid_v2_local, "evaluation_metrics/", f"hybrid_v2_results_ID_{id_num}_solver_{solver}.csv")

results_hybrid_df.to_csv(path_results, index=False)


# ## 4.4 LSTM

# ### 4.4.1 LSTM E-OBS

# In[ ]:


# user input
epochs = 50 # 30

# model parametres
input_size = 3 # number of variables (meteorological, not including qobs)
hidden_size = 10
output_size = 1
hidden_layers = 1
start_lr = 0.01
dropout_prob = 0.3
batch_size = 64
loss_func = "MSE"


# In[ ]:



results_lstm_dfs = []

# loop that rules it all
for id_num in IDs_red:
    print("ID:", id_num)

    # load data
    # load data for the full time period so start train to stop test, it is split later!
    df_lstm_eobs = load_data_new(path_vars_eobs_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_eobs_lstm_small, start_date=start_train, stop_date=stop_test, dataset='eobs', delimiter=';', qobs_mm='True')

    # prepare lstm data with function from script
    X_train_lstm_eobs, X_val_lstm_eobs, X_test_lstm_eobs, y_train_lstm_eobs, y_val_lstm_eobs, y_test_lstm_eobs, scaler_eobs = prep_data_lstm_new(df_lstm_eobs, start_train, stop_train, start_val, stop_val, start_test, stop_test, tensors=True)

    # push tensors to GPU if available
    X_train_lstm_eobs = X_train_lstm_eobs.to(device)
    X_val_lstm_eobs = X_val_lstm_eobs.to(device)
    X_test_lstm_eobs = X_test_lstm_eobs.to(device)
    y_train_lstm_eobs = y_train_lstm_eobs.to(device)
    y_val_lstm_eobs = y_val_lstm_eobs.to(device)
    y_test_lstm_eobs = y_test_lstm_eobs.to(device)


    # initialize
    model = LSTMModel(input_size, hidden_size, output_size, dropout_prob=dropout_prob)

    # push to GPU
    model.to(device)

    # loss function
    if loss_func == "MSE":
        criterion = nn.MSELoss()
        print("Using MSE loss func")
    elif loss_func == "RMSE":
        criterion = RMSLELoss()
        print("Using RMSE loss func")
    elif loss_func == "KGE":
        criterion = KGELossMOD()
        print("Using KGE loss func")
    else:
        print("no valid loss func defined")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)

    #### define lr scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # train
    model.train()

    best_val_loss = 500000 # just set to a high number ####

    for epoch in range(epochs):
        for i in range(0, len(X_train_lstm_eobs), batch_size):
            optimizer.zero_grad()
            batch_X = torch.Tensor(X_train_lstm_eobs[i:i+batch_size]).to(device)
            batch_y = torch.Tensor(y_train_lstm_eobs[i:i+batch_size]).to(device)
            y_pred_train = model(batch_X)
            loss = criterion(y_pred_train, batch_y)
            loss.backward()
            optimizer.step()


        scheduler.step(loss) ## update the scheduler, is this the right location?


        # also calculate validation loss
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for validation
            y_pred_val = model(torch.Tensor(X_val_lstm_eobs).to(device))
            loss_val = criterion(y_pred_val, torch.Tensor(y_val_lstm_eobs).to(device))
        model.train()  # Set the model back to training mode

        # make sure to save the best validation loss as state_dict

        # Check if the current validation loss is the best so far
        if loss_val < best_val_loss:
            print("Accessed loop")
            best_val_loss = loss_val
            # Save the model's state dictionary to a file
            best_model_state_lstm_v1 = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}, loss train: {loss.item()}, loss val: {loss_val.item()}')

        # save trained model parameters
    if environment=='local': ### fix if using local
        #torch.save(best_model_state_hybrid_v1, f'models/hybrid_v1_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{start_lr}.pth')
    if environment=='colab':
        torch.save(best_model_state_lstm_v1, f'/content/drive/MyDrive/msc_thesis/python/notebooks/colab/LSTMV1/trained_parameters/lstm_v1_id_{id_num}_loss_{loss_func}_lr_{start_lr}.pth')

    # load model here with best parameters?
    load_model = torch.load(f'/content/drive/MyDrive/msc_thesis/python/notebooks/colab/LSTMV1/trained_parameters/lstm_v1_id_{id_num}_loss_{loss_func}_lr_{start_lr}.pth')
    best_model = LSTMModel(input_size, hidden_size, output_size, dropout_prob=dropout_prob)
    best_model.load_state_dict(load_model)


    # evaluate model
    best_model.eval()

    # prediction on training data
    with torch.no_grad():
        y_pred_train_lstm_eobs = best_model(torch.Tensor(X_train_lstm_eobs))

    # prediction on validation data
    with torch.no_grad():
        y_pred_val_lstm_eobs = best_model(torch.Tensor(X_val_lstm_eobs))

    # prediction on testing data
    with torch.no_grad():
        y_pred_test_lstm_eobs = best_model(torch.Tensor(X_test_lstm_eobs))

    # rescaling # NOTE: this is a bit of a workaround as the data is scaled before it is plit into X and y, so the scaler expects 4 input tensors could also be fixed by changing the script
    # rescale obesrvations
    train_meteo_obs = torch.cat((X_train_lstm_eobs[:,0,:], y_train_lstm_eobs), dim=1) # concat X and y train (so meteo + obs) ### Taking the first of the 365 dimensions, is that okay???
    val_meteo_obs = torch.cat((X_val_lstm_eobs[:,0,:], y_val_lstm_eobs), dim=1)
    test_meteo_obs = torch.cat((X_test_lstm_eobs[:,0,:], y_test_lstm_eobs), dim=1) # concat X and y test (so meteo + obs)

    ### added .cpu()
    train_meteo_obs_rs = scaler_eobs.inverse_transform(train_meteo_obs.cpu()) # rescale training
    val_meteo_obs_rs = scaler_eobs.inverse_transform(val_meteo_obs.cpu())
    test_meteo_obs_rs = scaler_eobs.inverse_transform(test_meteo_obs.cpu()) # rescale testing


    train_obs_rs = train_meteo_obs_rs[:,-1] # get the rescaled training observations, complicated I know
    val_obs_rs = val_meteo_obs_rs[:,-1]
    test_obs_rs = test_meteo_obs_rs[:,-1]

    # make observaitons tensors (they are not tensors because of the minmax scaling from scipy)
    train_obs_rs_tensor = torch.tensor(train_obs_rs)
    val_obs_rs_tensor = torch.tensor(val_obs_rs)
    test_obs_rs_tensor = torch.tensor(test_obs_rs)

    # rescale prediction
    train_meteo_pred = torch.cat((X_train_lstm_eobs[:,0,:], y_pred_train_lstm_eobs), dim=1) # concat X train and prediction for y
    val_meteo_pred = torch.cat((X_val_lstm_eobs[:,0,:], y_pred_val_lstm_eobs), dim=1)
    test_meteo_pred = torch.cat((X_test_lstm_eobs[:,0,:], y_pred_test_lstm_eobs), dim=1) # concat X test and prediction for y

    ### added .cpu()
    train_meteo_pred_rs = scaler_eobs.inverse_transform(train_meteo_pred.cpu()) # rescale training
    val_meteo_pred_rs = scaler_eobs.inverse_transform(val_meteo_pred.cpu())
    test_meteo_pred_rs = scaler_eobs.inverse_transform(test_meteo_pred.cpu()) # rescale testing

    ### train_meteo_pred_rs = scaler_eobs.inverse_transform(train_meteo_pred) # rescale training
    ### val_meteo_pred_rs = scaler_eobs.inverse_transform(val_meteo_pred)
    ### test_meteo_pred_rs = scaler_eobs.inverse_transform(test_meteo_pred) # rescale testing
    train_pred_rs = train_meteo_pred_rs[:,-1] # get the rescaled training observations, complicated I know
    val_pred_rs = val_meteo_pred_rs[:,-1]
    test_pred_rs = test_meteo_pred_rs[:,-1]

    # make predictions tensors, also not super efficient but does not matter for now
    train_pred_rs_tensor = torch.tensor(train_pred_rs)
    val_pred_rs_tensor = torch.tensor(val_pred_rs)
    test_pred_rs_tensor = torch.tensor(test_pred_rs)

    # save training, validation and testing
    if environment == "colab":
        path_euler_training = os.path.join(path_lstm_v1_colab, "discharge/training/euler/", f"qsim_lstm_training_euler_ID_{id_num}_loss_func_{loss_func}.csv")
        path_euler_validation = os.path.join(path_lstm_v1_colab, "discharge/validation/euler/", f"qsim_lstm_validation_euler_ID_{id_num}_loss_func_{loss_func}.csv")
        path_euler_testing = os.path.join(path_lstm_v1_colab, "discharge/testing/euler/", f"qsim_lstm_testing_euler_ID_{id_num}_loss_func_{loss_func}.csv")

    if environment == "local":
        ### add if needed
        """
        path_euler_cal = os.path.join(path_hybrid_local, "discharge/cal/euler/", f"qsim_cal_euler_ID_{id_num}.csv")
        path_euler_test = os.path.join(path_hybrid_local, "discharge/test/euler/", f"qsim_test_euler_ID_{id_num}.csv")
        """

    # training, validation, testing
    np.savetxt(path_euler_training, train_pred_rs, delimiter=",")
    np.savetxt(path_euler_validation, val_pred_rs, delimiter=",")
    np.savetxt(path_euler_testing, test_pred_rs, delimiter=",")

    # save the observation timeseries
    if environment == "colab":
        path_qobs_mm_training = os.path.join(path_lstm_v1_colab, "discharge/training/observations/", f"qobs_mm_train_ID_{id_num}.csv")
        path_qobs_mm_validation = os.path.join(path_lstm_v1_colab, "discharge/validation/observations/", f"qobs_mm_train_ID_{id_num}.csv")
        path_qobs_mm_testing = os.path.join(path_lstm_v1_colab, "discharge/testing/observations/", f"qobs_mm_train_ID_{id_num}.csv")

    if environment == "local":
        ### add if needed
        """
        path_qobs_mm_cal = os.path.join(path_conceptual_v2_local, "discharge/cal/observations/", f"qobs_mm_cal_ID_{id_num}.csv")
        path_qobs_mm_test = os.path.join(path_conceptual_v2_local, "discharge/test/observations/", f"qobs_mm_test_ID_{id_num}.csv")
        """

    # training, validation, testing
    np.savetxt(path_qobs_mm_training, train_obs_rs, delimiter=",") # [:-1]
    np.savetxt(path_qobs_mm_validation, val_obs_rs, delimiter=",")
    np.savetxt(path_qobs_mm_testing, test_obs_rs, delimiter=",")


    # calculate evaluation metrics BUT leave 365 days for warm-up
    nse_loss = NSELoss()
    mse_loss = MSELoss()
    kge_loss = KGELoss()

    # loss training
    nse_loss_training = nse_loss(train_pred_rs_tensor[365:], train_obs_rs_tensor[365:])
    mse_loss_training = mse_loss(train_pred_rs_tensor[365:], train_obs_rs_tensor[365:])
    kge_loss_training = kge_loss(train_pred_rs_tensor[365:], train_obs_rs_tensor[365:])

    # loss validation
    nse_loss_validation = nse_loss(val_pred_rs_tensor[365:], val_obs_rs_tensor[365:])
    mse_loss_validation = mse_loss(val_pred_rs_tensor[365:], val_obs_rs_tensor[365:])
    kge_loss_validation = kge_loss(val_pred_rs_tensor[365:], val_obs_rs_tensor[365:])
    # loss test
    nse_loss_testing = nse_loss(test_pred_rs_tensor[365:], test_obs_rs_tensor[365:])
    mse_loss_testing = mse_loss(test_pred_rs_tensor[365:], test_obs_rs_tensor[365:])
    kge_loss_testing = kge_loss(test_pred_rs_tensor[365:], test_obs_rs_tensor[365:])

    # save evaluation metrics (ideally to one large file )
    row = {
        "id_num": id_num,
        "start training": start_train,
        "stop training": stop_train,
        "start test": start_test,
        "stop test": stop_test,
        "nse_loss_train": nse_loss_training,
        "mse_loss_train": mse_loss_training,
        "kge_loss_train": kge_loss_training,
        "nse_loss_val": nse_loss_validation,
        "mse_loss_val": mse_loss_validation,
        "kge_loss_val": kge_loss_validation,
        "nse_loss_test": nse_loss_testing,
        "mse_loss_test": mse_loss_testing,
        "kge_loss_test": kge_loss_testing
    }

    # Create a DataFrame for the current iteration
    iteration_df = pd.DataFrame([row])

    # Append the iteration DataFrame to the list
    results_lstm_dfs.append(iteration_df)

# Concatenate all DataFrames into one
results_lstm_df = pd.concat(results_lstm_dfs, ignore_index=True)


# Save the results DataFrame to a CSV file
if environment == "colab":
    path_results = os.path.join(path_lstm_v1_colab, "evaluation_metrics/", f"lstm_v1_results.csv")

if environment == "local":
    path_results = os.path.join(path_lstm_v1_local, "evaluation_metrics/", f"lstm_v1_results.csv")

results_lstm_df.to_csv(path_results, index=False)



# #### 4.4.1.1 LSTM Large E-OBS 

# In[ ]:


# user input
epochs = 50 # 30

# model parametres
input_size = 6 # number of variables (meteorological, not including qobs)
hidden_size = 10
output_size = 1
hidden_layers = 1
start_lr = 0.01
dropout_prob = 0.3
batch_size = 64
loss_func = "MSE"


# In[ ]:


results_lstm_dfs = []

# loop that rules it all
for id_num in IDs_red:
    print("ID:", id_num)

    # load data
    # load data for the full time period so start train to stop test, it is split later!
    df_lstm_eobs = load_data_new(path_vars_eobs_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_eobs_lstm_large, start_date=start_train, stop_date=stop_test, dataset='eobs', delimiter=';', qobs_mm='True')

    # prepare lstm data with function from script
    X_train_lstm_eobs, X_val_lstm_eobs, X_test_lstm_eobs, y_train_lstm_eobs, y_val_lstm_eobs, y_test_lstm_eobs, scaler_eobs = prep_data_lstm_new(df_lstm_eobs, start_train, stop_train, start_val, stop_val, start_test, stop_test, tensors=True)

    # push tensors to GPU if available
    X_train_lstm_eobs = X_train_lstm_eobs.to(device)
    X_val_lstm_eobs = X_val_lstm_eobs.to(device)
    X_test_lstm_eobs = X_test_lstm_eobs.to(device)
    y_train_lstm_eobs = y_train_lstm_eobs.to(device)
    y_val_lstm_eobs = y_val_lstm_eobs.to(device)
    y_test_lstm_eobs = y_test_lstm_eobs.to(device)


    # initialize
    model = LSTMModel(input_size, hidden_size, output_size, dropout_prob=dropout_prob)

    # push to GPU
    model.to(device)

    # loss function
    if loss_func == "MSE":
        criterion = nn.MSELoss()
        print("Using MSE loss func")
    elif loss_func == "RMSE":
        criterion = RMSLELoss()
        print("Using RMSE loss func")
    elif loss_func == "KGE":
        criterion = KGELossMOD()
        print("Using KGE loss func")
    else:
        print("no valid loss func defined")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)

    #### define lr scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # train
    model.train()

    best_val_loss = 500000 # just set to a high number ####

    for epoch in range(epochs):
        for i in range(0, len(X_train_lstm_eobs), batch_size):
            optimizer.zero_grad()
            batch_X = torch.Tensor(X_train_lstm_eobs[i:i+batch_size]).to(device)
            batch_y = torch.Tensor(y_train_lstm_eobs[i:i+batch_size]).to(device)
            y_pred_train = model(batch_X)
            loss = criterion(y_pred_train, batch_y)
            loss.backward()
            optimizer.step()


        scheduler.step(loss) ## update the scheduler, is this the right location?


        # also calculate validation loss
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for validation
            y_pred_val = model(torch.Tensor(X_val_lstm_eobs).to(device))
            loss_val = criterion(y_pred_val, torch.Tensor(y_val_lstm_eobs).to(device))
        model.train()  # Set the model back to training mode

        # make sure to save the best validation loss as state_dict

        # Check if the current validation loss is the best so far
        if loss_val < best_val_loss:
            print("Accessed loop")
            best_val_loss = loss_val
            # Save the model's state dictionary to a file
            best_model_state_lstm_v1 = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}, loss train: {loss.item()}, loss val: {loss_val.item()}')

        # save trained model parameters
    if environment=='local': ### fix if using local
        torch.save(best_model_state_hybrid_v1, f'models/hybrid_v1_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{start_lr}.pth')
    if environment=='colab':
        torch.save(best_model_state_lstm_v1, f'/content/drive/MyDrive/msc_thesis/python/notebooks/colab/LSTMV1Large/trained_parameters/lstm_v1_id_{id_num}_loss_{loss_func}_lr_{start_lr}.pth')

    # load model here with best parameters?
    load_model = torch.load(f'/content/drive/MyDrive/msc_thesis/python/notebooks/colab/LSTMV1Large/trained_parameters/lstm_v1_id_{id_num}_loss_{loss_func}_lr_{start_lr}.pth')
    best_model = LSTMModel(input_size, hidden_size, output_size, dropout_prob=dropout_prob)
    best_model.load_state_dict(load_model)


    # evaluate model
    best_model.eval()

    # prediction on training data
    with torch.no_grad():
        y_pred_train_lstm_eobs = best_model(torch.Tensor(X_train_lstm_eobs))

    # prediction on validation data
    with torch.no_grad():
        y_pred_val_lstm_eobs = best_model(torch.Tensor(X_val_lstm_eobs))

    # prediction on testing data
    with torch.no_grad():
        y_pred_test_lstm_eobs = best_model(torch.Tensor(X_test_lstm_eobs))

    # rescaling # NOTE: this is a bit of a workaround as the data is scaled before it is plit into X and y, so the scaler expects 4 input tensors could also be fixed by changing the script
    # rescale obesrvations
    train_meteo_obs = torch.cat((X_train_lstm_eobs[:,0,:], y_train_lstm_eobs), dim=1) # concat X and y train (so meteo + obs) ### Taking the first of the 365 dimensions, is that okay???
    val_meteo_obs = torch.cat((X_val_lstm_eobs[:,0,:], y_val_lstm_eobs), dim=1)
    test_meteo_obs = torch.cat((X_test_lstm_eobs[:,0,:], y_test_lstm_eobs), dim=1) # concat X and y test (so meteo + obs)

    ### added .cpu()
    train_meteo_obs_rs = scaler_eobs.inverse_transform(train_meteo_obs.cpu()) # rescale training
    val_meteo_obs_rs = scaler_eobs.inverse_transform(val_meteo_obs.cpu())
    test_meteo_obs_rs = scaler_eobs.inverse_transform(test_meteo_obs.cpu()) # rescale testing

    ### train_meteo_obs_rs = scaler_eobs.inverse_transform(train_meteo_obs) # rescale training
    ### val_meteo_obs_rs = scaler_eobs.inverse_transform(val_meteo_obs)
    ###test_meteo_obs_rs = scaler_eobs.inverse_transform(test_meteo_obs) # rescale testing

    train_obs_rs = train_meteo_obs_rs[:,-1] # get the rescaled training observations, complicated I know
    val_obs_rs = val_meteo_obs_rs[:,-1]
    test_obs_rs = test_meteo_obs_rs[:,-1]

    # make observaitons tensors (they are not tensors because of the minmax scaling from scipy)
    train_obs_rs_tensor = torch.tensor(train_obs_rs)
    val_obs_rs_tensor = torch.tensor(val_obs_rs)
    test_obs_rs_tensor = torch.tensor(test_obs_rs)

    # rescale prediction
    train_meteo_pred = torch.cat((X_train_lstm_eobs[:,0,:], y_pred_train_lstm_eobs), dim=1) # concat X train and prediction for y
    val_meteo_pred = torch.cat((X_val_lstm_eobs[:,0,:], y_pred_val_lstm_eobs), dim=1)
    test_meteo_pred = torch.cat((X_test_lstm_eobs[:,0,:], y_pred_test_lstm_eobs), dim=1) # concat X test and prediction for y

    ### added .cpu()
    train_meteo_pred_rs = scaler_eobs.inverse_transform(train_meteo_pred.cpu()) # rescale training
    val_meteo_pred_rs = scaler_eobs.inverse_transform(val_meteo_pred.cpu())
    test_meteo_pred_rs = scaler_eobs.inverse_transform(test_meteo_pred.cpu()) # rescale testing

    ### train_meteo_pred_rs = scaler_eobs.inverse_transform(train_meteo_pred) # rescale training
    ### val_meteo_pred_rs = scaler_eobs.inverse_transform(val_meteo_pred)
    ### test_meteo_pred_rs = scaler_eobs.inverse_transform(test_meteo_pred) # rescale testing
    train_pred_rs = train_meteo_pred_rs[:,-1] # get the rescaled training observations, complicated I know
    val_pred_rs = val_meteo_pred_rs[:,-1]
    test_pred_rs = test_meteo_pred_rs[:,-1]

    # make predictions tensors, also not super efficient but does not matter for now
    train_pred_rs_tensor = torch.tensor(train_pred_rs)
    val_pred_rs_tensor = torch.tensor(val_pred_rs)
    test_pred_rs_tensor = torch.tensor(test_pred_rs)

    # save training, validation and testing
    if environment == "colab":
        path_euler_training = os.path.join(path_lstm_v1_large_colab, "discharge/training/euler/", f"qsim_lstm_training_euler_ID_{id_num}_loss_func_{loss_func}.csv")
        path_euler_validation = os.path.join(path_lstm_v1_large_colab, "discharge/validation/euler/", f"qsim_lstm_validation_euler_ID_{id_num}_loss_func_{loss_func}.csv")
        path_euler_testing = os.path.join(path_lstm_v1_large_colab, "discharge/testing/euler/", f"qsim_lstm_testing_euler_ID_{id_num}_loss_func_{loss_func}.csv")

    if environment == "local":
        ### add if needed
        """
        path_euler_cal = os.path.join(path_hybrid_local, "discharge/cal/euler/", f"qsim_cal_euler_ID_{id_num}.csv")
        path_euler_test = os.path.join(path_hybrid_local, "discharge/test/euler/", f"qsim_test_euler_ID_{id_num}.csv")
        """

    # training, validation, testing
    np.savetxt(path_euler_training, train_pred_rs, delimiter=",")
    np.savetxt(path_euler_validation, val_pred_rs, delimiter=",")
    np.savetxt(path_euler_testing, test_pred_rs, delimiter=",")

    # save the observation timeseries
    if environment == "colab":
        path_qobs_mm_training = os.path.join(path_lstm_v1_large_colab, "discharge/training/observations/", f"qobs_mm_train_ID_{id_num}.csv")
        path_qobs_mm_validation = os.path.join(path_lstm_v1_large_colab, "discharge/validation/observations/", f"qobs_mm_train_ID_{id_num}.csv")
        path_qobs_mm_testing = os.path.join(path_lstm_v1_large_colab, "discharge/testing/observations/", f"qobs_mm_train_ID_{id_num}.csv")

    if environment == "local":
        ### add if needed
        """
        path_qobs_mm_cal = os.path.join(path_conceptual_v2_local, "discharge/cal/observations/", f"qobs_mm_cal_ID_{id_num}.csv")
        path_qobs_mm_test = os.path.join(path_conceptual_v2_local, "discharge/test/observations/", f"qobs_mm_test_ID_{id_num}.csv")
        """

    # training, validation, testing
    np.savetxt(path_qobs_mm_training, train_obs_rs, delimiter=",") # [:-1]
    np.savetxt(path_qobs_mm_validation, val_obs_rs, delimiter=",")
    np.savetxt(path_qobs_mm_testing, test_obs_rs, delimiter=",")


    # calculate evaluation metrics BUT leave 365 days for warm-up
    nse_loss = NSELoss()
    mse_loss = MSELoss()
    kge_loss = KGELoss()

    # loss training
    nse_loss_training = nse_loss(train_pred_rs_tensor[365:], train_obs_rs_tensor[365:])
    mse_loss_training = mse_loss(train_pred_rs_tensor[365:], train_obs_rs_tensor[365:])
    kge_loss_training = kge_loss(train_pred_rs_tensor[365:], train_obs_rs_tensor[365:])

    # loss validation
    nse_loss_validation = nse_loss(val_pred_rs_tensor[365:], val_obs_rs_tensor[365:])
    mse_loss_validation = mse_loss(val_pred_rs_tensor[365:], val_obs_rs_tensor[365:])
    kge_loss_validation = kge_loss(val_pred_rs_tensor[365:], val_obs_rs_tensor[365:])
    # loss test
    nse_loss_testing = nse_loss(test_pred_rs_tensor[365:], test_obs_rs_tensor[365:])
    mse_loss_testing = mse_loss(test_pred_rs_tensor[365:], test_obs_rs_tensor[365:])
    kge_loss_testing = kge_loss(test_pred_rs_tensor[365:], test_obs_rs_tensor[365:])

    # save evaluation metrics (ideally to one large file )
    row = {
        "id_num": id_num,
        "start training": start_train,
        "stop training": stop_train,
        "start test": start_test,
        "stop test": stop_test,
        "nse_loss_train": nse_loss_training,
        "mse_loss_train": mse_loss_training,
        "kge_loss_train": kge_loss_training,
        "nse_loss_val": nse_loss_validation,
        "mse_loss_val": mse_loss_validation,
        "kge_loss_val": kge_loss_validation,
        "nse_loss_test": nse_loss_testing,
        "mse_loss_test": mse_loss_testing,
        "kge_loss_test": kge_loss_testing
    }

    # Create a DataFrame for the current iteration
    iteration_df = pd.DataFrame([row])

    # Append the iteration DataFrame to the list
    results_lstm_dfs.append(iteration_df)

# Concatenate all DataFrames into one
results_lstm_df = pd.concat(results_lstm_dfs, ignore_index=True)


# Save the results DataFrame to a CSV file
if environment == "colab":
    path_results = os.path.join(path_lstm_v1_large_colab, "evaluation_metrics/", f"lstm_v1_large_results.csv")

if environment == "local":
    path_results = os.path.join(path_lstm_v1_large_local, "evaluation_metrics/", f"lstm_v1_large_results.csv")

results_lstm_df.to_csv(path_results, index=False)



# ### 4.4.2 LSTM LamaH-CE

# ###### 💍 rule them all V2

# In[ ]:


# user input
epochs = 50 # 30 +

# model parametres
input_size = 3 # number of variables (meteorological, not including qobs) ### X_train_lstm_lamah.shape[2]
hidden_size = 10
output_size = 1
hidden_layers = 1
start_lr = 0.01
dropout_prob = 0.3
batch_size = 64
loss_func = "MSE"

# representative catchments V2 (LamaH-CE)
IDs_red_v2 = [334, 743, 439, 24, 330, 75, 383]


# In[ ]:


# check IDs
IDs_red


# In[ ]:


results_lstm_dfs = []

# loop that rules it all
for id_num in IDs_red:
    print("ID:", id_num)

    # load data
    # load data for the full time period so start train to stop test, it is split later!
    df_lstm_lamah = load_data_new(path_vars_lamah_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_lamah_lstm_small, start_date=start_train, stop_date=stop_test, dataset='lamah', delimiter=';', qobs_mm='True')

    # prepare lstm data with function from script
    X_train_lstm_lamah, X_val_lstm_lamah, X_test_lstm_lamah, y_train_lstm_lamah, y_val_lstm_lamah, y_test_lstm_lamah, scaler_lamah = prep_data_lstm_new(df_lstm_lamah, start_train, stop_train, start_val, stop_val, start_test, stop_test, tensors=True)

    # push tensors to GPU if available
    X_train_lstm_lamah = X_train_lstm_lamah.to(device)
    X_val_lstm_lamah = X_val_lstm_lamah.to(device)
    X_test_lstm_lamah = X_test_lstm_lamah.to(device)
    y_train_lstm_lamah = y_train_lstm_lamah.to(device)
    y_val_lstm_lamah = y_val_lstm_lamah.to(device)
    y_test_lstm_lamah = y_test_lstm_lamah.to(device)


    # initialize
    model = LSTMModel(input_size, hidden_size, output_size, dropout_prob=dropout_prob)

    # push to GPU
    model.to(device)

    # loss function
    if loss_func == "MSE":
        criterion = nn.MSELoss()
        print("Using MSE loss func")
    elif loss_func == "RMSE":
        criterion = RMSLELoss()
        print("Using RMSE loss func")
    elif loss_func == "KGE":
        criterion = KGELossMOD()
        print("Using KGE loss func")
    else:
        print("no valid loss func defined")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)

    #### define lr scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # train
    model.train()

    best_val_loss = 50000000 # just a high number

    for epoch in range(epochs):
        for i in range(0, len(X_train_lstm_lamah), batch_size):
            optimizer.zero_grad()
            batch_X = torch.Tensor(X_train_lstm_lamah[i:i+batch_size]).to(device)
            batch_y = torch.Tensor(y_train_lstm_lamah[i:i+batch_size]).to(device)
            y_pred_train = model(batch_X)
            loss = criterion(y_pred_train, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step(loss) ## update the scheduler, is this the right location?

        # also calculate validation loss
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for validation
            y_pred_val = model(torch.Tensor(X_val_lstm_lamah).to(device))
            loss_val = criterion(y_pred_val, torch.Tensor(y_val_lstm_lamah).to(device))
        model.train()  # Set the model back to training mode

        # make sure to save the best validation loss as state_dict

        # Check if the current validation loss is the best so far
        if loss_val < best_val_loss:
            print("Accessed loop")
            best_val_loss = loss_val
            # Save the model's state dictionary to a file
            best_model_state_lstm_v2 = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}, loss train: {loss.item()}, loss val: {loss_val.item()}')

        # save trained model parameters
    if environment=='local': ### fix if using local
        torch.save(best_model_state_hybrid_v2, f'models/hybrid_v2_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{start_lr}.pth')
    if environment=='colab':
        torch.save(best_model_state_lstm_v2, f'/content/drive/MyDrive/msc_thesis/python/notebooks/colab/LSTMV2/trained_parameters/lstm_v2_id_{id_num}_loss_{loss_func}_lr_{start_lr}.pth')

    # load model here with best parameters?
    load_model = torch.load(f'/content/drive/MyDrive/msc_thesis/python/notebooks/colab/LSTMV2/trained_parameters/lstm_v2_id_{id_num}_loss_{loss_func}_lr_{start_lr}.pth')
    best_model = LSTMModel(input_size, hidden_size, output_size, dropout_prob=dropout_prob)
    best_model.load_state_dict(load_model)


    # evaluate model
    best_model.eval()


    # prediction on training data
    with torch.no_grad():
        y_pred_train_lstm_lamah = best_model(torch.Tensor(X_train_lstm_lamah)) ###

    # prediction on validation data
    with torch.no_grad():
        y_pred_val_lstm_lamah = best_model(torch.Tensor(X_val_lstm_lamah)) ###

    # prediction on testing data
    with torch.no_grad():
        y_pred_test_lstm_lamah = best_model(torch.Tensor(X_test_lstm_lamah)) ###

    # rescaling # NOTE: this is a bit of a workaround as the data is scaled before it is plit into X and y, so the scaler expects 4 input tensors could also be fixed by changing the script
    # rescale obesrvations
    train_meteo_obs = torch.cat((X_train_lstm_lamah[:,0,:], y_train_lstm_lamah), dim=1) # concat X and y train (so meteo + obs) ### Taking the first of the 365 dimensions, is that okay???
    val_meteo_obs = torch.cat((X_val_lstm_lamah[:,0,:], y_val_lstm_lamah), dim=1)
    test_meteo_obs = torch.cat((X_test_lstm_lamah[:,0,:], y_test_lstm_lamah), dim=1) # concat X and y test (so meteo + obs)

    ### added .cpu()
    train_meteo_obs_rs = scaler_lamah.inverse_transform(train_meteo_obs.cpu()) # rescale training
    val_meteo_obs_rs = scaler_lamah.inverse_transform(val_meteo_obs.cpu())
    test_meteo_obs_rs = scaler_lamah.inverse_transform(test_meteo_obs.cpu()) # rescale testing
    ### train_meteo_obs_rs = scaler_lamah.inverse_transform(train_meteo_obs) # rescale training
    ### val_meteo_obs_rs = scaler_lamah.inverse_transform(val_meteo_obs)
    ### test_meteo_obs_rs = scaler_lamah.inverse_transform(test_meteo_obs) # rescale testing
    train_obs_rs = train_meteo_obs_rs[:,-1] # get the rescaled training observations, complicated I know
    val_obs_rs = val_meteo_obs_rs[:,-1]
    test_obs_rs = test_meteo_obs_rs[:,-1]

    # make observaitons tensors (they are not tensors because of the minmax scaling from scipy)
    train_obs_rs_tensor = torch.tensor(train_obs_rs)
    val_obs_rs_tensor = torch.tensor(val_obs_rs)
    test_obs_rs_tensor = torch.tensor(test_obs_rs)

    # rescale prediction
    train_meteo_pred = torch.cat((X_train_lstm_lamah[:,0,:], y_pred_train_lstm_lamah), dim=1) # concat X train and prediction for y
    val_meteo_pred = torch.cat((X_val_lstm_lamah[:,0,:], y_pred_val_lstm_lamah), dim=1)
    test_meteo_pred = torch.cat((X_test_lstm_lamah[:,0,:], y_pred_test_lstm_lamah), dim=1) # concat X test and prediction for y

    ### added .cpu()
    train_meteo_pred_rs = scaler_lamah.inverse_transform(train_meteo_pred.cpu()) # rescale training
    val_meteo_pred_rs = scaler_lamah.inverse_transform(val_meteo_pred.cpu())
    test_meteo_pred_rs = scaler_lamah.inverse_transform(test_meteo_pred.cpu()) # rescale testing
    ### train_meteo_pred_rs = scaler_lamah.inverse_transform(train_meteo_pred) # rescale training
    ### val_meteo_pred_rs = scaler_lamah.inverse_transform(val_meteo_pred)
    ### test_meteo_pred_rs = scaler_lamah.inverse_transform(test_meteo_pred) # rescale testing
    train_pred_rs = train_meteo_pred_rs[:,-1] # get the rescaled training observations, complicated I know
    val_pred_rs = val_meteo_pred_rs[:,-1]
    test_pred_rs = test_meteo_pred_rs[:,-1]

    # make predictions tensors, also not super efficient but does not matter for now
    train_pred_rs_tensor = torch.tensor(train_pred_rs)
    val_pred_rs_tensor = torch.tensor(val_pred_rs)
    test_pred_rs_tensor = torch.tensor(test_pred_rs)

    # save training, validation and testing
    if environment == "colab":
        path_euler_training = os.path.join(path_lstm_v2_colab, "discharge/training/euler/", f"qsim_lstm_training_euler_ID_{id_num}_loss_func_{loss_func}.csv")
        path_euler_validation = os.path.join(path_lstm_v2_colab, "discharge/validation/euler/", f"qsim_lstm_validation_euler_ID_{id_num}_loss_func_{loss_func}.csv")
        path_euler_testing = os.path.join(path_lstm_v2_colab, "discharge/testing/euler/", f"qsim_lstm_testing_euler_ID_{id_num}_loss_func_{loss_func}.csv")

    if environment == "local":
        ### add if needed
        """
        path_euler_cal = os.path.join(path_hybrid_local, "discharge/cal/euler/", f"qsim_cal_euler_ID_{id_num}.csv")
        path_euler_test = os.path.join(path_hybrid_local, "discharge/test/euler/", f"qsim_test_euler_ID_{id_num}.csv")
        """

    # training, validation, testing
    np.savetxt(path_euler_training, train_pred_rs, delimiter=",")
    np.savetxt(path_euler_validation, val_pred_rs, delimiter=",")
    np.savetxt(path_euler_testing, test_pred_rs, delimiter=",")

    # save the observation timeseries
    if environment == "colab":
        path_qobs_mm_training = os.path.join(path_lstm_v2_colab, "discharge/training/observations/", f"qobs_mm_train_ID_{id_num}.csv")
        path_qobs_mm_validation = os.path.join(path_lstm_v2_colab, "discharge/validation/observations/", f"qobs_mm_train_ID_{id_num}.csv")
        path_qobs_mm_testing = os.path.join(path_lstm_v2_colab, "discharge/testing/observations/", f"qobs_mm_train_ID_{id_num}.csv")

    if environment == "local":
        ### add if needed
        """
        path_qobs_mm_cal = os.path.join(path_conceptual_v2_local, "discharge/cal/observations/", f"qobs_mm_cal_ID_{id_num}.csv")
        path_qobs_mm_test = os.path.join(path_conceptual_v2_local, "discharge/test/observations/", f"qobs_mm_test_ID_{id_num}.csv")
        """

    # training, validation, testing
    np.savetxt(path_qobs_mm_training, train_obs_rs, delimiter=",") # [:-1]
    np.savetxt(path_qobs_mm_validation, val_obs_rs, delimiter=",")
    np.savetxt(path_qobs_mm_testing, test_obs_rs, delimiter=",")


    # calculate evaluation metrics BUT leave 365 days for warm-up
    nse_loss = NSELoss()
    mse_loss = MSELoss()
    kge_loss = KGELoss()

    # loss training
    nse_loss_training = nse_loss(train_pred_rs_tensor[365:], train_obs_rs_tensor[365:])
    mse_loss_training = mse_loss(train_pred_rs_tensor[365:], train_obs_rs_tensor[365:])
    kge_loss_training = kge_loss(train_pred_rs_tensor[365:], train_obs_rs_tensor[365:])

    # loss validation
    nse_loss_validation = nse_loss(val_pred_rs_tensor[365:], val_obs_rs_tensor[365:])
    mse_loss_validation = mse_loss(val_pred_rs_tensor[365:], val_obs_rs_tensor[365:])
    kge_loss_validation = kge_loss(val_pred_rs_tensor[365:], val_obs_rs_tensor[365:])
    # loss test
    nse_loss_testing = nse_loss(test_pred_rs_tensor[365:], test_obs_rs_tensor[365:])
    mse_loss_testing = mse_loss(test_pred_rs_tensor[365:], test_obs_rs_tensor[365:])
    kge_loss_testing = kge_loss(test_pred_rs_tensor[365:], test_obs_rs_tensor[365:])

    # save evaluation metrics (ideally to one large file )
    row = {
        "id_num": id_num,
        "start training": start_train,
        "stop training": stop_train,
        "start test": start_test,
        "stop test": stop_test,
        "nse_loss_train": nse_loss_training,
        "mse_loss_train": mse_loss_training,
        "kge_loss_train": kge_loss_training,
        "nse_loss_val": nse_loss_validation,
        "mse_loss_val": mse_loss_validation,
        "kge_loss_val": kge_loss_validation,
        "nse_loss_test": nse_loss_testing,
        "mse_loss_test": mse_loss_testing,
        "kge_loss_test": kge_loss_testing
    }

    # Create a DataFrame for the current iteration
    iteration_df = pd.DataFrame([row])

    # Append the iteration DataFrame to the list
    results_lstm_dfs.append(iteration_df)

# Concatenate all DataFrames into one
results_lstm_df = pd.concat(results_lstm_dfs, ignore_index=True)


# Save the results DataFrame to a CSV file
if environment == "colab":
    path_results = os.path.join(path_lstm_v2_colab, "evaluation_metrics/", f"lstm_v2_results.csv")

if environment == "local":
    path_results = os.path.join(path_lstm_v2_local, "evaluation_metrics/", f"lstm_v2_results.csv")

results_lstm_df.to_csv(path_results, index=False)



# #### 4.4.2.1 LSTM Large LamaH-CE

# In[ ]:


# user input
epochs = 50 # 30 +

# model parametres
input_size = 6 # number of variables (meteorological, not including qobs) ### X_train_lstm_lamah.shape[2]
hidden_size = 10
output_size = 1
hidden_layers = 1
start_lr = 0.01
dropout_prob = 0.3
batch_size = 64
loss_func = "MSE"

# representative catchments V2 (LamaH-CE)
IDs_red_v2 = [334, 743, 439, 24, 330, 75, 383]


# In[ ]:


# check IDs
IDs_red


# In[ ]:


results_lstm_dfs = []

# loop that rules it all
for id_num in IDs_red:
    print("ID:", id_num)

    # load data
    # load data for the full time period so start train to stop test, it is split later!
    df_lstm_lamah = load_data_new(path_vars_lamah_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID=id_num, variables=variables_lamah_lstm_large, start_date=start_train, stop_date=stop_test, dataset='lamah', delimiter=';', qobs_mm='True')

    # prepare lstm data with function from script
    X_train_lstm_lamah, X_val_lstm_lamah, X_test_lstm_lamah, y_train_lstm_lamah, y_val_lstm_lamah, y_test_lstm_lamah, scaler_lamah = prep_data_lstm_new(df_lstm_lamah, start_train, stop_train, start_val, stop_val, start_test, stop_test, tensors=True)

    # push tensors to GPU if available
    X_train_lstm_lamah = X_train_lstm_lamah.to(device)
    X_val_lstm_lamah = X_val_lstm_lamah.to(device)
    X_test_lstm_lamah = X_test_lstm_lamah.to(device)
    y_train_lstm_lamah = y_train_lstm_lamah.to(device)
    y_val_lstm_lamah = y_val_lstm_lamah.to(device)
    y_test_lstm_lamah = y_test_lstm_lamah.to(device)


    # initialize
    model = LSTMModel(input_size, hidden_size, output_size, dropout_prob=dropout_prob)

    # push to GPU
    model.to(device)

    # loss function
    if loss_func == "MSE":
        criterion = nn.MSELoss()
        print("Using MSE loss func")
    elif loss_func == "RMSE":
        criterion = RMSLELoss()
        print("Using RMSE loss func")
    elif loss_func == "KGE":
        criterion = KGELossMOD()
        print("Using KGE loss func")
    else:
        print("no valid loss func defined")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)

    #### define lr scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # train
    model.train()

    best_val_loss = 50000000 # just a high number

    for epoch in range(epochs):
        for i in range(0, len(X_train_lstm_lamah), batch_size):
            optimizer.zero_grad()
            batch_X = torch.Tensor(X_train_lstm_lamah[i:i+batch_size]).to(device)
            batch_y = torch.Tensor(y_train_lstm_lamah[i:i+batch_size]).to(device)
            y_pred_train = model(batch_X)
            loss = criterion(y_pred_train, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step(loss) ## update the scheduler, is this the right location?

        # also calculate validation loss
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for validation
            y_pred_val = model(torch.Tensor(X_val_lstm_lamah).to(device))
            loss_val = criterion(y_pred_val, torch.Tensor(y_val_lstm_lamah).to(device))
        model.train()  # Set the model back to training mode

        # make sure to save the best validation loss as state_dict

        # Check if the current validation loss is the best so far
        if loss_val < best_val_loss:
            print("Accessed loop")
            best_val_loss = loss_val
            # Save the model's state dictionary to a file
            best_model_state_lstm_v2 = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}, loss train: {loss.item()}, loss val: {loss_val.item()}')

        # save trained model parameters
    if environment=='local': ### fix if using local
        torch.save(best_model_state_hybrid_v2, f'models/hybrid_v2_id_{id_num}_solver_{solver}_loss_{loss_func}_lr_{start_lr}.pth')
    if environment=='colab':
        torch.save(best_model_state_lstm_v2, f'/content/drive/MyDrive/msc_thesis/python/notebooks/colab/LSTMV2Large/trained_parameters/lstm_v2_id_{id_num}_loss_{loss_func}_lr_{start_lr}.pth')

    # load model here with best parameters?
    load_model = torch.load(f'/content/drive/MyDrive/msc_thesis/python/notebooks/colab/LSTMV2Large/trained_parameters/lstm_v2_id_{id_num}_loss_{loss_func}_lr_{start_lr}.pth')
    best_model = LSTMModel(input_size, hidden_size, output_size, dropout_prob=dropout_prob)
    best_model.load_state_dict(load_model)


    # evaluate model
    best_model.eval()


    # prediction on training data
    with torch.no_grad():
        y_pred_train_lstm_lamah = best_model(torch.Tensor(X_train_lstm_lamah)) ###

    # prediction on validation data
    with torch.no_grad():
        y_pred_val_lstm_lamah = best_model(torch.Tensor(X_val_lstm_lamah)) ###

    # prediction on testing data
    with torch.no_grad():
        y_pred_test_lstm_lamah = best_model(torch.Tensor(X_test_lstm_lamah)) ###

    # rescaling # NOTE: this is a bit of a workaround as the data is scaled before it is plit into X and y, so the scaler expects 4 input tensors could also be fixed by changing the script
    # rescale obesrvations
    train_meteo_obs = torch.cat((X_train_lstm_lamah[:,0,:], y_train_lstm_lamah), dim=1) # concat X and y train (so meteo + obs) ### Taking the first of the 365 dimensions, is that okay???
    val_meteo_obs = torch.cat((X_val_lstm_lamah[:,0,:], y_val_lstm_lamah), dim=1)
    test_meteo_obs = torch.cat((X_test_lstm_lamah[:,0,:], y_test_lstm_lamah), dim=1) # concat X and y test (so meteo + obs)

    ### added .cpu()
    train_meteo_obs_rs = scaler_lamah.inverse_transform(train_meteo_obs.cpu()) # rescale training
    val_meteo_obs_rs = scaler_lamah.inverse_transform(val_meteo_obs.cpu())
    test_meteo_obs_rs = scaler_lamah.inverse_transform(test_meteo_obs.cpu()) # rescale testing
    ### train_meteo_obs_rs = scaler_lamah.inverse_transform(train_meteo_obs) # rescale training
    ### val_meteo_obs_rs = scaler_lamah.inverse_transform(val_meteo_obs)
    ### test_meteo_obs_rs = scaler_lamah.inverse_transform(test_meteo_obs) # rescale testing
    train_obs_rs = train_meteo_obs_rs[:,-1] # get the rescaled training observations, complicated I know
    val_obs_rs = val_meteo_obs_rs[:,-1]
    test_obs_rs = test_meteo_obs_rs[:,-1]

    # make observaitons tensors (they are not tensors because of the minmax scaling from scipy)
    train_obs_rs_tensor = torch.tensor(train_obs_rs)
    val_obs_rs_tensor = torch.tensor(val_obs_rs)
    test_obs_rs_tensor = torch.tensor(test_obs_rs)

    # rescale prediction
    train_meteo_pred = torch.cat((X_train_lstm_lamah[:,0,:], y_pred_train_lstm_lamah), dim=1) # concat X train and prediction for y
    val_meteo_pred = torch.cat((X_val_lstm_lamah[:,0,:], y_pred_val_lstm_lamah), dim=1)
    test_meteo_pred = torch.cat((X_test_lstm_lamah[:,0,:], y_pred_test_lstm_lamah), dim=1) # concat X test and prediction for y

    ### added .cpu()
    train_meteo_pred_rs = scaler_lamah.inverse_transform(train_meteo_pred.cpu()) # rescale training
    val_meteo_pred_rs = scaler_lamah.inverse_transform(val_meteo_pred.cpu())
    test_meteo_pred_rs = scaler_lamah.inverse_transform(test_meteo_pred.cpu()) # rescale testing
    ### train_meteo_pred_rs = scaler_lamah.inverse_transform(train_meteo_pred) # rescale training
    ### val_meteo_pred_rs = scaler_lamah.inverse_transform(val_meteo_pred)
    ### test_meteo_pred_rs = scaler_lamah.inverse_transform(test_meteo_pred) # rescale testing
    train_pred_rs = train_meteo_pred_rs[:,-1] # get the rescaled training observations, complicated I know
    val_pred_rs = val_meteo_pred_rs[:,-1]
    test_pred_rs = test_meteo_pred_rs[:,-1]

    # make predictions tensors, also not super efficient but does not matter for now
    train_pred_rs_tensor = torch.tensor(train_pred_rs)
    val_pred_rs_tensor = torch.tensor(val_pred_rs)
    test_pred_rs_tensor = torch.tensor(test_pred_rs)

    # save training, validation and testing
    if environment == "colab":
        path_euler_training = os.path.join(path_lstm_v2_large_colab, "discharge/training/euler/", f"qsim_lstm_training_euler_ID_{id_num}_loss_func_{loss_func}.csv")
        path_euler_validation = os.path.join(path_lstm_v2_large_colab, "discharge/validation/euler/", f"qsim_lstm_validation_euler_ID_{id_num}_loss_func_{loss_func}.csv")
        path_euler_testing = os.path.join(path_lstm_v2_large_colab, "discharge/testing/euler/", f"qsim_lstm_testing_euler_ID_{id_num}_loss_func_{loss_func}.csv")

    if environment == "local":
        ### add if needed
        """
        path_euler_cal = os.path.join(path_hybrid_local, "discharge/cal/euler/", f"qsim_cal_euler_ID_{id_num}.csv")
        path_euler_test = os.path.join(path_hybrid_local, "discharge/test/euler/", f"qsim_test_euler_ID_{id_num}.csv")
        """

    # training, validation, testing
    np.savetxt(path_euler_training, train_pred_rs, delimiter=",")
    np.savetxt(path_euler_validation, val_pred_rs, delimiter=",")
    np.savetxt(path_euler_testing, test_pred_rs, delimiter=",")

    # save the observation timeseries
    if environment == "colab":
        path_qobs_mm_training = os.path.join(path_lstm_v2_large_colab, "discharge/training/observations/", f"qobs_mm_train_ID_{id_num}.csv")
        path_qobs_mm_validation = os.path.join(path_lstm_v2_large_colab, "discharge/validation/observations/", f"qobs_mm_train_ID_{id_num}.csv")
        path_qobs_mm_testing = os.path.join(path_lstm_v2_large_colab, "discharge/testing/observations/", f"qobs_mm_train_ID_{id_num}.csv")

    if environment == "local":
        ### add if needed
        """
        path_qobs_mm_cal = os.path.join(path_conceptual_v2_local, "discharge/cal/observations/", f"qobs_mm_cal_ID_{id_num}.csv")
        path_qobs_mm_test = os.path.join(path_conceptual_v2_local, "discharge/test/observations/", f"qobs_mm_test_ID_{id_num}.csv")
        """

    # training, validation, testing
    np.savetxt(path_qobs_mm_training, train_obs_rs, delimiter=",") # [:-1]
    np.savetxt(path_qobs_mm_validation, val_obs_rs, delimiter=",")
    np.savetxt(path_qobs_mm_testing, test_obs_rs, delimiter=",")


    # calculate evaluation metrics BUT leave 365 days for warm-up
    nse_loss = NSELoss()
    mse_loss = MSELoss()
    kge_loss = KGELoss()

    # loss training
    nse_loss_training = nse_loss(train_pred_rs_tensor[365:], train_obs_rs_tensor[365:])
    mse_loss_training = mse_loss(train_pred_rs_tensor[365:], train_obs_rs_tensor[365:])
    kge_loss_training = kge_loss(train_pred_rs_tensor[365:], train_obs_rs_tensor[365:])

    # loss validation
    nse_loss_validation = nse_loss(val_pred_rs_tensor[365:], val_obs_rs_tensor[365:])
    mse_loss_validation = mse_loss(val_pred_rs_tensor[365:], val_obs_rs_tensor[365:])
    kge_loss_validation = kge_loss(val_pred_rs_tensor[365:], val_obs_rs_tensor[365:])
    # loss test
    nse_loss_testing = nse_loss(test_pred_rs_tensor[365:], test_obs_rs_tensor[365:])
    mse_loss_testing = mse_loss(test_pred_rs_tensor[365:], test_obs_rs_tensor[365:])
    kge_loss_testing = kge_loss(test_pred_rs_tensor[365:], test_obs_rs_tensor[365:])

    # save evaluation metrics (ideally to one large file )
    row = {
        "id_num": id_num,
        "start training": start_train,
        "stop training": stop_train,
        "start test": start_test,
        "stop test": stop_test,
        "nse_loss_train": nse_loss_training,
        "mse_loss_train": mse_loss_training,
        "kge_loss_train": kge_loss_training,
        "nse_loss_val": nse_loss_validation,
        "mse_loss_val": mse_loss_validation,
        "kge_loss_val": kge_loss_validation,
        "nse_loss_test": nse_loss_testing,
        "mse_loss_test": mse_loss_testing,
        "kge_loss_test": kge_loss_testing
    }

    # Create a DataFrame for the current iteration
    iteration_df = pd.DataFrame([row])

    # Append the iteration DataFrame to the list
    results_lstm_dfs.append(iteration_df)

# Concatenate all DataFrames into one
results_lstm_df = pd.concat(results_lstm_dfs, ignore_index=True)


# Save the results DataFrame to a CSV file
if environment == "colab":
    path_results = os.path.join(path_lstm_v2_large_colab, "evaluation_metrics/", f"lstm_v2_large_results.csv")

if environment == "local":
    path_results = os.path.join(path_lstm_v2_large_local, "evaluation_metrics/", f"lstm_v2_large_results.csv")

results_lstm_df.to_csv(path_results, index=False)



# In[ ]:




