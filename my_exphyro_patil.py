#!/usr/bin/env python
# coding: utf-8

# ## NOTE:
# - **All of this can now be done in the "models_torchdiffeq.ipynb" notebook, including modified versions of EXP-Hydro. However, calibration takes much much longer so we might use this version for calibration.**

# # Overview
# This notebook will implement exphydro staying as close as possible to the original from Patil.
# 
# Instead of importing the scripts "ExphydroModel.py" and "ExphydroParameters.py", they will be copied into the notebook and changes will be made to them.
# 
# Potential changes:
# - Dataset -> Will be for LamaH
# - Input variables -> May add / remove input variables
# - Calibration approach may be changed
# - ODE solver (from hydroutils) may be replaced with another ODE solver
# 
# 
# Overview of the original code, only for the lumped model:
# Scripts with classes:  
# - ExphydroModel.py
# - ExphydroParamters.py
# 
# Scripts to run:
# 1. "Run_exphydro_lumped_singlerun.py" -> Parameters are defined by hand (this is just to check that the code runs)
# 2. "Run_exphydro_lumped_mc.py" -> monte carlo (slow)
# 3. "Run_exphydro_lumped_pso.py" -> particle swarm optimization (fast)

# # 0. Imports

# In[ ]:


# imports
import os
import sys
import torch
import datetime as dt


# In[ ]:


# lamah
path_vars_lamah_local = #
path_qobs_lamah_local = #

path_vars_lamah_colab = #
path_qobs_lamah_colab = 

# eobs (only meteorological forcings, qobs is form lamah dataset)
path_vars_eobs_local = #
path_vars_eobs_colab = #

path_catchment_attrs_colab = #
path_calibrated_parameters_v1 = #
path_calibrated_parameters_v2 = #

# user input
"""
Set environment to either "colab" or "local"
"""
environment = "colab"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # "cuda:0" instead of "cuda"???
print(device)

# specify which variables to load into the df
# do not include 'qobs'
variables_lamah = ['tmax', 'tmean', 'tmin', 'tdpmax', 'tdpmean', 'tdpmin', 'windu', 'windv', 'albedo', 'swe', 'sradmean', 'tradmean', 'et', 'prcp']
variables_eobs = ['prcp', 'tmean', 'tmin', 'tmax', 'seapress', 'humidity', 'windspeed', 'srad', 'albedo', 'pet', 'daylength', 'pev'] # also has "DOY"

### added for hydrological year shift
reference = "1981-01-01" # not needed anymore, right?

start_cal = "1981-10-01"
stop_cal = "2001-10-01" 
start_train = "1981-10-01" 
stop_train = "2001-10-01"
start_val = "2001-10-01"
stop_val = "2005-10-01"
start_test = "2005-10-01"
stop_test = "2017-10-01"


# In[ ]:


### for now we compute the length of the calibration period (equal to training period) like this
start_cal_date = dt.datetime.strptime(start_cal, "%Y-%m-%d")
stop_cal_date = dt.datetime.strptime(stop_cal, "%Y-%m-%d")
ref_date = dt.datetime.strptime(reference, "%Y-%m-%d")

# Calculate the number of days between the two dates
cal_period = (stop_cal_date - start_cal_date).days

print(f"Calibrate from {start_cal} to {stop_cal} which is equal to {cal_period} days.")


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


# clone exp-hydro from Github
get_ipython().system('git clone https://github.com/sopanpatil/exp-hydro.git')


# In[ ]:


# run the setup.py file
get_ipython().system('ls')
get_ipython().run_line_magic('cd', 'exp-hydro')
get_ipython().system('python3 setup.py install')


# In[ ]:


# this one should be correct
get_ipython().system('pip install git+https://github.com/sopanpatil/hydroutils.git')


# In[ ]:


# import hydroutils

"""
May have to restart the runtime for this cell to work.
"""

import hydroutils


# In[ ]:


import torch.nn as nn
import torch.optim as optim
import torchdiffeq
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import numpy

from hydroutils import OdeSolver
from hydroutils import Parameter
from hydroutils import ObjectiveFunction
from hydroutils import Calibration


# In[ ]:



# import from scripts
from data_processing_new import *
from loss_functions import *


# # 1. Functions and Classes

# ## 1.1 ExphydroModel (original) used for E-OBS

# In[ ]:


# "ExphydroModel.py"


# In[ ]:


# original Patil code
#!/usr/bin/env python

# Programmer(s): Sopan Patil.
# This file is part of the 'exphydro.lumped' package.

import numpy
from hydroutils import OdeSolver


class ExphydroModel(object):

    """ An EXP-HYDRO bucket has its own climate inputs.

    It also has the properties of storage (both soil
    and snow), stream discharge (qsim), snowmelt (melt) and
    evapotranspiration (et)
    """

    def __init__(self, p, pet, t):

        """ This method is used to initialise, i.e., create an instance of the ExphydroModel class.

        Syntax: ExphydroModel(p, pet, t)

        Args:
            (1) p: Daily precipitation time-series (mm/day)
            (2) pet: Daily potential evapotranspiration time-series (mm/day)
            (3) t: Daily mean air temperature time-series (deg C)

        """

        # Below are the climate inputs
        self.P = p  # Daily precipitation (mm/day)
        self.PET = pet  # Daily PET (mm/day)
        self.T = t  # Daily mean air temperature (deg C)

        self.timespan = self.P.shape[0]  # Time length of the simulation period

        # Below are the state and flux variables of EXP-HYDRO
        #  All of them are initialised to zero
        self.storage = numpy.zeros(2)  # Storage of soil and snow buckets (mm)
        self.qsim = numpy.zeros(self.timespan)  # Simulated streamflow (mm/day)
        self.et = numpy.zeros(self.timespan)  # Simulated ET (mm/day)
        self.melt = numpy.zeros(self.timespan)  # Simulated snowmelt (mm/day)

    # ----------------------------------------------------------------

    def waterbalance(self, t, s, para):

        """ This method provides the right hand side of the dS/dt equations."""

        # EXP-HYDRO parameter values from object para
        f = para.f.value
        ddf = para.ddf.value
        smax = para.smax.value
        qmax = para.qmax.value
        mint = para.mint.value
        maxt = para.maxt.value

        # The line below ensures that the time step of input and output variables is always an integer.
        # ODE solvers can take fractional time steps, for which input data does not exist.
        tt = int(min(round(t), self.timespan-1))

        # NOTE: The min condition in above line is very important and is needed when the ODE solver
        # jumps to a time-step that is beyond the time-series length.

        # Loading the input data for current time step
        p = self.P[tt]
        te = self.T[tt]
        pet = self.PET[tt]

        # Partitioning precipitation into rain and snow
        [ps, pr] = self.rainsnowpartition(p, te, mint)

        # Snow bucket
        m = self.snowbucket(s[0], te, ddf, maxt)

        # Soil bucket
        [et, qsub, qsurf] = self.soilbucket(s[1], pet, f, smax, qmax)

        # Water balance equations
        ds1 = ps - m
        ds2 = pr + m - et - qsub - qsurf

        ds = numpy.array([ds1, ds2])

        # Writing the flux calculations into output variables for the
        # current time step
        self.qsim[tt] = qsub + qsurf
        self.et[tt] = et
        self.melt[tt] = m

        return ds

    # ----------------------------------------------------------------

    @staticmethod
    def rainsnowpartition(p, t, mint):

        """ EXP-HYDRO equations to partition incoming precipitation
        into rain or snow."""

        if t < mint:
            psnow = p
            prain = 0
        else:
            psnow = 0
            prain = p

        return [psnow, prain]

    # ----------------------------------------------------------------

    @staticmethod
    def snowbucket(s, t, ddf, maxt):

        """ EXP-HYDRO equations for the snow bucket."""

        if t > maxt:
            if s > 0:
                melt = min(s, ddf*(t - maxt))
            else:
                melt = 0
        else:
            melt = 0

        return melt

    # ----------------------------------------------------------------

    @staticmethod
    def soilbucket(s, pet, f, smax, qmax):

        """ EXP-HYDRO equations for the soil bucket."""

        if s < 0:
            et = 0
            qsub = 0
            qsurf = 0
        elif s > smax:
            et = pet
            qsub = qmax
            qsurf = s - smax
        else:
            qsub = qmax * numpy.exp(-f * (smax - s))
            qsurf = 0
            et = pet * (s / smax)

        return [et, qsub, qsurf]

    # ----------------------------------------------------------------

    def simulate(self, para):

        """ This method performs the integration of dS/dt equations
        over the entire simulation time period
        """

        # Solving the ODE. To check which ODE solvers are available to use,
        # please check OdeSolver.py in hydroutils package
        OdeSolver.solve_rk4(self.waterbalance, self.storage, para, tlength=self.timespan)
        return self.qsim


# ## 1.1 a) ExpyhydroModel (modified) used for LamaH 🦙
# 
# This is modified so that it takes ET as input directly instead of PET.

# In[ ]:


# compact version, see documentation file for the changes to the model
class ExphydroModelMod(object):

    """ An EXP-HYDRO bucket has its own climate inputs.

    It also has the properties of storage (both soil
    and snow), stream discharge (qsim), snowmelt (melt) and
    evapotranspiration (et)
    """

    def __init__(self, p, et, t):

        """ This method is used to initialise, i.e., create an instance of the ExphydroModel class.

        Syntax: ExphydroModel(p, et, t)

        Args:
            (1) p: Daily precipitation time-series (mm/day)
            (2) et: Daily potential evapotranspiration time-series (mm/day)
            (3) t: Daily mean air temperature time-series (deg C)

        """

        # Below are the climate inputs
        self.P = p  # Daily precipitation (mm/day)
        self.ET = et
        self.T = t  # Daily mean air temperature (deg C)


        ############### change this at some point so that we can use trian, val, test data
        self.timespan = self.P.shape[0]  # Time length of the simulation period

        self.storage = numpy.zeros(2)  # Storage of soil and snow buckets (mm)
        self.qsim = numpy.zeros(self.timespan)  # Simulated streamflow (mm/day)
        self.melt = numpy.zeros(self.timespan)  # Simulated snowmelt (mm/day)

    # ----------------------------------------------------------------

    def waterbalance(self, t, s, para):

        """ This method provides the right hand side of the dS/dt equations."""

        # EXP-HYDRO parameter values from object para
        f = para.f.value
        ddf = para.ddf.value
        smax = para.smax.value
        qmax = para.qmax.value
        mint = para.mint.value
        maxt = para.maxt.value

        # The line below ensures that the time step of input and output variables is always an integer.
        # ODE solvers can take fractional time steps, for which input data does not exist.

        """
        This might be nice for the neural ode model...
        """
        tt = int(min(round(t), self.timespan-1))

        # NOTE: The min condition in above line is very important and is needed when the ODE solver
        # jumps to a time-step that is beyond the time-series length.

        # Loading the input data for current time step
        p = self.P[tt]
        te = self.T[tt]
        et = self.ET[tt]

        # Partitioning precipitation into rain and snow
        [ps, pr] = self.rainsnowpartition(p, te, mint)

        # Snow bucket
        m = self.snowbucket(s[0], te, ddf, maxt)
        [qsub, qsurf] = self.soilbucket(s[1], f, smax, qmax)

        # Water balance equations
        ds1 = ps - m
        ds2 = pr + m - et - qsub - qsurf

        ds = numpy.array([ds1, ds2])

        # Writing the flux calculations into output variables for the
        # current time step
        self.qsim[tt] = qsub + qsurf
        self.melt[tt] = m

        return ds

    # ----------------------------------------------------------------

    @staticmethod
    def rainsnowpartition(p, t, mint):

        """ EXP-HYDRO equations to partition incoming precipitation
        into rain or snow."""

        if t < mint:
            psnow = p
            prain = 0
        else:
            psnow = 0
            prain = p

        return [psnow, prain]

    # ----------------------------------------------------------------

    @staticmethod
    def snowbucket(s, t, ddf, maxt):

        """ EXP-HYDRO equations for the snow bucket."""

        if t > maxt:
            if s > 0:
                melt = min(s, ddf*(t - maxt))
            else:
                melt = 0
        else:
            melt = 0
        return melt

    # ----------------------------------------------------------------

    @staticmethod
    def soilbucket(s, f, smax, qmax):

        """ EXP-HYDRO equations for the soil bucket."""
        if s < 0:
            qsub = 0
            qsurf = 0
        elif s > smax:
            qsub = qmax
            qsurf = s - smax
        else:
            qsub = qmax * numpy.exp(-f * (smax - s))
            qsurf = 0

        return [qsub, qsurf]

    # ----------------------------------------------------------------

    def simulate(self, para):

        """ This method performs the integration of dS/dt equations
        over the entire simulation time period
        """

        # Solving the ODE. To check which ODE solvers are available to use,
        # please check OdeSolver.py in hydroutils package
        OdeSolver.solve_rk4(self.waterbalance, self.storage, para, tlength=self.timespan)
        return self.qsim


# ## 1.2 ExphydroParameters

# In[ ]:


# "ExphydroParameters.py"


# In[ ]:


#!/usr/bin/env python

# Programmer(s): Sopan Patil.
# This file is part of the 'exphydro.lumped' package.
class ExphydroParameters(object):

    def __init__(self):

        """ Each parameter set contains a random realisation of all six
        EXP-HYDRO parameters as well as default values of Nash-Sutcliffe
        and Kling-Gupta efficiencies
        """

        self.f = Parameter(0, 0.1)
        self.smax = Parameter(100.0, 1500.0)
        self.qmax = Parameter(10.0, 50.0)
        self.ddf = Parameter(0.0, 5.0)
        self.mint = Parameter(-3.0, 0.0)
        self.maxt = Parameter(0.0, 3.0)

        self.objval = -9999  # This is the objective function value

    # ----------------------------------------------------------------

    def assignvalues(self, f, smax, qmax, ddf, mint, maxt):

        """ This method is used to manually assign parameter values,
        which are given by the user as input arguments.
        """

        self.f.value = f
        self.smax.value = smax
        self.qmax.value = qmax
        self.ddf.value = ddf
        self.mint.value = mint
        self.maxt.value = maxt

    # ----------------------------------------------------------------

    def updateparameters(self, param1, param2, w):

        """ This method is used for PSO algorithm.
            Each parameter in the model has to do the following
            two things:
            (1) Update its velocity
            (2) Update its value
        """

        # Update parameter velocities
        self.f.updatevelocity(param1.f, param2.f, w)
        self.ddf.updatevelocity(param1.ddf, param2.ddf, w)
        self.smax.updatevelocity(param1.smax, param2.smax, w)
        self.qmax.updatevelocity(param1.qmax, param2.qmax, w)
        self.mint.updatevelocity(param1.mint, param2.mint, w)
        self.maxt.updatevelocity(param1.maxt, param2.maxt, w)

        # Update parameter values
        self.f.updatevalue()
        self.ddf.updatevalue()
        self.smax.updatevalue()
        self.qmax.updatevalue()
        self.mint.updatevalue()
        self.maxt.updatevalue()


# # 2. Data
# Load LamaH data.

# Training and testing periods in LamaH-CE:
# 
# Calibration: 1982 - 2000 (using 1981 as warm-up)
# Validation: 2000 - 2017

# Functions currently in script:
# 
# 
# *   load_data
# *   prep_data
# *   loss_functions
# 
# 

# In[ ]:


# load catchment attrs with all catchments that have gap-free runoff timeseries from 1981 to 2017
catchment_attrs_red = pd.read_csv(path_catchment_attrs_colab + "Catchment_attributes_nogaps_fullperiod.csv", delimiter=",")


# In[ ]:


catchment_attrs_red.shape


# In[ ]:


# TEMP
ID = 58


# In[ ]:


# load data
df = load_data_new(path_vars_lamah_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID, variables, start_cal, stop_cal,  qobs_mm=True)


# In[ ]:


df.tail(2)


# In[ ]:


# assign observations from LamaH
P = df['prcp'].values
### T = np.mean(df[['tmax', 'tmin']], axis=1) # calculate mean of tmax and tmin
T = df['tmean'].values
ET = df['et'].values
Qobs = df['qobs_mm'].values # this is important! EXP-HYDRO expects units mm not m3/d


# # 3. Model Runs
# 1. Single run with defined parameters
# 2. Monte Carlo
# 3. Particle Swarm Optimization

# In[ ]:


### is this even needed??
"""
#!/usr/bin/env python

# SET WORKING DIRECTORY

# Getting current directory, i.e., directory containing this file
dir1 = os.path.dirname(os.path.abspath('__file__'))

# Setting to current directory
os.chdir(dir1)
"""


# # Important to add the 365 days for Warm up period

# In[ ]:


# this is the whole calibration period but with a WARMUP period
calperiods_obs = [365, cal_period]
calperiods_sim = [365, cal_period]


# In[ ]:


calperiods_obs


# ## 3.1 EXP-Hydro Single Run Version (can go!)
# 
# ---
# 
# 

# In[ ]:


# "Run_exphydro_lumped_singlerun.py"

# Programmer(s): Sopan Patil.

""" MAIN PROGRAM FILE
Run this file to perform a single run of the EXP-HYDRO model
with user provided parameter values.
"""


# In[ ]:


# Initialise EXP-HYDRO model parameters object
params = ExphydroParameters()

# Specify the parameter values
# Please refer to Patil and Stieglitz (2014) for model parameter descriptions
f = 0.07
smax = 200
qmax = 20
ddf = 2
mint = -1
maxt = 1

# Assign the above parameter values into the model parameters object
params.assignvalues(f, smax, qmax, ddf, mint, maxt)

# Initialise the model by loading its climate inputs
model = ExphydroModelMod(P, ET, T)

### I guess this is a bit of a special case as we have no calibration in this case
simperiods_obs = [365, 1000]
simperiods_sim = [365, 1000]

# Run the model and calculate objective function value for the simulation period
Qsim = model.simulate(params)
kge = ObjectiveFunction.klinggupta(Qobs[simperiods_obs[0]:simperiods_obs[1]+1],
                                   Qsim[simperiods_sim[0]:simperiods_sim[1]+1])
print('KGE value = ', kge)


# In[ ]:





# In[ ]:


# Plot the observed and simulated hydrographs
plt.figure(figsize=(15,5))
plt.plot(Qobs[simperiods_obs[0]:simperiods_obs[1]+1], 'b-')
plt.plot(Qsim[simperiods_sim[0]:simperiods_sim[1]+1], 'r-')
plt.show()


# ## 3.2 EXP-Hydro Monte Carlo Version

# In[ ]:


# "Run_exphydro_lumped_mc.py"

# Programmer(s): Sopan Patil.

""" MAIN PROGRAM FILE
Run this file to optimise the EXP-HYDRO model parameters
using Monte Carlo optimisation algorithm.

Please note that this is a slow optimisation method compared to
Particle Swarm Optimisation (PSO), and requires a very large number
of iterations (> 10000) to get reliable optimisation.
"""


# In[ ]:


# Specify the no. of iterations
niter = 100

# Generate 'niter' initial EXP-HYDRO model parameters
params = [ExphydroParameters() for j in range(niter)]

# Initialise the model by loading its climate inputs
model = ExphydroModelMod(P, ET, T)

# Calibrate the model to identify optimal parameter set
paramsmax = Calibration.montecarlo_maximise(model, params, Qobs, ObjectiveFunction.klinggupta,
                                            calperiods_obs, calperiods_sim)
print('Calibration run KGE value = ', paramsmax.objval)

# Run the optimised model for validation period

### this takes all time steps beyond the calibration period

Qsim = model.simulate(paramsmax)
kge = ObjectiveFunction.klinggupta(Qobs[calperiods_obs[1]:], Qsim[calperiods_sim[1]:])
print('Independent run KGE value = ', kge)


# In[ ]:


# Plot the observed and simulated hydrographs
plt.figure(figsize=(15,5))
plt.plot(Qobs[calperiods_obs[0]-2000:-1], 'b-')
plt.plot(Qsim[calperiods_sim[0]-2000:-1], 'r-')
plt.show()


# In[ ]:





# ## 3.3 EXP-Hydro Particle Swarm Optimization Version
# 

# ### 3.3.1 EXP-HYDRO (original) E-OBS

# In[ ]:


# "Run_exphydro_lumped_pso.py"

# Programmer(s): Sopan Patil.

""" MAIN PROGRAM FILE
Run this file to optimise the model parameters of the spatially lumped
version of EXP-HYDRO model using Particle Swarm Optimisation (PSO) algorithm.
"""


# In[ ]:


# set paramters for calibration
# Specify the no. of parameter sets (particles) in a PSO swarm
npart = 20

method = "PSO" # this is just for saving the file


# In[ ]:


# now create a loop to do this for all catchments
for ID_num in catchment_attrs_red['ID']:

    # use this since code was interrupted
    if ID_num > 320: # so can jump back in if code is interrupted

        # ID_num is now the current ID

        # load dataframe
        df = load_data_new(path_vars_eobs_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID_num, variables_eobs, start_cal, stop_test, dataset="eobs", qobs_mm=True)

        # assign the input variables
        P = df['prcp'].values
        T = df['tmean'].values
        PET = df['pet'].values # importand change compared to V2
        Qobs = df['qobs_mm'].values # this is important! EXP-HYDRO expects units mm not m3/d

        print(f"STARTING CALIBRATION FOR CATCHMENT: {ID_num}")

        # start timer
        start_time = time.time()

        # Generate 'npart' initial EXP-HYDRO model parameters
        params = [ExphydroParameters() for j in range(npart)]

        # Initialise the model by loading its climate inputs
        model = ExphydroModel(P, PET, T) # important select the original model (V1) and pass PET instead of ET

        # Calibrate the model to identify optimal parameter set
        paramsmax = Calibration.pso_maximise(model, params, Qobs, ObjectiveFunction.klinggupta, calperiods_obs, calperiods_sim)
        print('Calibration run KGE value = ', paramsmax.objval)

        # Run the optimised model for validation period
        Qsim = model.simulate(paramsmax)

        # calculate the Kling-Gupta Efficiency for the period after the calibration ot the end!
        kge = ObjectiveFunction.klinggupta(Qobs[calperiods_obs[1]:], Qsim[calperiods_sim[1]:])
        print('Independent run KGE value = ', kge)

        print("Total runtime: %s seconds" % (time.time() - start_time))

        # calibrated parameters
        f = paramsmax.f.value
        smax = paramsmax.smax.value
        qmax = paramsmax.qmax.value
        ddf = paramsmax.ddf.value
        mint = paramsmax.mint.value
        maxt = paramsmax.maxt.value

        params_cal = np.array([f, smax, qmax, ddf, mint, maxt])
        print(f"Calibrated parameters: {params_cal}")

        # save parameters
        filename = f"calibrated_parameters_ID_{ID_num}_method_{method}_swarm_size_{npart}.npy"
        file_path = os.path.join(path_calibrated_parameters_v1, "1981_2001/", filename)
        np.save(file_path, params_cal)


# In[ ]:





# ### 3.3.2 EXP-HYDRO (modified) LamaH-CE

# In[ ]:


# "Run_exphydro_lumped_pso.py"

# Programmer(s): Sopan Patil.

""" MAIN PROGRAM FILE
Run this file to optimise the model parameters of the spatially lumped
version of EXP-HYDRO model using Particle Swarm Optimisation (PSO) algorithm.
"""


# In[ ]:


# set paramters for calibration
# Specify the no. of parameter sets (particles) in a PSO swarm
npart = 20

method = "PSO" # this is just for saving the file


# 🔥 **NOTE:** The data loader was changed so it takes as input now the start and end of the data period. Since we want to load the WHOLE timeseries here (for the independent run) we need to go from "start_cal" to "stop_test"

# In[ ]:


# now create a loop to do this for all catchments
for ID_num in catchment_attrs_red['ID']:

    # use this since code was interrupted
    if ID_num > 562: # 255

        # ID_num is now the current ID

        # load dataframe
        df = load_data_new(path_vars_lamah_colab, path_qobs_lamah_colab, path_catchment_attrs_colab, ID_num, variables_lamah, start_cal, stop_test, qobs_mm=True)

        # assign the input variables
        P = df['prcp'].values
        T = df['tmean'].values
        ET = df['et'].values
        Qobs = df['qobs_mm'].values # this is important! EXP-HYDRO expects units mm not m3/d

        print(f"STARTING CALIBRATION FOR CATCHMENT: {ID_num}")

        # start timer
        start_time = time.time()

        # Generate 'npart' initial EXP-HYDRO model parameters
        params = [ExphydroParameters() for j in range(npart)]

        # Initialise the model by loading its climate inputs
        model = ExphydroModelMod(P, ET, T)

        # Calibrate the model to identify optimal parameter set
        paramsmax = Calibration.pso_maximise(model, params, Qobs, ObjectiveFunction.klinggupta, calperiods_obs, calperiods_sim)
        print('Calibration run KGE value = ', paramsmax.objval)

        # Run the optimised model for validation period
        Qsim = model.simulate(paramsmax)

        # calculate the Kling-Gupta Efficiency for the period after the calibration ot the end!
        kge = ObjectiveFunction.klinggupta(Qobs[calperiods_obs[1]:], Qsim[calperiods_sim[1]:])
        print('Independent run KGE value = ', kge)

        print("Total runtime: %s seconds" % (time.time() - start_time))

        # calibrated parameters
        f = paramsmax.f.value
        smax = paramsmax.smax.value
        qmax = paramsmax.qmax.value
        ddf = paramsmax.ddf.value
        mint = paramsmax.mint.value
        maxt = paramsmax.maxt.value

        params_cal = np.array([f, smax, qmax, ddf, mint, maxt])
        print(f"Calibrated parameters: {params_cal}")

        # save parameters
        filename = f"calibrated_parameters_ID_{ID_num}_method_{method}_swarm_size_{npart}.npy"
        file_path = os.path.join(path_calibrated_parameters_v2, "1981_2001/", filename)
        np.save(file_path, params_cal)




# In[ ]:




