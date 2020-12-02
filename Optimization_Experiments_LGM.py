#!/usr/bin/env python
# coding: utf-8

# # Optimizations of ZOEE to HadCM3, CESM, Pages2k - p1000

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

from ZOEE.modules.configuration import importer, add_sellersparameters, parameterinterpolatorstepwise
from ZOEE.modules.variables import variable_importer, Vars, Base
from ZOEE.modules.optimization import optimization, ZOEE_optimization
# from ZOEE import update_plotstyle, moving_average
from ZOEE.modules.rk4 import rk4alg

# import matplotlib
# update_plotstyle()
# matplotlib.rcParams['lines.linewidth']=1.1


# ## General optimization setup

# In[3]:


Lat_10, ZMT_10 = np.loadtxt('Experiments/ERA5/ERA_1961_1990_10deg_ZMTanomaly.txt', delimiter=',')

# In[4]:


"""Declaration of parameter setup, with initial parameters, parameter boundaries and 
the parameter pertubation to estimate the cost function gradient"""
P0 = np.array([70 * 4.2e6, 200, 1.9, 1.165, 1.165, 1.165])
# P0=Get_PGamma[0]
Pmin = np.array([1 * 4.2e6, 170, 1.3, 0.9, 0.9, 0.9])
Pmax = np.array([100 * 4.2e6, 240, 2.5, 1.3, 1.3, 1.3])
P_pert_ratio = 1 / 10000

# # Optimizations of ZOEE to HadCM3 - LGM

# ## General optimization setup

# In[5]:


Lat_10, HadCM3_LGM_ZMT_target = np.loadtxt('Experiments/HadCM3/HadCM3_LGM_av_10deg_ZMT.csv', delimiter=',')
Lat_10, HadCM3_LGM_ZMT_target_anomaly = np.loadtxt('Experiments/HadCM3/HadCM3_LGM_av_10deg_ZMTanomaly.csv',
                                                   delimiter=',')
HadCM3_LGM_GMT = np.loadtxt('Experiments/HadCM3/HadCM3_LGM_850_1850_anomaly.txt', delimiter=',')

# In[8]:


## ZMT anomaly

"""Decleration of optimization configuration"""
grid = np.linspace(-85, 85, 18)
optimization_setup_LGM_an = optimization(mode='Coupled',
                                         target={'ZMT': HadCM3_LGM_ZMT_target_anomaly, 'GMT': HadCM3_LGM_GMT[1]},
                                         ZMT_response=True,
                                         GMT_response=True,
                                         response_average_length=30 * 12,
                                         num_steps=20,
                                         num_data=12000,
                                         gamma0=1e-8,
                                         cost_function_type='LeastSquare',
                                         cost_weight='cross_weight',
                                         cost_ratio=None,
                                         ZMT=HadCM3_LGM_ZMT_target,
                                         GMT=283.15,
                                         precision=0,
                                         grid=grid)

"""Declaration of parameter setup, with initial parameters, parameter boundaries and 
the parameter pertubation to estimate the cost function gradient"""

optimization_setup_LGM_an.give_parameters(P0, Pmin, Pmax, P_pert_ratio)

# In[9]:


"""Import the configuration that is required to run your specific model"""

config_HadCM3_LGM = importer('Experiments/HadCM3/Config_HadCM3_LGM_fixed.ini')
parallel_config = {'number_of_parameters': 6, 'number_of_cycles': 1, 'number_of_parallels': 13}
variable_importer(config_HadCM3_LGM, initialZMT=False, parallel=True, parallel_config=parallel_config)
config_HadCM3_LGM, Sellers = add_sellersparameters(config_HadCM3_LGM, parameterinterpolatorstepwise,
                                                   'ZOEE/config/SellersParameterization.ini', 4, 2, True, False)

"""Import the class of your model that has to be defined in ZOEE.modules.optimization. And give it whatever 
configuration it requires"""

parameter_labels = [['eqparam', 'c_ao'], ['func3', 'a'], ['func3', 'b'], ['func4', 'factor_oc'],
                    ['func4', 'factor_kwv'], ['func4', 'factor_kair']]
parameter_levels = np.array([None, None, None, None, None, None])
elevation = -0.0065 * np.array(Sellers[1][1])
ZOEE_HadCM3_LGM = ZOEE_optimization(6, parameter_labels, parameter_levels, True, elevation, 'Coupled', 12000,
                                    monthly=True)
# model_setup=[2,'ZMT',parameter_labels,parameter_levels,elevation,True]

"""Execture optimize to start the optimization, giving it your model imported in the step before and configuration
required to run your model"""

print("Optimization >>> HadCM3 LGM target (ZMT anomaly)")
F_HadCM3_LGM_an, dF_HadCM3_LGM_an, P_HadCM3_LGM_an, Ptrans_HadCM3_LGM_an, gamma_HadCM3_LGM_an, data_HadCM3_LGM_an = optimization_setup_LGM_an.optimize(
    ZOEE_HadCM3_LGM, config_HadCM3_LGM)

# In[ ]:


np.savetxt('Experiments/Output/HadCM3_LGM_P_20_resp_an.txt', P_HadCM3_LGM_an, delimiter=',')
np.savetxt('Experiments/Output/HadCM3_LGM_F_20_resp_an.txt', F_HadCM3_LGM_an, delimiter=',')
np.savetxt('Experiments/Output/HadCM3_LGM_dF_20_resp_an.txt', dF_HadCM3_LGM_an, delimiter=',')
np.savetxt('Experiments/Output/HadCM3_LGM_ZMT_20_resp_an.txt', data_HadCM3_LGM_an[0][:, 0], delimiter=',')
np.savetxt('Experiments/Output/HadCM3_LGM_GMT_20_resp_an.txt', data_HadCM3_LGM_an[1][:, 0], delimiter=',')

# In[ ]:


# ZMT absolute

"""Decleration of optimization configuration"""
grid = np.linspace(-85, 85, 18)
optimization_setup_LGM = optimization(mode='Coupled',
                                      target={'ZMT': HadCM3_LGM_ZMT_target, 'GMT': HadCM3_LGM_GMT[1]},
                                      ZMT_response=False,
                                      GMT_response=True,
                                      response_average_length=30 * 12,
                                      num_steps=20,
                                      num_data=12000,
                                      gamma0=1e-8,
                                      cost_function_type='LeastSquare',
                                      cost_weight='cross_weight',
                                      cost_ratio=None,
                                      ZMT=HadCM3_LGM_ZMT_target,
                                      GMT=283.15,
                                      precision=0,
                                      grid=grid)

"""Declaration of parameter setup, with initial parameters, parameter boundaries and 
the parameter pertubation to estimate the cost function gradient"""

optimization_setup_LGM.give_parameters(P0, Pmin, Pmax, P_pert_ratio)

# In[ ]:


"""Execture optimize to start the optimization, giving it your model imported in the step before and configuration
required to run your model"""

print("Optimization >>> HadCM3 LGM target (ZMT absolute)")
F_HadCM3_LGM, dF_HadCM3_LGM, P_HadCM3_LGM, Ptrans_HadCM3_LGM, gamma_HadCM3_LGM, data_HadCM3_LGM = optimization_setup_LGM.optimize(
    ZOEE_HadCM3_LGM, config_HadCM3_LGM)

# In[ ]:


np.savetxt('Experiments/Output/HadCM3_LGM_P_20_abs.txt', P_HadCM3_LGM, delimiter=',')
np.savetxt('Experiments/Output/HadCM3_LGM_F_20_abs.txt', F_HadCM3_LGM, delimiter=',')
np.savetxt('Experiments/Output/HadCM3_LGM_dF_20_abs.txt', dF_HadCM3_LGM, delimiter=',')
np.savetxt('Experiments/Output/HadCM3_LGM_ZMT_20_abs.txt', data_HadCM3_LGM[0][:, 0], delimiter=',')
np.savetxt('Experiments/Output/HadCM3_LGM_GMT_20_abs.txt', data_HadCM3_LGM[1][:, 0], delimiter=',')
