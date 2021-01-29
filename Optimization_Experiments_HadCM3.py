#!/usr/bin/env python
# coding: utf-8

# # Optimizations of ZOEE to HadCM3 Data

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

from ZOEE.modules.configuration import importer, add_sellersparameters, parameterinterpolatorstepwise
from ZOEE.modules.variables import variable_importer, Vars, Base
from ZOEE.modules.optimization import optimization, ZOEE_optimization
# from ZOEE import update_plotstyle, moving_average
from ZOEE.modules.rk4 import rk4alg
import pandas as pd

# import matplotlib
# update_plotstyle()
# matplotlib.rcParams['lines.linewidth']=1.1


# ## Target Data

# In[2]:


HadCM3_ZMT = pd.read_csv('Experiments/HadCM3/HadCM3_ZMT_10deg.csv')
HadCM3_ZMT_anomaly = pd.read_csv('Experiments/HadCM3/HadCM3_ZMT_10deg.csv')
HadCM3_GMT = pd.read_csv('Experiments/HadCM3/HadCM3_850.csv')
HadCM3_GMT_anomaly = pd.read_csv('Experiments/HadCM3/HadCM3_850_anomaly.csv')

# In[3]:


HadCM3_ZMT.columns

# In[4]:


Config_data = {'xnagd': 'Config_HadCM3_fixed.ini', 'xnagf': 'Config_HadCM3_fixed.ini',
               'xnagg': 'Config_HadCM3_fixed.ini',
               'pi_forc': 'Config_HadCM3_fixed.ini',
               'xmzke': 'Config_HadCM3_LGM_fixed.ini', 'xmzkg': 'Config_HadCM3_LGM_fixed.ini',
               'xmzkh': 'Config_HadCM3_LGM_fixed.ini',
               'LGM_forc': 'Config_HadCM3_LGM_fixed.ini',
               'xmzkb': 'Config_HadCM3_LGM_fixed_m.ini', 'xmzkc': 'Config_HadCM3_LGM_fixed_p.ini'}

# ## General optimization setup

# In[5]:


P0 = np.array([70 * 4.2e6, 200, 1.9, 1, 1, 1])
# P0=Get_PGamma[0]
Pmin = np.array([1 * 4.2e6, 170, 1.3, 0.8, 0.8, 0.8])
Pmax = np.array([100 * 4.2e6, 240, 2.5, 1.2, 1.2, 1.2])
P_pert_ratio = 1 / 10000

parameter_labels = [['eqparam', 'c_ao'], ['func3', 'a'], ['func3', 'b'], ['func4', 'factor_oc'],
                    ['func4', 'factor_kwv'], ['func4', 'factor_kair']]
parameter_levels = np.array([None, None, None, None, None, None])

# In[6]:


"""Decleration of optimization configuration"""
optimization_setup = optimization(mode='Coupled',
                                  target=None,
                                  ZMT_response=False,
                                  GMT_response=True,
                                  response_average_length=30 * 12,
                                  num_steps=20,
                                  num_data=12000,
                                  gamma0=1e-8,
                                  cost_function_type='LeastSquare',
                                  cost_weight='cross_weight',
                                  cost_ratio=None,
                                  ZMT=HadCM3_ZMT['pi_forc'] + 288.15,
                                  GMT=288.15,
                                  precision=0,
                                  grid=HadCM3_ZMT['lat'].values)

"""Declaration of parameter setup, with initial parameters, parameter boundaries and 
the parameter pertubation to estimate the cost function gradient"""

optimization_setup.give_parameters(P0, Pmin, Pmax, P_pert_ratio)

"""Decleration of optimization configuration"""
optimization_setup_an = optimization(mode='Coupled',
                                     target=None,
                                     ZMT_response=True,
                                     GMT_response=True,
                                     response_average_length=30 * 12,
                                     num_steps=20,
                                     num_data=12000,
                                     gamma0=1e-8,
                                     cost_function_type='LeastSquare',
                                     cost_weight='cross_weight',
                                     cost_ratio=None,
                                     ZMT=HadCM3_ZMT['pi_forc'] + 288.15,
                                     GMT=288.15,
                                     precision=0,
                                     grid=HadCM3_ZMT['lat'].values)

"""Declaration of parameter setup, with initial parameters, parameter boundaries and 
the parameter pertubation to estimate the cost function gradient"""

optimization_setup_an.give_parameters(P0, Pmin, Pmax, P_pert_ratio)

## ZMT anomaly

"""Decleration of optimization configuration"""
optimization_setup_LGM_an = optimization(mode='Coupled',
                                         target=None,
                                         ZMT_response=True,
                                         GMT_response=True,
                                         response_average_length=30 * 12,
                                         num_steps=20,
                                         num_data=12000,
                                         gamma0=1e-8,
                                         cost_function_type='LeastSquare',
                                         cost_weight='cross_weight',
                                         cost_ratio=None,
                                         ZMT=HadCM3_ZMT['LGM_forc'],
                                         GMT=283.15,
                                         precision=0,
                                         grid=HadCM3_ZMT['lat'].values)

"""Declaration of parameter setup, with initial parameters, parameter boundaries and 
the parameter pertubation to estimate the cost function gradient"""

optimization_setup_LGM_an.give_parameters(P0, Pmin, Pmax, P_pert_ratio)

optimization_setup_LGM = optimization(mode='Coupled',
                                      target=None,
                                      ZMT_response=False,
                                      GMT_response=True,
                                      response_average_length=30 * 12,
                                      num_steps=20,
                                      num_data=12000,
                                      gamma0=1e-8,
                                      cost_function_type='LeastSquare',
                                      cost_weight='cross_weight',
                                      cost_ratio=None,
                                      ZMT=HadCM3_ZMT['LGM_forc'],
                                      GMT=283.15,
                                      precision=0,
                                      grid=HadCM3_ZMT['lat'].values)

"""Declaration of parameter setup, with initial parameters, parameter boundaries and 
the parameter pertubation to estimate the cost function gradient"""

optimization_setup_LGM.give_parameters(P0, Pmin, Pmax, P_pert_ratio)

# ## PI runs

# ### ZMT absolute, GMT anomaly

# In[8]:


for run in ['xnagd', 'xnagf', 'xnagg', 'pi_forc']:
    """Import the configuration that is required to run your specific model"""

    config_HadCM3 = importer('Experiments/HadCM3/' + Config_data[run])
    parallel_config = {'number_of_parameters': 6, 'number_of_cycles': 1, 'number_of_parallels': 13}
    variable_importer(config_HadCM3, initialZMT=False, parallel=True, parallel_config=parallel_config)
    config_HadCM3, Sellers = add_sellersparameters(config_HadCM3, parameterinterpolatorstepwise,
                                                   'ZOEE/config/SellersParameterization.ini', 4, 2, True, False)
    elevation = -0.0065 * np.array(Sellers[1][1])

    """Import the class of your model that has to be defined in ZOEE.modules.optimization. And give it whatever 
    configuration it requires"""
    ZOEE_HadCM3 = ZOEE_optimization(6, parameter_labels, parameter_levels, True, elevation, 'Coupled', 12000,
                                    monthly=True)

    # model_setup=[2,'ZMT',parameter_labels,parameter_levels,elevation,True]

    """Execture optimize to start the optimization, giving it your model imported in the step before and configuration
    required to run your model"""
    optimization_setup.target = {'ZMT': HadCM3_ZMT[run], 'GMT': HadCM3_GMT_anomaly[run]}
    optimization_setup.num_data = 12000
    # optimization_setup.response=False
    print("Optimization >>> {}".format(run))
    F_HadCM3, dF_HadCM3, P_HadCM3, Ptrans_HadCM3, gamma_HadCM3, Data_HadCM3 = optimization_setup.optimize(ZOEE_HadCM3,
                                                                                                          config_HadCM3)

    df = pd.DataFrame()
    df['F'] = pd.Series(F_HadCM3.tolist())
    df['dF'] = pd.Series(dF_HadCM3.tolist())
    df['P'] = pd.Series(P_HadCM3.tolist())
    df['Ptrans'] = pd.Series(Ptrans_HadCM3.tolist())
    df['Gamma'] = pd.Series(gamma_HadCM3.tolist())
    df['ZMT'] = pd.Series(Data_HadCM3[0][:, 0].tolist())
    df['GMT'] = pd.Series(Data_HadCM3[1][:, 0].tolist())
    df.to_csv('Experiments/Output/' + run + '_20abs.csv')


# ### ZMT anomaly, GMT anomaly

# In[ ]:


for run in ['xnagd', 'xnagf', 'xnagg', 'pi_forc']:
    """Import the configuration that is required to run your specific model"""

    config_HadCM3 = importer('Experiments/HadCM3/' + Config_data[run])
    parallel_config = {'number_of_parameters': 6, 'number_of_cycles': 1, 'number_of_parallels': 13}
    variable_importer(config_HadCM3, initialZMT=False, parallel=True, parallel_config=parallel_config)
    config_HadCM3, Sellers = add_sellersparameters(config_HadCM3, parameterinterpolatorstepwise,
                                                   'ZOEE/config/SellersParameterization.ini', 4, 2, True, False)
    elevation = -0.0065 * np.array(Sellers[1][1])

    """Import the class of your model that has to be defined in ZOEE.modules.optimization. And give it whatever 
    configuration it requires"""
    ZOEE_HadCM3 = ZOEE_optimization(6, parameter_labels, parameter_levels, True, elevation, 'Coupled', 12000,
                                    monthly=True)

    # model_setup=[2,'ZMT',parameter_labels,parameter_levels,elevation,True]

    """Execture optimize to start the optimization, giving it your model imported in the step before and configuration
    required to run your model"""
    optimization_setup_an.target = {'ZMT': HadCM3_ZMT_anomaly[run], 'GMT': HadCM3_GMT_anomaly[run]}
    optimization_setup_an.num_data = 12000
    # optimization_setup.response=False
    print("Optimization >>> {}".format(run))
    F_HadCM3, dF_HadCM3, P_HadCM3, Ptrans_HadCM3, gamma_HadCM3, Data_HadCM3 = optimization_setup_an.optimize(
        ZOEE_HadCM3, config_HadCM3)

    df = pd.DataFrame()
    df['F'] = pd.Series(F_HadCM3.tolist())
    df['dF'] = pd.Series(dF_HadCM3.tolist())
    df['P'] = pd.Series(P_HadCM3.tolist())
    df['Ptrans'] = pd.Series(Ptrans_HadCM3.tolist())
    df['Gamma'] = pd.Series(gamma_HadCM3.tolist())
    df['ZMT'] = pd.Series(Data_HadCM3[0][:, 0].tolist())
    df['GMT'] = pd.Series(Data_HadCM3[1][:, 0].tolist())
    df.to_csv('Experiments/Output/' + run + '_20an.csv')


# ## LGM runs

# ### LGM - ZMT absolute, GMT anomaly -  4x 185ppm, 1x 150ppm, 1x 210ppm

# In[ ]:


for run in ['xmzke', 'xmzkg', 'xmzkh', 'LGM_forc', 'xmzkb', 'xmzkc']:
    """Import the configuration that is required to run your specific model"""

    config_HadCM3 = importer('Experiments/HadCM3/' + Config_data[run])
    parallel_config = {'number_of_parameters': 6, 'number_of_cycles': 1, 'number_of_parallels': 13}
    variable_importer(config_HadCM3, initialZMT=False, parallel=True, parallel_config=parallel_config)
    config_HadCM3, Sellers = add_sellersparameters(config_HadCM3, parameterinterpolatorstepwise,
                                                   'ZOEE/config/SellersParameterization.ini', 4, 2, True, False)
    elevation = -0.0065 * (np.array(Sellers[1][1]) + 125)

    """Import the class of your model that has to be defined in ZOEE.modules.optimization. And give it whatever 
    configuration it requires"""
    ZOEE_HadCM3 = ZOEE_optimization(6, parameter_labels, parameter_levels, True, elevation, 'Coupled', 12000,
                                    monthly=True)

    # model_setup=[2,'ZMT',parameter_labels,parameter_levels,elevation,True]

    """Execture optimize to start the optimization, giving it your model imported in the step before and configuration
    required to run your model"""
    optimization_setup_LGM.target = {'ZMT': HadCM3_ZMT[run], 'GMT': HadCM3_GMT_anomaly[run]}
    optimization_setup_LGM.num_data = 12000
    # optimization_setup.response=False
    print("Optimization >>> {}".format(run))
    F_HadCM3, dF_HadCM3, P_HadCM3, Ptrans_HadCM3, gamma_HadCM3, Data_HadCM3 = optimization_setup_LGM.optimize(
        ZOEE_HadCM3, config_HadCM3)

    df = pd.DataFrame()
    df['F'] = pd.Series(F_HadCM3.tolist())
    df['dF'] = pd.Series(dF_HadCM3.tolist())
    df['P'] = pd.Series(P_HadCM3.tolist())
    df['Ptrans'] = pd.Series(Ptrans_HadCM3.tolist())
    df['Gamma'] = pd.Series(gamma_HadCM3.tolist())
    df['ZMT'] = pd.Series(Data_HadCM3[0][:, 0].tolist())
    df['GMT'] = pd.Series(Data_HadCM3[1][:, 0].tolist())
    df.to_csv('Experiments/Output/' + run + '_20abs.csv')


# ### LGM - ZMT anomaly, GMT anomaly -  4x 185ppm, 1x 150ppm, 1x 210ppm

# In[ ]:


for run in ['xmzke', 'xmzkg', 'xmzkh', 'LGM_forc', 'xmzkb', 'xmzkc']:
    """Import the configuration that is required to run your specific model"""

    config_HadCM3 = importer('Experiments/HadCM3/' + Config_data[run])
    parallel_config = {'number_of_parameters': 6, 'number_of_cycles': 1, 'number_of_parallels': 13}
    variable_importer(config_HadCM3, initialZMT=False, parallel=True, parallel_config=parallel_config)
    config_HadCM3, Sellers = add_sellersparameters(config_HadCM3, parameterinterpolatorstepwise,
                                                   'ZOEE/config/SellersParameterization.ini', 4, 2, True, False)
    elevation = -0.0065 * (np.array(Sellers[1][1]) + 125)

    """Import the class of your model that has to be defined in ZOEE.modules.optimization. And give it whatever 
    configuration it requires"""
    ZOEE_HadCM3 = ZOEE_optimization(6, parameter_labels, parameter_levels, True, elevation, 'Coupled', 12000,
                                    monthly=True)

    # model_setup=[2,'ZMT',parameter_labels,parameter_levels,elevation,True]

    """Execture optimize to start the optimization, giving it your model imported in the step before and configuration
    required to run your model"""
    optimization_setup_LGM_an.target = {'ZMT': HadCM3_ZMT_anomaly[run], 'GMT': HadCM3_GMT_anomaly[run]}
    optimization_setup_LGM_an.num_data = 12000
    # optimization_setup.response=False
    print("Optimization >>> {}".format(run))
    F_HadCM3, dF_HadCM3, P_HadCM3, Ptrans_HadCM3, gamma_HadCM3, Data_HadCM3 = optimization_setup_LGM_an.optimize(
        ZOEE_HadCM3, config_HadCM3)

    df = pd.DataFrame()
    df['F'] = pd.Series(F_HadCM3.tolist())
    df['dF'] = pd.Series(dF_HadCM3.tolist())
    df['P'] = pd.Series(P_HadCM3.tolist())
    df['Ptrans'] = pd.Series(Ptrans_HadCM3.tolist())
    df['Gamma'] = pd.Series(gamma_HadCM3.tolist())
    df['ZMT'] = pd.Series(Data_HadCM3[0][:, 0].tolist())
    df['GMT'] = pd.Series(Data_HadCM3[1][:, 0].tolist())
    df.to_csv('Experiments/Output/' + run + '_20an.csv')
