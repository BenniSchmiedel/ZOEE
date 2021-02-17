#!/usr/bin/env python
# coding: utf-8

# # Optimizations of ZOEE to HadCM3 Data
# 
#  1) 
#     - The model is optimized to PI Data, with the standard Parameters A,B,C_ao,f_wv,f_sh,f_oc.
#         - PI_forc
#     - In the next step the model is optimized to LGM Data with only f_wv,f_sh,f_oc
#         - LGM_forc, xmzkb, xmzkc
#  
#  2) 
#      - The model is optimized to LGM Data, with the standard Parameters A,B,C_ao,f_wv,f_sh,f_oc.
#         - LGM_forc
#     - In the next step the model is optimized to altered LGM runs with only f_wv,f_sh,f_oc
#         - xmzkb, xmzkc

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

from ZOEE.modules.configuration import importer, add_sellersparameters, parameterinterpolatorstepwise
from ZOEE.modules.variables import variable_importer, Vars, Base
from ZOEE.modules.optimization import optimization, ZOEE_optimization
#from ZOEE import update_plotstyle, moving_average
from ZOEE.modules.rk4 import rk4alg
import pandas as pd
#import matplotlib
#update_plotstyle()
#matplotlib.rcParams['lines.linewidth']=1.1


# ## Target Data

# In[ ]:


HadCM3_ZMT=pd.read_csv('Experiments/HadCM3/HadCM3_ZMT_10deg.csv')
HadCM3_ZMT_anomaly=pd.read_csv('Experiments/HadCM3/HadCM3_ZMT_anomaly_10deg.csv')
HadCM3_GMT=pd.read_csv('Experiments/HadCM3/HadCM3_850.csv')
HadCM3_GMT_anomaly=pd.read_csv('Experiments/HadCM3/HadCM3_850_anomaly.csv')


# In[ ]:


HadCM3_ZMT.columns


# In[ ]:


Config_data={'xnagd':'Config_HadCM3_fixed.ini', 'xnagf':'Config_HadCM3_fixed.ini', 'xnagg':'Config_HadCM3_fixed.ini',
             'pi_forc':'Config_HadCM3_fixed.ini',
             'xmzke':'Config_HadCM3_LGM_fixed.ini', 'xmzkg':'Config_HadCM3_LGM_fixed.ini','xmzkh':'Config_HadCM3_LGM_fixed.ini',
             'LGM_forc':'Config_HadCM3_LGM_fixed.ini',
             'xmzkb':'Config_HadCM3_LGM_fixed_m.ini', 'xmzkc':'Config_HadCM3_LGM_fixed_p.ini'}


# ## General optimization setup

# In[ ]:


P0_6=np.array([70*4.2e6,200,1.9,1,1,1])
#P0=Get_PGamma[0]
Pmin_6=np.array([1*4.2e6,170,1.3,0.8,0.8,0.8])
Pmax_6=np.array([100*4.2e6,240,2.5,1.2,1.2,1.2])
P_pert_ratio_6=1/10000

parameter_labels_6=[['eqparam','c_ao'],['func3','a'],['func3','b'],['func4','factor_oc'],['func4','factor_kwv'],['func4','factor_kair']]
parameter_levels_6=np.array([None,None,None,None,None,None])

P0_3=np.array([1,1,1])
#P0=Get_PGamma[0]
Pmin_3=np.array([0.8,0.8,0.8])
Pmax_3=np.array([1.2,1.2,1.2])
P_pert_ratio_3=1/10000

parameter_labels_3=[['func4','factor_oc'],['func4','factor_kwv'],['func4','factor_kair']]
parameter_levels_3=np.array([None,None,None])


# In[ ]:


kwargs={'mode':'Coupled',
        'target':None,
        'ZMT_response':False, 
        'GMT_response':True,
        'response_average_length':30*12,
        'num_steps':20,
        'num_data':12000,
        'gamma0':1e-8,
        'cost_function_type':'LeastSquare',
        'cost_weight':'cross_weight',
        'cost_ratio':None,
        'ZMT':HadCM3_ZMT_anomaly['pi_forc']+288.15,
        'GMT':288.15,
        'precision':0,
        'grid' : HadCM3_ZMT['lat'].values}

kwargs_an = kwargs.copy()
kwargs_an['ZMT_response']= True

kwargs_LGM = kwargs.copy()
kwargs_LGM['ZMT']=HadCM3_ZMT_anomaly['LGM_forc']+283.15,
kwargs_LGM['GMT']=283.15

kwargs_LGM_an = kwargs_LGM.copy()
kwargs_LGM_an['ZMT_response']=True


# In[ ]:


"""Decleration of optimization configuration"""
optimization_setup_6 = optimization(**kwargs)
optimization_setup_6.give_parameters(P0_6,Pmin_6,Pmax_6,P_pert_ratio_6)

optimization_setup_6_an = optimization(**kwargs_an)
optimization_setup_6_an.give_parameters(P0_6,Pmin_6,Pmax_6,P_pert_ratio_6)

optimization_setup_3_LGM = optimization(**kwargs_LGM)
optimization_setup_3_LGM.give_parameters(P0_3,Pmin_3,Pmax_3,P_pert_ratio_3)

optimization_setup_3_LGM_an = optimization(**kwargs_LGM_an)
optimization_setup_3_LGM_an.give_parameters(P0_3,Pmin_3,Pmax_3,P_pert_ratio_3)

#optimization_setup_2 = optimization(**kwargs)
#optimization_setup_2.give_parameters(P0_1,Pmin_1,Pmax_1,P_pert_ratio_1)

#optimization_setup_2_an = optimization(**kwargs_an)
#optimization_setup_2_an.give_parameters(P0_1,Pmin_1,Pmax_1,P_pert_ratio_1)

optimization_setup_6_LGM = optimization(**kwargs_LGM)
optimization_setup_6_LGM.give_parameters(P0_6,Pmin_6,Pmax_6,P_pert_ratio_6)

optimization_setup_6_LGM_an = optimization(**kwargs_LGM_an)
optimization_setup_6_LGM_an.give_parameters(P0_6,Pmin_6,Pmax_6,P_pert_ratio_6)


# ## 1) PI into LGM transfer only

# ### ZMT absolute, GMT anomaly

# In[ ]:


### Run pi_forc Optimizations

run='pi_forc'

config_HadCM3=importer('Experiments/HadCM3/'+Config_data[run])
parallel_config = {'number_of_parameters': 6, 'number_of_cycles': 1,'number_of_parallels': 13}
variable_importer(config_HadCM3,initialZMT=False,parallel=True,parallel_config=parallel_config)
config_HadCM3,Sellers=add_sellersparameters(config_HadCM3, parameterinterpolatorstepwise,                                            'ZOEE/config/SellersParameterization.ini',4,2,True,False)
elevation=-0.0065*np.array(Sellers[1][1])

"""Import the class of your model that has to be defined in ZOEE.modules.optimization. And give it whatever 
configuration it requires"""
ZOEE_HadCM3 = ZOEE_optimization(6,parameter_labels_6,parameter_levels_6,True,elevation,'Coupled',12000,monthly=True)

#model_setup=[2,'ZMT',parameter_labels,parameter_levels,elevation,True]

"""Execture optimize to start the optimization, giving it your model imported in the step before and configuration
required to run your model"""
optimization_setup_6.target={'ZMT':HadCM3_ZMT[run],'GMT':HadCM3_GMT_anomaly[run]}

print("Optimization >>> {}".format(run))
F_HadCM3, dF_HadCM3, P_HadCM3, Ptrans_HadCM3, gamma_HadCM3, Data_HadCM3=optimization_setup_6.optimize(ZOEE_HadCM3,config_HadCM3)

df = pd.DataFrame()
df['F'] = pd.Series(F_HadCM3.tolist())
df['dF'] = pd.Series(dF_HadCM3.tolist())
df['P'] = pd.Series(P_HadCM3.tolist())
df['Ptrans'] = pd.Series(Ptrans_HadCM3.tolist())
df['Gamma'] = pd.Series(gamma_HadCM3.tolist())
df['ZMT'] = pd.Series(Data_HadCM3[0][:,0].tolist())
df['GMT'] = pd.Series(Data_HadCM3[1][:,0].tolist())
df.to_csv('Experiments/Output/'+run+'_conf1_abs.csv')

### Run LGM optimizations

for run in ['LGM_forc','xmzkb','xmzkc']: 

    config_HadCM3=importer('Experiments/HadCM3/'+Config_data[run])
    parallel_config = {'number_of_parameters': 3, 'number_of_cycles': 1,'number_of_parallels': 7}
    variable_importer(config_HadCM3,initialZMT=False,parallel=True,parallel_config=parallel_config)
    config_HadCM3,Sellers=add_sellersparameters(config_HadCM3, parameterinterpolatorstepwise,                                                'ZOEE/config/SellersParameterization.ini',4,2,True,False)
    elevation=-0.0065*(np.array(Sellers[1][1])+125)

    """Import the class of your model that has to be defined in ZOEE.modules.optimization. And give it whatever 
    configuration it requires"""
    ZOEE_HadCM3 = ZOEE_optimization(3,parameter_labels_3,parameter_levels_3,True,elevation,'Coupled',12000,monthly=True)

    #model_setup=[2,'ZMT',parameter_labels,parameter_levels,elevation,True]

    """Execture optimize to start the optimization, giving it your model imported in the step before and configuration
    required to run your model"""
    optimization_setup_3_LGM.target={'ZMT':HadCM3_ZMT[run],'GMT':HadCM3_GMT_anomaly[run]}

    print("Optimization >>> {}".format(run))
    F_HadCM3, dF_HadCM3, P_HadCM3, Ptrans_HadCM3, gamma_HadCM3, Data_HadCM3=    optimization_setup_3_LGM.optimize(ZOEE_HadCM3,config_HadCM3)

    df = pd.DataFrame()
    df['F'] = pd.Series(F_HadCM3.tolist())
    df['dF'] = pd.Series(dF_HadCM3.tolist())
    df['P'] = pd.Series(P_HadCM3.tolist())
    df['Ptrans'] = pd.Series(Ptrans_HadCM3.tolist())
    df['Gamma'] = pd.Series(gamma_HadCM3.tolist())
    df['ZMT'] = pd.Series(Data_HadCM3[0][:,0].tolist())
    df['GMT'] = pd.Series(Data_HadCM3[1][:,0].tolist())
    df.to_csv('Experiments/Output/'+run+'_conf1_abs.csv')


# ### ZMT anomaly, GMT anomaly

# In[ ]:


### Run pi_forc Optimizations

run='pi_forc'

config_HadCM3=importer('Experiments/HadCM3/'+Config_data[run])
parallel_config = {'number_of_parameters': 6, 'number_of_cycles': 1,'number_of_parallels': 13}
variable_importer(config_HadCM3,initialZMT=False,parallel=True,parallel_config=parallel_config)
config_HadCM3,Sellers=add_sellersparameters(config_HadCM3, parameterinterpolatorstepwise,                                            'ZOEE/config/SellersParameterization.ini',4,2,True,False)
elevation=-0.0065*np.array(Sellers[1][1])

"""Import the class of your model that has to be defined in ZOEE.modules.optimization. And give it whatever 
configuration it requires"""
ZOEE_HadCM3 = ZOEE_optimization(6,parameter_labels_6,parameter_levels_6,True,elevation,'Coupled',12000,monthly=True)

#model_setup=[2,'ZMT',parameter_labels,parameter_levels,elevation,True]

"""Execture optimize to start the optimization, giving it your model imported in the step before and configuration
required to run your model"""
optimization_setup_6_an.target={'ZMT':HadCM3_ZMT_anomaly[run],'GMT':HadCM3_GMT_anomaly[run]}

print("Optimization >>> {}".format(run))
F_HadCM3, dF_HadCM3, P_HadCM3, Ptrans_HadCM3, gamma_HadCM3, Data_HadCM3=optimization_setup_6_an.optimize(ZOEE_HadCM3,config_HadCM3)

df = pd.DataFrame()
df['F'] = pd.Series(F_HadCM3.tolist())
df['dF'] = pd.Series(dF_HadCM3.tolist())
df['P'] = pd.Series(P_HadCM3.tolist())
df['Ptrans'] = pd.Series(Ptrans_HadCM3.tolist())
df['Gamma'] = pd.Series(gamma_HadCM3.tolist())
df['ZMT'] = pd.Series(Data_HadCM3[0][:,0].tolist())
df['GMT'] = pd.Series(Data_HadCM3[1][:,0].tolist())
df.to_csv('Experiments/Output/'+run+'_conf1_an.csv')

### Run LGM optimizations

for run in ['LGM_forc','xmzkb','xmzkc']: 

    config_HadCM3=importer('Experiments/HadCM3/'+Config_data[run])
    parallel_config = {'number_of_parameters': 3, 'number_of_cycles': 1,'number_of_parallels': 7}
    variable_importer(config_HadCM3,initialZMT=False,parallel=True,parallel_config=parallel_config)
    config_HadCM3,Sellers=add_sellersparameters(config_HadCM3, parameterinterpolatorstepwise,                                                'ZOEE/config/SellersParameterization.ini',4,2,True,False)
    elevation=-0.0065*(np.array(Sellers[1][1])+125)

    """Import the class of your model that has to be defined in ZOEE.modules.optimization. And give it whatever 
    configuration it requires"""
    ZOEE_HadCM3 = ZOEE_optimization(3,parameter_labels_3,parameter_levels_3,True,elevation,'Coupled',12000,monthly=True)

    #model_setup=[2,'ZMT',parameter_labels,parameter_levels,elevation,True]

    """Execture optimize to start the optimization, giving it your model imported in the step before and configuration
    required to run your model"""
    optimization_setup_3_LGM_an.target={'ZMT':HadCM3_ZMT_anomaly[run],'GMT':HadCM3_GMT_anomaly[run]}

    print("Optimization >>> {}".format(run))
    F_HadCM3, dF_HadCM3, P_HadCM3, Ptrans_HadCM3, gamma_HadCM3, Data_HadCM3=    optimization_setup_3_LGM_an.optimize(ZOEE_HadCM3,config_HadCM3)

    df = pd.DataFrame()
    df['F'] = pd.Series(F_HadCM3.tolist())
    df['dF'] = pd.Series(dF_HadCM3.tolist())
    df['P'] = pd.Series(P_HadCM3.tolist())
    df['Ptrans'] = pd.Series(Ptrans_HadCM3.tolist())
    df['Gamma'] = pd.Series(gamma_HadCM3.tolist())
    df['ZMT'] = pd.Series(Data_HadCM3[0][:,0].tolist())
    df['GMT'] = pd.Series(Data_HadCM3[1][:,0].tolist())
    df.to_csv('Experiments/Output/'+run+'_conf1_an.csv')


# ## 2) LGM into LGM +/- CO2

# In[ ]:


### Run LGM_forc Optimizations

run='LGM_forc'

config_HadCM3=importer('Experiments/HadCM3/'+Config_data[run])
parallel_config = {'number_of_parameters': 6, 'number_of_cycles': 1,'number_of_parallels': 13}
variable_importer(config_HadCM3,initialZMT=False,parallel=True,parallel_config=parallel_config)
config_HadCM3,Sellers=add_sellersparameters(config_HadCM3, parameterinterpolatorstepwise,                                            'ZOEE/config/SellersParameterization.ini',4,2,True,False)
elevation=-0.0065*(np.array(Sellers[1][1])+125)

"""Import the class of your model that has to be defined in ZOEE.modules.optimization. And give it whatever 
configuration it requires"""
ZOEE_HadCM3 = ZOEE_optimization(6,parameter_labels_6,parameter_levels_6,True,elevation,'Coupled',12000,monthly=True)

#model_setup=[2,'ZMT',parameter_labels,parameter_levels,elevation,True]

"""Execture optimize to start the optimization, giving it your model imported in the step before and configuration
required to run your model"""
optimization_setup_6_LGM.target={'ZMT':HadCM3_ZMT[run],'GMT':HadCM3_GMT_anomaly[run]}

print("Optimization >>> {}".format(run))
F_HadCM3, dF_HadCM3, P_HadCM3, Ptrans_HadCM3, gamma_HadCM3, Data_HadCM3=optimization_setup_6_LGM.optimize(ZOEE_HadCM3,config_HadCM3)

df = pd.DataFrame()
df['F'] = pd.Series(F_HadCM3.tolist())
df['dF'] = pd.Series(dF_HadCM3.tolist())
df['P'] = pd.Series(P_HadCM3.tolist())
df['Ptrans'] = pd.Series(Ptrans_HadCM3.tolist())
df['Gamma'] = pd.Series(gamma_HadCM3.tolist())
df['ZMT'] = pd.Series(Data_HadCM3[0][:,0].tolist())
df['GMT'] = pd.Series(Data_HadCM3[1][:,0].tolist())
df.to_csv('Experiments/Output/'+run+'_conf2_abs.csv')

### Run LGM pm optimizations

for run in ['xmzkb','xmzkc']: 

    config_HadCM3=importer('Experiments/HadCM3/'+Config_data[run])
    parallel_config = {'number_of_parameters': 3, 'number_of_cycles': 1,'number_of_parallels': 7}
    variable_importer(config_HadCM3,initialZMT=False,parallel=True,parallel_config=parallel_config)
    config_HadCM3,Sellers=add_sellersparameters(config_HadCM3, parameterinterpolatorstepwise,                                                'ZOEE/config/SellersParameterization.ini',4,2,True,False)
    elevation=-0.0065*(np.array(Sellers[1][1])+125)

    """Import the class of your model that has to be defined in ZOEE.modules.optimization. And give it whatever 
    configuration it requires"""
    ZOEE_HadCM3 = ZOEE_optimization(3,parameter_labels_3,parameter_levels_3,True,elevation,'Coupled',12000,monthly=True)

    #model_setup=[2,'ZMT',parameter_labels,parameter_levels,elevation,True]

    """Execture optimize to start the optimization, giving it your model imported in the step before and configuration
    required to run your model"""
    optimization_setup_3_LGM.target={'ZMT':HadCM3_ZMT[run],'GMT':HadCM3_GMT_anomaly[run]}

    print("Optimization >>> {}".format(run))
    F_HadCM3, dF_HadCM3, P_HadCM3, Ptrans_HadCM3, gamma_HadCM3, Data_HadCM3=    optimization_setup_3_LGM.optimize(ZOEE_HadCM3,config_HadCM3)

    df = pd.DataFrame()
    df['F'] = pd.Series(F_HadCM3.tolist())
    df['dF'] = pd.Series(dF_HadCM3.tolist())
    df['P'] = pd.Series(P_HadCM3.tolist())
    df['Ptrans'] = pd.Series(Ptrans_HadCM3.tolist())
    df['Gamma'] = pd.Series(gamma_HadCM3.tolist())
    df['ZMT'] = pd.Series(Data_HadCM3[0][:,0].tolist())
    df['GMT'] = pd.Series(Data_HadCM3[1][:,0].tolist())
    df.to_csv('Experiments/Output/'+run+'_conf2_abs.csv')


# In[ ]:


### Run LGM_forc Optimizations

run='LGM_forc'

config_HadCM3=importer('Experiments/HadCM3/'+Config_data[run])
parallel_config = {'number_of_parameters': 6, 'number_of_cycles': 1,'number_of_parallels': 13}
variable_importer(config_HadCM3,initialZMT=False,parallel=True,parallel_config=parallel_config)
config_HadCM3,Sellers=add_sellersparameters(config_HadCM3, parameterinterpolatorstepwise,                                            'ZOEE/config/SellersParameterization.ini',4,2,True,False)
elevation=-0.0065*(np.array(Sellers[1][1])+125)

"""Import the class of your model that has to be defined in ZOEE.modules.optimization. And give it whatever 
configuration it requires"""
ZOEE_HadCM3 = ZOEE_optimization(6,parameter_labels_6,parameter_levels_6,True,elevation,'Coupled',12000,monthly=True)

#model_setup=[2,'ZMT',parameter_labels,parameter_levels,elevation,True]

"""Execture optimize to start the optimization, giving it your model imported in the step before and configuration
required to run your model"""
optimization_setup_6_LGM_an.target={'ZMT':HadCM3_ZMT[run],'GMT':HadCM3_GMT_anomaly[run]}

print("Optimization >>> {}".format(run))
F_HadCM3, dF_HadCM3, P_HadCM3, Ptrans_HadCM3, gamma_HadCM3, Data_HadCM3=optimization_setup_6_LGM_an.optimize(ZOEE_HadCM3,config_HadCM3)

df = pd.DataFrame()
df['F'] = pd.Series(F_HadCM3.tolist())
df['dF'] = pd.Series(dF_HadCM3.tolist())
df['P'] = pd.Series(P_HadCM3.tolist())
df['Ptrans'] = pd.Series(Ptrans_HadCM3.tolist())
df['Gamma'] = pd.Series(gamma_HadCM3.tolist())
df['ZMT'] = pd.Series(Data_HadCM3[0][:,0].tolist())
df['GMT'] = pd.Series(Data_HadCM3[1][:,0].tolist())
df.to_csv('Experiments/Output/'+run+'_conf2_an.csv')

### Run LGM pm optimizations

for run in ['xmzkb','xmzkc']: 

    config_HadCM3=importer('Experiments/HadCM3/'+Config_data[run])
    parallel_config = {'number_of_parameters': 3, 'number_of_cycles': 1,'number_of_parallels': 7}
    variable_importer(config_HadCM3,initialZMT=False,parallel=True,parallel_config=parallel_config)
    config_HadCM3,Sellers=add_sellersparameters(config_HadCM3, parameterinterpolatorstepwise,                                                'ZOEE/config/SellersParameterization.ini',4,2,True,False)
    elevation=-0.0065*(np.array(Sellers[1][1])+125)

    """Import the class of your model that has to be defined in ZOEE.modules.optimization. And give it whatever 
    configuration it requires"""
    ZOEE_HadCM3 = ZOEE_optimization(3,parameter_labels_3,parameter_levels_3,True,elevation,'Coupled',12000,monthly=True)

    #model_setup=[2,'ZMT',parameter_labels,parameter_levels,elevation,True]

    """Execture optimize to start the optimization, giving it your model imported in the step before and configuration
    required to run your model"""
    optimization_setup_3_LGM_an.target={'ZMT':HadCM3_ZMT_anomaly[run],'GMT':HadCM3_GMT_anomaly[run]}

    print("Optimization >>> {}".format(run))
    F_HadCM3, dF_HadCM3, P_HadCM3, Ptrans_HadCM3, gamma_HadCM3, Data_HadCM3=    optimization_setup_3_LGM_an.optimize(ZOEE_HadCM3,config_HadCM3)

    df = pd.DataFrame()
    df['F'] = pd.Series(F_HadCM3.tolist())
    df['dF'] = pd.Series(dF_HadCM3.tolist())
    df['P'] = pd.Series(P_HadCM3.tolist())
    df['Ptrans'] = pd.Series(Ptrans_HadCM3.tolist())
    df['Gamma'] = pd.Series(gamma_HadCM3.tolist())
    df['ZMT'] = pd.Series(Data_HadCM3[0][:,0].tolist())
    df['GMT'] = pd.Series(Data_HadCM3[1][:,0].tolist())
    df.to_csv('Experiments/Output/'+run+'_conf2_an.csv')


# In[ ]:




