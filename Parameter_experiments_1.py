#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from ZOEE.modules.configuration import importer, add_sellersparameters, parameterinterpolatorstepwise
from ZOEE.modules.variables import variable_importer, Vars, Base
from ZOEE.modules.optimization import ZOEE_optimization
# from ZOEE import update_plotstyle, moving_average
from ZOEE.modules.rk4 import rk4alg

# In[48]:


Pmin = np.array([1 * 4.2e6, 150, 1.1, 0.8, 0.8, 0.8])
Pmax = np.array([200 * 4.2e6, 270, 2.8, 1.4, 1.4, 1.4])
parameter_labels = [['eqparam', 'c_ao'], ['func3', 'a'], ['func3', 'b'], ['func4', 'factor_oc'],
                    ['func4', 'factor_kwv'], ['func4', 'factor_kair']]
parameter_levels = np.array([None, None, None, None, None, None])

Parameters = []
for i in range(6):
    Parameters.append({'name': parameter_labels[i],
                       'level': parameter_levels[i],
                       'range': np.linspace(Pmin[i], Pmax[i], 100)})

# # Parameter experiments - HadCM3 config

# In[49]:


config_HadCM3 = importer('Experiments/HadCM3/Config_HadCM3.ini')
parallel_config = {'number_of_parameters': 1, 'number_of_cycles': 1, 'number_of_parallels': 100}
variable_importer(config_HadCM3, initialZMT=True, parallel=True, parallel_config=parallel_config)
config_HadCM3, Sellers = add_sellersparameters(config_HadCM3, parameterinterpolatorstepwise,
                                               'ZOEE/config/SellersParameterization.ini', 4, 2, True, True)
elevation = -0.0065 * np.array(Sellers[1][1])

# In[50]:


for i in range(6):
    print('---------- HadCM3 Parameter ' + str(i) + ' -----------')
    ZOEE = ZOEE_optimization(1, Parameters[i]['name'], Parameters[i]['level'], True, elevation)
    config_overwrite = ZOEE._overwrite_parameters(config_HadCM3, Parameters[i]['range'])

    if len(list(config_HadCM3['eqparam'].keys())) > 1:
        raise Exception('Config overwrite acts recursively. Outdated version of ZOEE.modules.optimization is called.')
    # config_addparameters=add_parameters(config_addsellers,parameter_values,parameter_labels)

    variable_importer(config_overwrite, initialZMT=True, parallel=True, parallel_config=parallel_config, control=True)
    CTRL = rk4alg(config_overwrite, progressbar=False, monthly=True)

    variable_importer(config_overwrite, initialZMT=False, parallel=True, parallel_config=parallel_config, control=False)
    Vars.T = CTRL[1][-1]
    FULL = rk4alg(config_overwrite, progressbar=False, monthly=True)

    np.savetxt('Experiments/Output/Parametertest_HadCM3_' + str(i) + '.txt',
               [Parameters[i]['range'], *np.transpose(CTRL[1][-1]), *FULL[2]],
               delimiter=',')
print('HadCM3 - Finished. Next: CESM')

# # Parameter experiments - CESM config

# In[51]:


config_CESM = importer('Experiments/CESM/Config_CESM.ini')
parallel_config = {'number_of_parameters': 1, 'number_of_cycles': 1, 'number_of_parallels': 100}
variable_importer(config_CESM, initialZMT=True, parallel=True, parallel_config=parallel_config)
config_CESM, Sellers = add_sellersparameters(config_CESM, parameterinterpolatorstepwise,
                                             'ZOEE/config/SellersParameterization.ini', 4, 2, True, True)
elevation = -0.0065 * np.array(Sellers[1][1])

# In[52]:


for i in range(6):
    print('---------- CESM Parameter ' + str(i) + ' -----------')
    ZOEE = ZOEE_optimization(1, Parameters[i]['name'], Parameters[i]['level'], True, elevation)
    config_overwrite = ZOEE._overwrite_parameters(config_CESM, Parameters[i]['range'])
    if len(list(config_HadCM3['eqparam'].keys())) > 1:
        raise Exception('Config overwrite acts recursively. Outdated version of ZOEE.modules.optimization is called.')
    # config_addparameters=add_parameters(config_addsellers,parameter_values,parameter_labels)

    variable_importer(config_overwrite, initialZMT=True, parallel=True, parallel_config=parallel_config, control=True)
    CTRL = rk4alg(config_overwrite, progressbar=False, monthly=True)

    variable_importer(config_overwrite, initialZMT=False, parallel=True, parallel_config=parallel_config, control=False)
    Vars.T = CTRL[1][-1]
    FULL = rk4alg(config_overwrite, progressbar=False, monthly=True)

    np.savetxt('Experiments/Output/Parametertest_CESM_' + str(i) + '.txt',
               [Parameters[i]['range'], *np.transpose(CTRL[1][-1]), *FULL[2]],
               delimiter=',')
print('CESM - Finished. Next: Pages')

# # Parameter experiments - Pages config

# In[53]:


config_Pages = importer('Experiments/Pages2k/Config_Pages.ini')
parallel_config = {'number_of_parameters': 1, 'number_of_cycles': 1, 'number_of_parallels': 100}
variable_importer(config_Pages, initialZMT=True, parallel=True, parallel_config=parallel_config)
config_Pages, Sellers = add_sellersparameters(config_Pages, parameterinterpolatorstepwise,
                                              'ZOEE/config/SellersParameterization.ini', 4, 2, True, True)
elevation = -0.0065 * np.array(Sellers[1][1])

# In[54]:


for i in range(6):
    print('---------- Pages Parameter ' + str(i) + ' -----------')
    ZOEE = ZOEE_optimization(1, Parameters[i]['name'], Parameters[i]['level'], True, elevation)
    config_overwrite = ZOEE._overwrite_parameters(config_Pages, Parameters[i]['range'])
    if len(list(config_HadCM3['eqparam'].keys())) > 1:
        raise Exception('Config overwrite acts recursively. Outdated version of ZOEE.modules.optimization is called.')
    # config_addparameters=add_parameters(config_addsellers,parameter_values,parameter_labels)

    variable_importer(config_overwrite, initialZMT=True, parallel=True, parallel_config=parallel_config, control=True)
    CTRL = rk4alg(config_overwrite, progressbar=False)

    variable_importer(config_overwrite, initialZMT=False, parallel=True, parallel_config=parallel_config, control=False)
    Vars.T = CTRL[1][-1]
    FULL = rk4alg(config_overwrite, progressbar=False)

    np.savetxt('Experiments/Output/Parametertest_Pages_' + str(i) + '.txt',
               [Parameters[i]['range'], *np.transpose(CTRL[1][-1]), *FULL[2]],
               delimiter=',')

print('Pages - Finished. --------------- Thanks Elisa! --------------------')

# In[ ]:
