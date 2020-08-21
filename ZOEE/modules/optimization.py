"""
The ``ZOEE.modules.optimizationimization`` includes tools for optimizationimization and distribution ob parallel simulations.
``coremodule`` is an algorithm which allows parameter optimizationimization to adapt the model to either a target climatology (ZMT), a target response (GMT) or a coupled target (ZMT+GMT). It uses the gradient descent method from machine learning to decrease the deviation/cost function F (least squares) between the model output and the target.

The gradient descent method estimates better fitting parameters with

.. math::

    P_{n+1}= P_{n} - \gamma_n \cdot \nabla F_n (P_{n}),\;\; n\geq 0

where :math:`P` is a set of initial parameters :math:`\gamma` the learning rate and :math:`\nabla F` the gradient of the least squares, each of the optimizationimization step *i*, which are calculated as follows:

.. math::

    \gamma =& \\frac{|(P_{n}-P_{n-1})\cdot (dF_{n}-dF_{n-1})|}{||(dF_n-dF_{n-1})||^2}\\\\
    F=& \sum_{k=0}^{\tile{k}} (T_{data}-T_{target})^2\\\\
    dF=&\\frac{dF}{dP}

As this algorithm is applied to optimizationimize the cost function of a multivariate parameter space, all parameters are normalized to avoid unit errors and allow to set climatological boundaries (Pmin and Pmax). When the algorithm is run to optimizationimize a coupled target (ZMT and GMT) the cost functions from each target comparison are weighted with a climatology-to-response ratio :math:'0\leq r_{CR} \leq 1'. 

.. math::

    F_{total}=r_{CR}\cdot F_{Climatology}+(1-r_{CR})\cdot F_{Response}

When the algorithm is run is operates the following scheme: 1) Run the model to an equilibrium state. 2) Calculate cost-function (squared residuals). 3) Calculate the cost-function gradient. 4) Calculate learning rate :math:'\gamma'. 5) Estimate next step parameters. 6) Limit new parameters to climatologic boundary conditions. 
"""

import numpy as np


class optimization:

    def __init__(self, mode='Coupled', target=0, ZMT_response=True, GMT_response=True, num_steps=10, num_data=None,
                 gamma0=1e-8,
                 cost_function_type='LeastSquare', cost_ratio=0.5, ZMT=0, GMT=0,
                 precision=0, grid=None):

        self.num_steps = num_steps  # How many optimization steps
        self.mode = mode  # Optimization type
        self.num_data = num_data  # How many datapoints from the model (in time)
        self.target = target  # Target data
        self.gamma0 = gamma0  # Initial gamma (stepsize)
        self.cost_ratio = cost_ratio  # ratio between ZMT and GMT
        self.ZMT_initial = ZMT  # Initial ZMT
        self.GMT_initial = GMT  # Initial GMT
        self.precision = precision  # Precision to which is optimized, then stopped.
        self.cost_function_type = cost_function_type  # Cost function type (least squares)
        self.grid = grid  # Grid (1D latitude resolution)
        self.ZMT_response = ZMT_response  # GMT response/anomaly or not
        self.GMT_response = GMT_response  # GMT response/anomaly or not

        self.current_step = 0

    def _test_for_parameters(self):
        """Check for all the necessary parameters"""
        errormsg = "give_parameters was not executed, please run it to provide the necessary parameters"
        try:
            self.P_initial
        except NameError:
            errormsg
        try:
            self.P_min
        except NameError:
            errormsg
        try:
            self.P_max
        except NameError:
            errormsg
        try:
            self.P_ratio
        except NameError:
            errormsg

    def give_parameters(self, P_initial, P_min, P_max, P_ratio):

        self.num_paras = len(P_initial)  # How many parameters to be optimized
        self.parallels = self.num_paras * 2 + 1  # How many (parallel) simulations to be performend
        self.P_initial = P_initial  # Initial parameters (the ones to optimize)
        self.P_min = P_min  # Lower climatologic boundary
        self.P_max = P_max  # Upper climatologic boundary
        # self_Pnorm_initial = (P_initial - P_min) / (P_max - P_min)
        self.P_ratio = P_ratio  # Perturbation ratio (ratio to the range between the boundaries)
        self.P_pert = (P_max - P_min) * P_ratio  # Parameter perturbation, p +/- dp
        self.P_pert_trans = self.P_pert
        # self.Pnorm_pert=np.tile(P_ratio,)

    def optimize(self, model, model_config):
        """Execute to start the parameter optimization procedure for simulations with any model.
        The script executing the respective model and its specific configuration is inbound in the interface
        .run_model(model, model_config) and has to be provided in order to correctly run the procedure.
        """
        from tqdm import tnrange

        self._test_for_parameters()

        # create arrays
        F = np.zeros((self.num_steps, self.parallels))
        dF = np.zeros((self.num_steps, self.num_paras))
        P = np.zeros((self.num_steps, self.num_paras))
        Ptrans = np.zeros((self.num_steps, self.num_paras))
        gamma = np.zeros(self.num_steps)

        # create arrays for the model data depending on mode, for coupled it is split in two different ones
        if self.mode == 'GMT_Single':
            dataout = np.zeros((self.num_steps, self.parallels))
        elif self.mode == 'ZMT':
            dataout = np.zeros((self.num_steps, self.parallels, len(self.grid)))
        elif self.mode == 'GMT':
            dataout = np.zeros((self.num_steps, self.parallels, self.num_data))
        elif self.mode == 'Coupled':
            dataout_ZMT = np.zeros((self.num_steps, self.parallels, len(self.grid)))
            dataout_GMT = np.zeros((self.num_steps, self.parallels, self.num_data))
        else:
            raise Exception(
                'Optimization mode unknown, please use one of the following: ZMT, GMT, GMT_Single or Coupled')

        # The algorithm starts, loop over the number of optimizationimizationsteps "maxlength"
        for i in tnrange(self.num_steps):
            # first step: Calculate normalized parameters and their pertubation
            print('Iteration no.' + str(i))
            if i == 0:
                P[i] = self.P_initial
                Ptrans[i] = (self.P_initial - self.P_min) / (self.P_max - self.P_min)

            # ********************* Interface function for the model used to simulate******************
            # run the model to create the model data which is compared with the target. Two outputs if mode is coupled.
            data = self.run_model(model, model_config, P)
            # ********************************************************************************

            # Assign data and compare the model data with the targetdata. In coupled mode, include ratio of GMT to ZMT costfunction
            if self.mode == 'Coupled':
                dataout_ZMT[i] = data[0]
                dataout_GMT[i] = data[1]

                target_ZMT = self.target['ZMT']
                target_GMT = self.target['GMT']
                F_ZMT = self.target_comparison(dataout_ZMT[i], 'ZMT', target_ZMT)
                F_GMT = self.target_comparison(dataout_GMT[i], 'GMT', target_GMT)
                F[i] = self.cost_ratio * F_ZMT + (1 - self.cost_ratio) * F_GMT
            else:
                dataout[i] = data
                F[i] = self.target_comparison(dataout[i], self.mode, self.target)

            # Calculate the gradient of the costfunction
            dF[i] = self._get_gradient(F[i])

            # Calculate the gamma-factor (learning rate)
            if i == 0:
                gamma[i] = self.gamma0
            else:
                gamma[i] = self._get_stepsize(dF[i - 1], dF[i], Ptrans[i - 1], Ptrans[i])

            # Check if a certain precision is reached, if so, break the loop and return optimization data.
            if self._precision_check(dF[0], dF[i]):
                print('stop', i)
                P = P[:i]
                Ptrans = Ptrans[:i]
                F = F[:i]
                dF = dF[:i]
                gamma = gamma[:i]
                break

            # Calculate new parameters from gamma and costfunction gradient dF
            Ptrans_next = self._new_parameters(Ptrans[i], gamma[i], dF[i])

            # If new parameters are out of the climatological boundary, force them to the exceeded boundary (0 for minimum or 1 for maximum)
            for k in range(self.num_paras):
                if Ptrans_next[k] < 0:
                    Ptrans_next[k] = 0.
                if Ptrans_next[k] > 1:
                    Ptrans_next[k] = 1.

            # Fill arrays with the newly calculated values
            if i < self.num_steps - 1:
                Ptrans[i + 1] = Ptrans_next
                P[i + 1] = self.P_min + Ptrans_next * (self.P_max - self.P_min)
                print(F[i])
                print(gamma[i])
                print(P[i + 1])

            self.current_step += 1

        if self.mode == 'Coupled':
            dataout = [dataout_ZMT, dataout_GMT]
        return F, dF, P, Ptrans, gamma, dataout

    def target_comparison(self, data, mode, target):
        S_i = np.zeros(self.parallels)
        if self.cost_function_type == 'LeastSquare':
            if mode == 'GMT_Single':
                S_i = (np.array(data) - target) ** 2
            elif mode == 'ZMT':
                for i in range(self.parallels):
                    if len(data[i]) == len(target):
                        S_i[i] = np.nansum(((data[i] - target) * np.cos(self.grid * np.pi / 180) / np.mean(
                            np.cos(self.grid * np.pi / 180))) ** 2)
                    elif len(data[i]) == len(S_i):
                        S_i[i] = np.nansum(((data[:, i] - target) * np.cos(self.grid * np.pi / 180) / np.mean(
                            np.cos(self.grid * np.pi / 180))) ** 2)
            elif mode == 'GMT':
                for i in range(self.parallels):
                    S_i[i] = np.nansum((data[i] - target) ** 2)

        return S_i

    def _get_gradient(self, F):
        dF = np.zeros(self.num_paras)
        for k in range(self.num_paras):
            dF[k] = (F[2 * k + 2] - F[2 * k + 1]) / (2 * self.P_ratio)
        return dF

    def _get_stepsize(self, dF0, dF1, P0, P1):
        gamma = np.abs(np.dot(P1 - P0, dF1 - dF0) / np.dot(np.abs(dF1 - dF0), np.abs(dF1 - dF0)))
        return gamma

    def _precision_check(self, dF0, dF):
        dFabs = np.sqrt(np.dot(dF, dF))
        dF0abs = np.sqrt(np.dot(dF0, dF0))
        if dFabs / dF0abs <= self.precision:
            return True

    def _new_parameters(self, P, gamma, dF):
        P_next = P - gamma * dF
        return P_next

    def run_model(self, model, model_config, P):
        """Function to run the model simulation required for the optimization. The used model has to be defined as a
        seperate class 'model' which has atleast the subcfunction 'model.run()', which will start a simulation with this specific
        model. 'model.run()' is provided the following standard variables:
        - mode: The mode of optimization if required
        - ZMT: The zonal mean temperature
        - GMT: The global mean temperature
        - control: A variable to state if the simulation shall be a control simulation or a full run.

        All additional model specific parameters will have to be provided in one array model_config,
        or provided to the class ahead of executing 'optimization.optimize()' !!!
        """

        # Reshape parameters into [Set_0, Set_1-, Set_1+, Set_2-, Set_2+,...]
        # With Set_0 the unperturbed parameter set, Set_1- the set with parameter 1 negatively perturbed...
        P_config = self._reshape_parameters(P)

        # Coupled, ZMT and GMT_Single are always control
        # if self.mode == 'Coupled' or self.mode == 'ZMT' or self.mode == 'GMT_Single':
        # control = True

        # if control:
        # model = model_name(model_setup)
        # First run control to determine initial temperature profile with the new parameter set

        if self.current_step == 0 and np.shape(self.ZMT_initial) != (self.parallels, len(self.grid)):
            self.ZMT_initial = np.tile(self.ZMT_initial, (self.parallels, 1))
            if np.shape(self.ZMT_initial) != (self.parallels, len(self.grid)):
                raise Exception('ZMT_initial shape not understood')

        if self.current_step == 0 and np.shape(self.GMT_initial) != (self.parallels,):
            self.GMT_initial = np.tile(self.GMT_initial, self.parallels)
            if np.shape(self.GMT_initial) != (self.parallels,):
                raise Exception('GMT_initial shape not understood')

        data_CTRL = model.run(model_config, P_config, self.mode, self.ZMT_initial, self.GMT_initial, control=True)
        self.ZMT_initial, self.GMT_initial = data_CTRL[0][-1], data_CTRL[1][-1]

        # Check dimension
        if np.shape(data_CTRL[1][-1]) != (self.parallels,):
            raise Exception("GMT output from " + str(model) + " in CTRL should have shape (len(parallels), ) \
             but instead has" + str(np.shape(data_CTRL[1][-1])))
        if np.shape(data_CTRL[0][-1]) != (self.parallels, len(self.grid)):
            raise Exception("ZMT output from " + str(model) + " in CTRL should have shape (len(parallels) ,\
             len(grid)) but instead has" + str(np.shape(data_CTRL[0][-1])))

        # Then run full simulation with new parameter set
        # control = False
        if self.mode == 'Coupled' or self.mode == 'GMT':
            # if not control and self.current_step == 0:
            #    ZMT = np.tile(self.ZMT_initial, (self.parallels, 1))
            #    GMT = np.tile(self.GMT_initial, self.parallels)
            data_FULL = model.run(model_config, P_config, self.mode, self.ZMT_initial, self.GMT_initial, control=False)

            if np.shape(data_FULL[1]) != (self.num_data, self.parallels):
                raise Exception("GMT output from " + str(model) + " in FULL should have shape (len(parallels) ,\
                 len(datapoints)) but instead has" + str(np.shape(data_FULL[1])))
            if np.shape(data_FULL[0][-1]) != (self.parallels, len(self.grid)):
                raise Exception("ZMT output from " + str(model) + " in CTRL should have shape (len(parallels) ,\
                 len(grid)) but instead has" + str(np.shape(data_FULL[0][-1])))

        # Output data
        if self.mode == 'ZMT':
            if self.ZMT_response:
                data_out = np.transpose(np.transpose(data_CTRL[0][-1]) - np.average(data_CTRL[0][-1], weights=np.cos(
                    self.grid * np.pi / 180)))
            else:
                data_out = data_CTRL[0][-1]
        elif self.mode == 'GMT':
            if self.GMT_response:
                data_out = np.transpose(data_FULL[1] - data_FULL[1][0])
            else:
                data_out = np.transpose(data_FULL[1])
        elif self.mode == 'Coupled':
            if self.ZMT_response:
                dataZMT = np.transpose(np.transpose(data_CTRL[0][-1]) - np.average(data_CTRL[0][-1], weights=np.cos(
                    self.grid * np.pi / 180)))
            else:
                dataZMT = data_CTRL[0][-1]
            if self.GMT_response:
                dataGMT = np.transpose(data_FULL[1] - data_FULL[1][0])
            else:
                dataGMT = np.transpose(data_FULL[1])
            data_out = [dataZMT, dataGMT]
        elif self.mode == 'GMT_Single':
            data_out = data_CTRL[0][-1]

        return data_out

    def _reshape_parameters(self, P):
        P_config = np.zeros((self.num_paras, self.parallels))

        for i in range(self.num_paras):
            for k in range(self.parallels):
                P_config[i, k] = P[self.current_step][i]
            P_config[i][i * 2 + 1] = P[self.current_step][i] - self.P_pert[i]
            P_config[i][i * 2 + 2] = P[self.current_step][i] + self.P_pert[i]
        return P_config

#---------Provide here the model specific run-script to be included in the interfrace of optimization.optimize-------
class ZOEE_optimization:
    """ZOEE specific implementations to run the optimization"""

    def __init__(self, num_params, mode, labels, levels, elevation, elevation_values, monthly=True):
        self.parallel = True
        self.labels = labels
        self.levels = levels
        self.monthly = monthly
        self.num_params = num_params
        self.mode = mode
        self.elevation = elevation
        self.elevation_values = elevation_values

    def _overwrite_parameters(self, config, P_config):

        for i in range(self.num_params):
            if self.levels[i] is None:
                if self.labels[i][0][:4] == 'func':
                    config['funccomp']['funcparam'][self.labels[i][0]][self.labels[i][1]] = P_config[i]
                if self.labels[i][0] == 'eqparam':
                    config[self.labels[i][0]][self.labels[i][1]] = P_config[i]
            else:
                if type(config['funccomp']['funcparam'][self.labels[i][0]][self.labels[i][1]]) == float:
                    raise Exception('parameter no. ' + str(i) + 'not defined in 1d space')
                elif np.shape(config['funccomp']['funcparam'][self.labels[i][0]][self.labels[i][1]]) == (
                        self.levels[i],):
                    config['funccomp']['funcparam'][self.labels[i][0]][self.labels[i][1]] = \
                        np.transpose(np.tile(config['funccomp']['funcparam'][self.labels[i][0]][self.labels[i][1]],
                                             (P_config[i].size, 1)))
                if self.labels[i][0][:4] == 'func':
                    config['funccomp']['funcparam'][self.labels[i][0]][self.labels[i][1]][self.levels[i]] = P_config[i]
                if self.labels[i][0] == 'eqparam':
                    config[self.labels[i][0]][self.labels[i][1]][i] = P_config[i]
        return config

    def run(self, config, P_config, mode, ZMT, GMT, control=False):
        from .variables import variable_importer, Vars
        from .rk4 import rk4alg
        from .functions import cosd

        self.mode = mode
        parallel_config = {'number_of_parameters': self.num_params, 'number_of_cycles': 1,
                           'number_of_parallels': int(self.num_params * 2 + 1)}

        variable_importer(config, initialZMT=False, parallel=self.parallel, parallel_config=parallel_config,
                          control=control)
        config = self._overwrite_parameters(config, P_config)

        Vars.T, Vars.T_global = ZMT, GMT
        data = rk4alg(config, progressbar=True, monthly=self.monthly)
        if self.elevation:
            data_out = [data[1] + self.elevation_values,
                        data[2][1:] + np.average(self.elevation_values, weights=cosd(Vars.Lat))]
        else:
            data_out = [data[1], data[2][1:]]
        return data_out
