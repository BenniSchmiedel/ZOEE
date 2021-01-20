from ZOEE.modules.configuration import importer, add_sellersparameters, parameterinterpolatorstepwise
from ZOEE.modules.variables import variable_importer
from ZOEE.modules.rk4 import rk4alg


def main():
    configuration = importer('EBM1D_Sellers_dynamic_config.ini')
    variable_importer(configuration)
    configuration, paras = add_sellersparameters(configuration, parameterinterpolatorstepwise,
                                                 'SellersParameterization.ini', 2, 0,
                                                 True, True)

    outputdata = rk4alg(configuration)
    return outputdata
