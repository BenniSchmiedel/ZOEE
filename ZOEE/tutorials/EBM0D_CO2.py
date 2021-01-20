from ZOEE.modules.configuration import importer
from ZOEE.modules.variables import variable_importer
from ZOEE.modules.rk4 import rk4alg


def main():
    configuration = importer('EBM0D_CO2_config.ini')

    variable_importer(configuration, control=True)
    Time_Spinup, ZMT_Spinup, GMT_Spinup = rk4alg(configuration)

    variable_importer(configuration, control=False)
    Vars.T = ZMT_Spinup[-1]
    outputdata = rk4alg(configuration)
    return outputdata
