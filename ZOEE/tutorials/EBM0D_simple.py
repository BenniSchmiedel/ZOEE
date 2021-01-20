from ZOEE.modules.configuration import importer
from ZOEE.modules.variables import variable_importer
from ZOEE.modules.rk4 import rk4alg


def main():
    configuration = importer('EBM0D_simple_config.ini')

    variable_importer(configuration)
    outputdata = rk4alg(configuration)
    return outputdata
