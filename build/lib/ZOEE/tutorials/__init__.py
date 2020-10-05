import matplotlib.pyplot as plt
import numpy as np
from ZOEE.modules.variables import Vars, Base
from ZOEE.modules.functions import plotmeanstd

def plot_time_temp(outputdata):
    plt.plot(np.array(outputdata[0])/Base.stepsize_of_integration/365,outputdata[2])
    plt.xlabel('Time [years]')
    plt.ylabel('GMT [K]')
    plt.show()

def plot_lat_temp(outputdata):
    T=plotmeanstd(outputdata[1])
    plt.plot(Vars.Lat,T[0])
    plt.xlabel('Latitude [°]'); plt.ylabel('ZMT [K]')
    plt.xticks(np.linspace(-90,90,7))

    plt.show()
