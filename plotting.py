import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helper import resample_motec_data

def gg_comp(data_path, target_lap_num, sim_velocities, sim_ax, sim_ay):
    df = resample_motec_data(data_path, target_lap_num)

    df = df.dropna(subset=['GPS Latitude', 'GPS Longitude', 'Lap Time', 'Lap Number'], axis=0)
    subset = df[df['Lap Number'] == target_lap_num]


    # get velocity values from subset (data logger records in km/h, convert to m/s)
    ay_values = list(subset['Vehicle Acceleration Lateral'])
    ax_values = list(subset['Vehicle Acceleration Longitudinal'])
    velocity_values = list(subset['Vehicle Speed'])


    # for ay, ax, v in zip(ay_values, ax_values, velocity_values):
    #     # for this value of v, query the max ay and ax value possible
    #     print("ax: ", ax)
    #     print("ay: ", ay)
    #     print("v: ", v/3.6)
    #     print()

    plt.scatter(ay_values, ax_values, color='green')
    plt.scatter(sim_ay, sim_ax, color='red')
    plt.show()


def plot_ax_ay():
    df = pd.read_csv(data_path, encoding='ISO-8859-1')
    df = df.dropna(subset=['GPS Latitude', 'GPS Longitude', 'Lap Time', 'Lap Number'], axis=0)
    subset = df[df['Lap Number'] == target_lap_num]


    # get velocity values from subset (data logger records in km/h, convert to m/s)
    velocities = list(subset['Vehicle Acceleration Lateral'])

    fig, ax = plot_gps_laptime(velocities, centerline, subset['Lap Time'].iloc[0], ax=ax, vmin=vmin, vmax=vmax)
    return fig, ax
