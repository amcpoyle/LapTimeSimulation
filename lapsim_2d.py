import pandas as pd
import numpy as np
from Car import Car
from Tire import Tire
from gg_multiplot import multiplot
import matplotlib.pyplot as plt
from helper import getPoints, interpolate_common_pts, load_gg_data
from ggv_3d import plot_ggv
from track_preprocessing import get_centerline, plot_optimal, plot_laptime
from optimize_lap import optimize

g = 9.807

def main():
    global g

    velocity_values = [8, 9, 10, 12, 15, 20, 25, 30, 35, 40]
    # Computing our track from left and right KML boundaries
    left_data, right_data, original_centerline, original_kappa, opt_centerline, kappa_s = get_centerline("./tracks/ev endurance 2024 left scaled.kml", "./tracks/ev endurance 2024 right scaled.kml", optimize=True) 
    df = load_gg_data(velocity_values, 0)

    # 3D GGV surface plot
    df = load_gg_data(velocity_values, 0)
    surf, fig = plot_ggv(df, velocity_values)
    # fig.show()
    # plt.show()

    # Calling optimize to get an optimized lap time on a 2D-planar track
    track_velocities, lap_time = optimize(original_centerline, original_kappa, surf, df)
    plot_laptime(track_velocities, left_data, right_data, original_centerline, lap_time)


if __name__ == '__main__':
    main()
