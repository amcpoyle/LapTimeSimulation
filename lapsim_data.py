import pandas as pd
import numpy as np
from Car import Car
from Tire import Tire
from gg_multiplot import multiplot
import matplotlib.pyplot as plt
from helper import getPoints, interpolate_common_pts, load_gg_data
from ggv_3d import plot_ggv
from track_preprocessing import get_centerline, plot_optimal, plot_laptime, smooth_line
from optimize_lap import optimize
from track_from_data import track_from_gps, plot_gps_laptime, smooth_gps_line, heatmap_gps
from plotting import gg_comp

velocity_values = [8, 9, 10, 12, 15, 20, 25, 30, 35, 40]
df = load_gg_data(velocity_values, 0)

# 3D GGV surface plot
df = load_gg_data(velocity_values, 0)
surf, fig = plot_ggv(df, velocity_values)

# importing our centerline data from gps
data_path = "./data/validation_data/leads_testing_1.csv"
lap_num = 8
x_data, y_data = track_from_gps(data_path, lap_num)

print("GPS track path loaded")

# get curvature data from this centerline
original_centerline = np.column_stack((x_data, y_data))
centerline, centerline_curvature = smooth_gps_line(original_centerline, smoothing=5.0, num_points=16000)

print("Centerline smoothed")

# values from our simulation
track_velocities, lap_time, ay_values, ax_values = optimize(centerline, centerline_curvature, surf, df)
print("LENGTH OF SIM VELOCITIES: ", len(track_velocities))

shared_vmin = np.min(track_velocities)
shared_vmax = np.max(track_velocities)

fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
plot_gps_laptime(track_velocities, centerline, lap_time, ax=left, vmin=shared_vmin, vmax=shared_vmax)
heatmap_gps(data_path, lap_num, ax=right, vmin=shared_vmin, vmax=shared_vmax)
left.set_title("Simulated: " + left.get_title())
right.set_title("Real: " + right.get_title())


# gg_comp(data_path, lap_num, track_velocities, ax_values, ay_values)

plt.tight_layout()
plt.show()
