import pandas as pd
import numpy as np
from Car import Car
from Tire import Tire
from gg_multiplot import multiplot
import matplotlib.pyplot as plt
from helper import getPoints, interpolate_common_pts, load_gg_data, resample_motec_data
from ggv_3d import plot_ggv
from track_preprocessing import get_centerline, plot_optimal, plot_laptime, smooth_line
from optimize_lap import optimize
from track_from_data import track_from_gps, plot_gps_laptime, smooth_gps_line, compute_gps
from plotting import gg_comp

velocity_values = [8, 9, 10, 12, 15, 20, 25, 30, 35, 40]

# 3D GGV surface plot
df = load_gg_data(velocity_values, 0) # ggv df
surf, fig = plot_ggv(df, velocity_values)

# importing our centerline data from gps
data_path = "./data/validation_data/leads_testing_1.csv"
lap_num = 8
x_data, y_data = track_from_gps(data_path, lap_num)
num_pts = 2000

print("GPS track path loaded")

# get curvature data from this centerline
original_centerline = np.column_stack((x_data, y_data))
centerline, centerline_curvature = smooth_gps_line(original_centerline, smoothing=5.0, num_points=num_pts)

# get rid of curvature outliers to ensure that there is nothing crazy from gps noise
# there were curvature spikes so the simulation had to brake/slow down a ton even though
# in real life the track curvature was generally smooth/continuous
kappa_threshold = np.percentile(np.abs(centerline_curvature), 99)
centerline_curvature = np.clip(centerline_curvature, -kappa_threshold, kappa_threshold)

print("Centerline smoothed")

# so at this point we have the centerline and centerline curvature from our real data

# values from our simulation
track_velocities, lap_time, ay_values, ax_values = optimize(centerline, centerline_curvature, surf, df)

shared_vmin = np.min(track_velocities)
shared_vmax = np.max(track_velocities)

# values from real data
real_velocities, real_lap_time, real_ay_values, real_ax_values = compute_gps(data_path, lap_num, centerline, centerline_curvature, num_pts=num_pts)

shared_vmin_real = np.min(real_velocities)
shared_vmax_real = np.max(real_velocities)

shared_vmin = min(shared_vmin, shared_vmin_real)
shared_vmax = max(shared_vmax, shared_vmax_real)

fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))

# plotting our simulation results
plot_gps_laptime(track_velocities, centerline, lap_time, ax=left, vmin=shared_vmin, vmax=shared_vmax)

# plotting our real results
plot_gps_laptime(real_velocities, centerline, real_lap_time, ax=right, vmin=shared_vmin, vmax=shared_vmax)

left.set_title("Simulated: " + left.get_title())
right.set_title("Real: " + right.get_title())


# gg_comp(data_path, lap_num, track_velocities, ax_values, ay_values)

s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))])
plt.figure()
plt.plot(s, track_velocities, label='Simulated')
plt.plot(s, real_velocities, label='Real')
plt.xlabel("Arc length (m)")
plt.ylabel("Velocity (m/s)")
plt.legend()

plt.tight_layout()
plt.show()
"""
DEBUGGING: is there ever a time where the real life ay or ax at a certain velocity surpasses
the ay or ax that the GGV says is possible at that velocity?
"""
# data_df = resample_motec_data(data_path, lap_num) 
# 
# get_pts = getPoints(df) 
# for v, ax, ay in zip(real_velocities, real_ax_values, real_ay_values):
#     ggv_ay_max = get_pts.ay_max(v)
#     ggv_ax_accel = get_pts.query(ggv_ay_max, v, mode='accel')
#     ggv_ax_brake = get_pts.query(ggv_ay_max, v, mode='brake')
# 
#     if abs(ggv_ay_max) < abs(ay):
#             print(f"ERROR: real ay = {abs(ay)}, min ax ggv = {abs(ggv_ay_max)}")
# 
#     if ax < 0:
#         # we are braking
#         if ax < ggv_ax_brake:
#             # problem
#             print(f"ERROR: real ax = {ax}, min ax ggv = {ggv_ax_brake}")
#     else:
#         # we are accelerating
#         if ax > ggv_ax_accel:
#             print(f"ERROR: real ax = {ax}, min ax ggv = {ggv_ax_accel}")
#    
# print("---------------------------------------")


# for i, (v_real, v_sim, kappa) in enumerate(zip(real_velocities, track_velocities, centerline_curvature)):
#     if v_real > v_sim*1.02:
#         required_ay = (v_real**2)*abs(kappa)
#         ggv_ay_limit = get_pts.ay_max(v_real)
#         print(f"Point {i}: v_real={v_real:.1f}, v_sim={v_sim:.1f}, "
#               f"kappa={abs(kappa):.4f}, "
#               f"required_ay={required_ay:.2f}, ggv_ay_limit={ggv_ay_limit:.2f}, "
#               f"{'GGV TOO LOW' if required_ay > ggv_ay_limit else 'CURVATURE TOO HIGH'}")


# GGV debugging
# get_pts = getPoints(df)
# print("---------------------")
# # compare ggv limit against measured lateral accel (not against curvature)
# for v, real_ay in zip(real_velocities, real_ay_values):
#     ggv_limit = get_pts.ay_max(v)
#     if abs(real_ay) > ggv_limit:
#         print(f"v={v:.1f}, measured ay={abs(real_ay):.2f}, GGV limit={ggv_limit:.2f}, excess={abs(real_ay) - ggv_limit: .2f}")
