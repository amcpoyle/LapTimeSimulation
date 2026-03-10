import pandas as pd
import numpy as np
from Car import Car
from Tire import Tire
from gg_multiplot import multiplot
import matplotlib.pyplot as plt
from helper import getPoints, interpolate_common_pts, load_gg_data
from ggv_3d import plot_ggv
from track_preprocessing import get_centerline, plot_optimal, plot_laptime
from optimize_lap import optimize3d
from track_preprocessing3d import compute_surface, compute_centerline_slope3d, smooth_line
from track_preprocessing3d import plot_3d_centerline, plot_laptime3d

kml_file_left = "./tracks/spa left.kml"
kml_file_right = "./tracks/spa right.kml"
srtm_file = "./tracks/spa_SRTM.tif"

velocity_values = [8, 9, 10, 12, 15, 20, 25, 30, 35, 40]
df = load_gg_data(velocity_values, 0)
surf, fig = plot_ggv(df, velocity_values)

X,Y, Z, centerline = compute_surface(kml_file_left, kml_file_right, srtm_file)

plot_3d_centerline(X, Y, Z, centerline)

centerline_slope = compute_centerline_slope3d(centerline)
centerline_coords, centerline_curvature, centerline_curvature_xy = smooth_line(centerline, smoothing=0.0, z_smoothing=None)


forward_copy, lap_time, max_velocities = optimize3d(centerline, centerline_curvature, centerline_curvature_xy, centerline_slope, surf, df)
print("LAP TIME", lap_time)

plot_laptime3d(forward_copy, X, Y, Z, centerline_coords, lap_time)
