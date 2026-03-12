import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d, splprep, splev
from scipy.spatial import cKDTree
# import plotly.express as px
from helper import resample_motec_data

# TODO: rename this file

def compute_gps(data_path, target_lap_num, centerline, curvature, num_pts=2000):
    df = resample_motec_data(data_path, target_lap_num)
    df = df.dropna(subset=['GPS Latitude', 'GPS Longitude', 'Lap Time', 'Lap Number'], axis=0)
    subset = df[df['Lap Number'] == target_lap_num]
    print("LENGTH OF REAL DATA: ", len(subset))

    x, y = track_from_gps(data_path, target_lap_num)
    original_centerline = np.column_stack([x, y])

    centerline, curvature = smooth_gps_line(original_centerline, smoothing=5.0, num_points=num_pts)

    # ORIGINAL CODE (before arc-length alignment fix)
    # velocities = list(subset['Vehicle Speed'] / 3.6)
    # ax_values = list(subset['Vehicle Acceleration Longitudinal'])
    # ay_values = list(subset['Vehicle Acceleration Lateral'])
    # lap_time = subset['Lap Time'].iloc[0]
    # return velocities, lap_time, ay_values, ax_values

    # arc-length along the smoothed centerline (num_pts points)
    seg_lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    s_centerline = np.concatenate([[0], np.cumsum(seg_lengths)])

    # project each raw GPS point onto the nearest centerline point to get its arc-length
    tree = cKDTree(centerline)
    raw_xy = np.column_stack([x[:-1], y[:-1]])  # drop appended closing point
    _, indices = tree.query(raw_xy)
    s_raw = s_centerline[indices]

    # raw channels (data logger records speed in km/h, convert to m/s)
    v_raw = subset['Vehicle Speed'].to_numpy() / 3.6
    ax_raw = subset['Vehicle Acceleration Longitudinal'].to_numpy()
    ay_raw = subset['Vehicle Acceleration Lateral'].to_numpy()

    # sort by arc-length so interpolation is monotone
    order = np.argsort(s_raw)
    s_sorted = s_raw[order]
    v_sorted = v_raw[order]
    ax_sorted = ax_raw[order]
    ay_sorted = ay_raw[order]

    # remove duplicate arc-length values (keep first occurrence after sort)
    _, unique_idx = np.unique(s_sorted, return_index=True)
    s_sorted = s_sorted[unique_idx]
    v_sorted = v_sorted[unique_idx]
    ax_sorted = ax_sorted[unique_idx]
    ay_sorted = ay_sorted[unique_idx]

    # interpolate onto the centerline arc-length grid
    interp_v  = interp1d(s_sorted, v_sorted,  kind='linear', fill_value='extrapolate')
    interp_ax = interp1d(s_sorted, ax_sorted, kind='linear', fill_value='extrapolate')
    interp_ay = interp1d(s_sorted, ay_sorted, kind='linear', fill_value='extrapolate')

    velocities = interp_v(s_centerline)
    ax_values  = interp_ax(s_centerline)
    ay_values  = interp_ay(s_centerline)

    lap_time = subset['Lap Time'].iloc[-1]
    return velocities, lap_time, ay_values, ax_values

def track_from_gps(data_path, target_lap_num):
    df = resample_motec_data(data_path, target_lap_num)
    # df = pd.read_csv(data_path, encoding='ISO-8859-1')
    df = df.dropna(subset=['GPS Latitude', 'GPS Longitude', 'Lap Time', 'Lap Number'], axis=0)
    subset = df[df['Lap Number'] == target_lap_num]
    
    lat = subset['GPS Latitude'].to_numpy()
    lon = subset['GPS Longitude'].to_numpy()
    
    # Close the curve by appending the first point
    lat = np.append(lat, lat[0])
    lon = np.append(lon, lon[0])
    
    # plt.plot(lon, lat)
    # plt.show()
    
    # Convert GPS to cartesian coordinates (meters) using equirectangular projection
    R = 6371000  # Earth radius in meters
    lat_ref = np.radians(lat[0])
    
    x = R * np.radians(lon - lon[0]) * np.cos(lat_ref)
    y = R * np.radians(lat - lat[0])

    return x,y
    
    # Rotate 90 degrees counter-clockwise
    # x, y = -y, x.copy()
    
    # plt.plot(x, y)
    # plt.xlabel('X (m)')
    # plt.ylabel('Y (m)')
    # plt.axis('equal')
    # plt.show()

def smooth_gps_line(centerline, smoothing=5.0, num_points=None):
    # set smoothing=0.0 for this function to just become a
    # function for getting curvature
    if num_points is None:
        num_points = len(centerline)

    if np.allclose(centerline[0], centerline[-1]):
        centerline = centerline[:-1]

    # remove duplicate consecutive points - splprep will fail with dups
    dists = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    keep = np.append(dists > 0, True)
    centerline = centerline[keep]

    # fit periodic spline
    tck, u = splprep([centerline[:, 0], centerline[:, 1]], s=smoothing, per=True)
    u_new = np.linspace(0, 1, num_points)
    x_smooth, y_smooth = splev(u_new, tck, der=0)

    dx, dy = splev(u_new, tck, der=1)
    ddx, ddy = splev(u_new, tck, der=2)
    curvature = (dx*ddy - dy*ddx)/(dx**2 + dy**2)**1.5
    return np.column_stack([x_smooth, y_smooth]), curvature

def plot_gps_laptime(track_velocities, centerline, lap_time, ax=None, vmin=None, vmax=None):
    # centerline only (no track boundaries)
    points = np.array(centerline).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # normalize velocity values to 0-1 for colormap
    if vmin is None:
        vmin = np.min(track_velocities)
    if vmax is None:
        vmax = np.max(track_velocities)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # create LineCollection with colormap - what shows  up in the graph
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(np.array(track_velocities[:-1]))
    lc.set_linewidth(2)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(f"Lap time = {round(lap_time,4)}s")

    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label('Velocity (m/s)')
    return fig, ax
    # plt.show()
    
