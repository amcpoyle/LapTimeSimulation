
from pykml import parser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
import casadi as ca
import matplotlib.cm as cm

from track_preprocessing import kml_to_df, kml_to_array, to_cartesian, kml_to_latlon
import elevation
import rasterio
from rasterio.transform import rowcol

radius_earth = 6.371*(10**6) # earth's radius in meters

kml_file_left = "./tracks/spa left.kml"
kml_file_right = "./tracks/spa right.kml"
srtm_file = "./tracks/spa_SRTM.tif"

def get_dem(dem_path, lats, lons):
    with rasterio.open(dem_path) as src:
        elevations = []
        for lat, lon in zip(lats, lons):
            row, col = rowcol(src.transform, lon, lat)
            elev = src.read(1)[row, col]
            elevations.append(elev)
    return elevations

def kml_to_latlon_3d(file_path):

    with open(file_path, 'r') as f:
        # doc = parser.parse(f)
        root = parser.parse(f).getroot()

    coor = (root.Document.Placemark.Polygon.outerBoundaryIs.LinearRing.coordinates).text.strip()

    lats = []
    lons = []
    coor_split = coor.split(' ')

    for c in coor_split:
        # c = one triple
        c_split = c.split(",")
        lon = float(c_split[0])
        lat = float(c_split[1])
        alt = float(c_split[2])

        lats.append(lat)
        lons.append(lon)

    lats = np.array(lats)
    lons = np.array(lons)

    dlat = lats - lats[0]
    dlon = lons - lons[0]
    dist = np.sqrt(dlat**2 + dlon**2)
    turnaround = np.argmax(dist)

    lats = lats[:turnaround]
    lons = lons[:turnaround]
    return lats, lons

def smooth_elevations(elevations, window=51, polyorder=3):
    """Pre-smooth raw DEM elevations with Savitzky-Golay filter
    to remove quantization steps before spline fitting."""
    # window must be odd and > polyorder
    if window % 2 == 0:
        window += 1
    if window > len(elevations):
        window = len(elevations) // 2 * 2 + 1
    return savgol_filter(elevations, window, polyorder)

def shared_z_profile(left_cart, right_cart):
    """For each point on left, find nearest point on right (by XY),
    average their Z values, and assign the shared Z to both."""
    left_xy = left_cart[:, :2]
    right_xy = right_cart[:, :2]

    # left -> right correspondence
    tree_r = cKDTree(right_xy)
    _, idx_r = tree_r.query(left_xy)

    # right -> left correspondence
    tree_l = cKDTree(left_xy)
    _, idx_l = tree_l.query(right_xy)

    # average Z at corresponding points
    left_z_avg = (left_cart[:, 2] + right_cart[idx_r, 2]) / 2.0
    right_z_avg = (right_cart[:, 2] + left_cart[idx_l, 2]) / 2.0

    left_cart[:, 2] = left_z_avg
    right_cart[:, 2] = right_z_avg
    return left_cart, right_cart

def smooth_line(centerline, smoothing=5.0, z_smoothing=None, num_points=None):
    # set smoothing=0.0 for this function to just become a
    # function for getting curvature
    if z_smoothing is None:
        z_smoothing = smoothing

    if  num_points is None:
        num_points = len(centerline)

    if np.allclose(centerline[0], centerline[-1]):
        centerline = centerline[:-1]

    # smoothing for lat/lon data
    tck, u = splprep([centerline[:, 0], centerline[:, 1]],
                     s=smoothing, per=True)
    u_new = np.linspace(0, 1, num_points)
    x_smooth, y_smooth = splev(u_new, tck, der=0)

    # smooth z on its own because it might need more smmoothing
    tck_z, _ = splprep([centerline[:,2]], u=u, s=z_smoothing, per=True)
    z_smooth = splev(u_new, tck_z, der=0)[0]

    dx, dy = splev(u_new, tck, der=1)
    ddx, ddy = splev(u_new, tck, der=2)

    dz = splev(u_new, tck_z, der=1)[0]
    ddz = splev(u_new, tck_z, der=2)[0]

    # new curvature computation
    cross_x = dy*ddz - dz*ddy
    cross_y = dz*ddx - dx*ddz
    cross_z = dx*ddy - dy*ddx

    cross_magnitude = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
    speed = np.sqrt(dx**2 + dy**2 + dz**2)

    # only look at speed in the xy-plane instead
    speed_xy = np.sqrt(dx**2 + dy**2)
    curvature = cross_magnitude/(speed**3) # advanced dynamics hw 3 thank you for this equation
    curvature_xy = np.abs(cross_z)/(speed_xy**3)
    return np.column_stack([x_smooth, y_smooth, z_smooth]), curvature, curvature_xy

def to_cartesian_3d(origin, data):
    global radius_earth

    origin_lat = np.deg2rad(origin[0])
    origin_lon = np.deg2rad(origin[1])
    origin_elev = origin[2]

    data_new = []
    for triple in data:
        triple_lat = np.deg2rad(triple[0])
        triple_lon = np.deg2rad(triple[1])
        triple_alt = triple[2]

        x = radius_earth*(triple_lon - origin_lon)*np.cos(origin_lat)
        y = radius_earth*(triple_lat - origin_lat)
        z = triple_alt - origin_elev

        data_new.append([x, y, z])

    data_new = np.array(data_new)
    data_smoothed, curvature, curvature_xy = smooth_line(data_new, smoothing=30.0, z_smoothing=50.0, num_points=1000)
    return data_smoothed, curvature, curvature_xy

def align_boundaries_3d(left_cart, right_cart, num_points=1000):
    tree_r = cKDTree(right_cart[:, :2])
    _, idx_r = tree_r.query(left_cart[:, :2])

    centerline = (left_cart + right_cart[idx_r])/2.0 # initial centerline

    # parameterize centerline by arc length
    diffs = np.diff(centerline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s_center = np.concatenate([[0], np.cumsum(seg_lengths)])
    s_center /= s_center[-1]

    tree_c = cKDTree(centerline[:,:2])

    _, idx_lc = tree_c.query(left_cart[:,:2])
    s_left = s_center[idx_lc]
    _, idx_rc = tree_c.query(right_cart[:,:2])
    s_right = s_center[idx_rc]

    left_order = np.argsort(s_left)
    right_order = np.argsort(s_right)

    s_left_sorted = s_left[left_order]
    left_sorted = left_cart[left_order]

    s_right_sorted = s_right[right_order]
    right_sorted = right_cart[right_order]

    # remove dup station values (?)
    mask_l = np.diff(s_left_sorted, prepend=-1) > 0
    s_left_sorted = s_left_sorted[mask_l]
    left_sorted = left_sorted[mask_l]
    
    mask_r = np.diff(s_right_sorted, prepend=-1) > 0
    s_right_sorted = s_right_sorted[mask_r]
    right_sorted = right_sorted[mask_r]

    s_shared = np.linspace(0, 1, num_points, endpoint=True)

    left_resampled = np.column_stack([
        np.interp(s_shared, s_left_sorted, left_sorted[:,0]),
        np.interp(s_shared, s_left_sorted, left_sorted[:,1]),
        np.interp(s_shared, s_left_sorted, left_sorted[:,2]),
    ])
    
    right_resampled = np.column_stack([
        np.interp(s_shared, s_right_sorted, right_sorted[:,0]),
        np.interp(s_shared, s_right_sorted, right_sorted[:,1]),
        np.interp(s_shared, s_right_sorted, right_sorted[:,2]),
    ])
    return left_resampled, right_resampled

def arc_length_parameterize(lats, lons):
    distances = np.zeros(len(lats))
    for i in range(1, len(lats)):
        dx = lons[i] - lons[i-1]
        dy = lats[i] - lats[i-1]
        distances[i] = distances[i-1] + np.sqrt(dx**2 + dy**2)
    return distances

# get the centerline of the surface
def compute_centerline3d(left_data, right_data):
    centerline = []
    for left_pt, right_pt in zip(left_data, right_data):
        x_center = 0.5*(left_pt[0] + right_pt[0])
        y_center = 0.5*(left_pt[1] + right_pt[1])
        center_elev = 0.5*(left_pt[2] + right_pt[2])
        centerline.append([x_center, y_center, center_elev])
    
    if centerline[0] != centerline[-1]:
        centerline.append(centerline[0])

    return np.array(centerline)

def compute_centerline_curvature3d(centerline):
    # take first and second derivatives
    v = np.diff(centerline, axis=0)
    a = np.diff(v, axis=0)

    curvature = np.zeros(len(centerline))

    for i in range(1, len(centerline) - 1):
        # using identity 1/rho = |v x a|/|v|^3
        cross = np.cross(v[i-1], a[i-1])
        numerator = np.linalg.norm(cross)
        denominator = np.linalg.norm(v[i-1])**3

        if denom > 0:
            curvature[i] = numerator/denominator
    return curvature

def compute_centerline_slope3d(centerline):
    dz = np.diff(centerline[:,2])
    # dx, dy = np.diff(centerline[:,:2],axis=0)
    ds_horiz = np.linalg.norm(np.diff(centerline[:,:2], axis=0), axis=1)
    theta = np.arctan2(dz, ds_horiz) # slope angle
    return theta


def compute_surface(kml_file_left, kml_file_right, srtm_file):
    left_lats, left_lons = kml_to_latlon(kml_file_left)
    left_elevations = get_dem(srtm_file, left_lats, left_lons)
    left_elevations = smooth_elevations(np.array(left_elevations), window=15)

    left_data = []
    for lat, lon, elev in zip(left_lats, left_lons, left_elevations):
        left_data.append([lat, lon, elev])

    right_lats, right_lons = kml_to_latlon(kml_file_right)
    right_elevations = get_dem(srtm_file, right_lats, right_lons)
    right_elevations = smooth_elevations(np.array(right_elevations), window=15)

    right_data = []
    for lat, lon, elev in zip(right_lats, right_lons, right_elevations):
        right_data.append([lat, lon, elev])
    
    origin = left_data[0]
    
    left_cart, left_curvature, left_curvature_xy = to_cartesian_3d(origin, left_data)
    right_cart, right_curvature, right_curvature_xy = to_cartesian_3d(origin, right_data)

    # shared elevation between boundaries
    left_cart, right_cart = shared_z_profile(left_cart, right_cart)
    left_cart, right_cart = align_boundaries_3d(left_cart, right_cart, num_points=1000)

    # create a surface mesh from the boundaries
    # close the loop by appending the first point
    left_closed = np.vstack([left_cart, left_cart[0:1]])
    right_closed = np.vstack([right_cart, right_cart[0:1]])

    n = len(left_closed)
    s = np.linspace(0, 1, 20)
    t = np.arange(n)
    
    S, T = np.meshgrid(s,t)
    X = (1-S)*left_closed[T.astype(int),0] + S*right_closed[T.astype(int), 0]
    Y = (1-S)*left_closed[T.astype(int),1] + S*right_closed[T.astype(int), 1]
    Z = (1-S)*left_closed[T.astype(int),2] + S*right_closed[T.astype(int), 2]

    centerline = compute_centerline3d(left_closed, right_closed)

    return X, Y, Z, centerline

def plot_3d_centerline(X, Y, Z, centerline):
    # 3d boundary plots
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.5)
    ax.plot(centerline[:, 0], centerline[:, 1], centerline[:,2], color='red')
    
    # ax.set_zlim(-max(centerline[:,2]), max(centerline[:,2]))
    ax.set_zlim(-300, 300)
    plt.show()

def plot_laptime3d(velocity_values, X, Y, Z, centerline_coords, lap_time):
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(111, projection='3d')

    # centerline - create line segments for coloring
    points = np.array(centerline_coords).reshape(-1,1,3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = Normalize(vmin=np.min(velocity_values), vmax=np.max(velocity_values))

    lc = Line3DCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(velocity_values[:-1])
    lc.set_linewidth(5)

    ax.plot_surface(X, Y, Z, color='grey', alpha=0.2)
    ax.add_collection(lc)
    ax.autoscale()
    # ax.set_aspect("equal")
    ax.set_zlim(-300, 300)
    ax.set_title(f"Lap time = {round(lap_time, 4)}s")
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Velocity (m/s)")
    plt.show()

# bounds = [west, south, east, north] bounding box
# min lon, min lat, max lon, max lat
# bounds = [5.950147945081963, 50.42699735421149,
#         5.980228693555301, 50.44981905989641]

