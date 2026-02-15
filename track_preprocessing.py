from pykml import parser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
from scipy.interpolate import splprep, splev
import casadi as ca
import matplotlib.cm as cm

# global constants
radius_earth = 6.371*(10**6) # earth's radius in meters

kml_file_left = "./tracks/pitt race karting left.kml"
kml_file_right = "./tracks/pitt race karting right.kml"

# for terminal highlighting
class bcolors:
    FAIL = '\033[91m'
    OKGREEN = '\033[92m'


def kml_to_df(file_path, boundary_side, track_name):

    with open(file_path, 'r') as f:
        # doc = parser.parse(f)
        root = parser.parse(f).getroot()
    
    coor = (root.Document.Placemark.Polygon.outerBoundaryIs.LinearRing.coordinates).text.strip()
    
    lats = []
    lons = []
    alts = []
    coor_split = coor.split(' ')
    
    for c in coor_split:
        # c = one triple
        c_split = c.split(",")
        lon = c_split[0]
        lat = c_split[1]
        alt = c_split[2]
    
        lats.append(lat)
        lons.append(lon)
        alts.append(alt)
    
    # TODO: utilize a faster data structure for future processing
    coor_df = pd.DataFrame({'lat': lats, 'lon': lons, 'alt': alts, 'boundary_side': [boundary_side]*len(lats), 'track': [track_name]*len(lats)})

    return coor_df


def kml_to_array(file_path):

    with open(file_path, 'r') as f:
        # doc = parser.parse(f)
        root = parser.parse(f).getroot()
    
    coor = (root.Document.Placemark.Polygon.outerBoundaryIs.LinearRing.coordinates).text.strip()
    
    triples = []
    coor_split = coor.split(' ')

    for c in coor_split:
        # c = one triple
        c_split = c.split(",")
        lon = float(c_split[0])
        lat = float(c_split[1])
        alt = float(c_split[2])

        triple = [lat, lon, alt]
        triples.append(triple)

    coor_array = np.array(triples)
    return coor_array

def kml_to_latlon(file_path):

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
    return lats, lons
    

# track length calculation
"""
Optimization variables:
    optimization, cost curvature, cost track limits smoothness, cost track limits error,
    cost centerline

"""


def to_cartesian(origin, data):
    """
    converting longitude, latitude to cartesian coordinates so that our
    boundaries are able to be plotted
    """
    global radius_earth
    # origin_lat = origin[0]
    # origin_lon = origin[1]
    # origin_alt = origin[2] # should just be 0 for now

    # origin_x = radius_earth*np.cos(origin_lat)*np.cos(origin_lon)
    # origin_y = radius_earth*np.cos(origin_lat)*np.sin(origin_lon)
    origin_lat = np.deg2rad(origin[0])
    origin_lon = np.deg2rad(origin[1])


    data_new = []
    # loop through every point in the left boundary and convert
    for triple in data:
        triple_lat = np.deg2rad(triple[0])
        triple_lon = np.deg2rad(triple[1])

        # triple_x = radius_earth*np.cos(triple_lat)*np.cos(triple_lon)
        # triple_y = radius_earth*np.cos(triple_lat)*np.sin(triple_lon)
        x = radius_earth*(triple_lon - origin_lon)*np.cos(origin_lat)
        y = radius_earth*(triple_lat - origin_lat)

        data_new.append([x, y])

    # smooth our boundaries
    data_new = np.array(data_new)
    data_smoothed, curvature = smooth_line(data_new, smoothing=15.0, num_points=1000)
    return data_smoothed


def plot_boundary(data):
    # plot the boundary of the track from cartesian data
    x_vals = []
    y_vals = []
    for i in data:
        x_vals.append(i[0])
        y_vals.append(i[1])

    plt.plot(x_vals, y_vals)
    plt.show()

def plot_track(left_data, right_data):
    x_vals_left = []
    y_vals_left = []
    for i in left_data:
        x_vals_left.append(i[0])
        y_vals_left.append(i[1])
    
    x_vals_right = []
    y_vals_right = []
    for i in right_data:
        x_vals_right.append(i[0])
        y_vals_right.append(i[1])

    plt.plot(x_vals_left, y_vals_left)
    plt.plot(x_vals_right, y_vals_right)
    plt.show()

def plot_centerline(left_data, right_data, centerline_data):
    x_vals_left = []
    y_vals_left = []
    for i in left_data:
        x_vals_left.append(i[0])
        y_vals_left.append(i[1])
    
    x_vals_right = []
    y_vals_right = []
    for i in right_data:
        x_vals_right.append(i[0])
        y_vals_right.append(i[1])

    x_vals_centerline = []
    y_vals_centerline = []
    for i in centerline_data:
        x_vals_centerline.append(i[0])
        y_vals_centerline.append(i[1])

    plt.plot(x_vals_left, y_vals_left, color='black', alpha=0.6)
    plt.plot(x_vals_right, y_vals_right, color='black', alpha=0.6)
    plt.plot(x_vals_centerline, y_vals_centerline, color='red')
    plt.show()

def plot_optimal(left_data, right_data, optimal_data, show_original=False, original_line=None):
    # left boundary
    x_vals_left = []
    y_vals_left = []
    for i in left_data:
        x_vals_left.append(i[0])
        y_vals_left.append(i[1])
    
    # right boundary
    x_vals_right = []
    y_vals_right = []
    for i in right_data:
        x_vals_right.append(i[0])
        y_vals_right.append(i[1])

    # optimal centerline
    x_vals_optimal = []
    y_vals_optimal = []
    for i in optimal_data:
        x_vals_optimal.append(i[0])
        y_vals_optimal.append(i[1])

    plt.plot(x_vals_left, y_vals_left, color='black', alpha=0.6)
    plt.plot(x_vals_right, y_vals_right, color='black', alpha=0.6)
    plt.plot(x_vals_optimal, y_vals_optimal, color='green')

    if show_original:
        x_vals_original = []
        y_vals_original = []
        for i in original_line:
            x_vals_original.append(i[0])
            y_vals_original.append(i[1])

        plt.plot(x_vals_original, y_vals_original, color='red', alpha=0.6)

    plt.show()

def plot_laptime(velocity_values, left_data, right_data, centerline, lap_time):
    # left boundary
    x_vals_left = []
    y_vals_left = []
    for i in left_data:
        x_vals_left.append(i[0])
        y_vals_left.append(i[1])

    # right boundary
    x_vals_right = []
    y_vals_right = []
    for i in right_data:
        x_vals_right.append(i[0])
        y_vals_right.append(i[1])

    # centerline - create line segments for coloring
    points = np.array(centerline).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # normalize velocity values to 0-1 for colormap
    norm = Normalize(vmin=np.min(velocity_values), vmax=np.max(velocity_values))

    # create LineCollection with colormap
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(velocity_values[:-1])  # one fewer color than points (for segments)
    lc.set_linewidth(2)

    fig, ax = plt.subplots()
    ax.plot(x_vals_left, y_vals_left, color='black', alpha=0.6)
    ax.plot(x_vals_right, y_vals_right, color='black', alpha=0.6)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(f"Lap time = {round(lap_time, 4)}s")

    # add colorbar to show velocity scale
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label('Velocity (m/s)')
    # fig.text(0.5, -0.1, f"Lap time = {lap_time}s")

    plt.show()




def calc_distance(pt0, pt1):
    x0, y0 = pt0
    x1, y1 = pt1
    dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    return dist


def compute_equidistant(right_data):
    # TODO: change variable names to get rid of "right" because this is now a func
    right_distances = np.zeros(len(right_data))
    # compute arc length along the right boundary
    for i in range(len(right_data)):
        x, y = right_data[i]
        if i == 0:
            # we do not have a previous distance to reference
            right_distances[i] = 0
        else:
            last_point_x, last_point_y = right_data[i-1]
            # we have a previous distance
            current_dist = calc_distance([last_point_x, last_point_y], [x, y])
            new_dist = right_distances[i - 1] + current_dist
            right_distances[i] = new_dist

    # RESAMPLE the right boundary for equal spacing
    right_same_dist = np.linspace(0, right_distances[-1], len(right_distances))

    # assigning our original right boundary points to our new, ideal equidistant points
    x_interp = interp1d(right_distances, right_data[:,0], kind='linear')
    y_interp = interp1d(right_distances, right_data[:,1], kind='linear')

    right_x_new = x_interp(right_same_dist)
    right_y_new = y_interp(right_same_dist)

    right_data_equidist = np.column_stack([right_x_new, right_y_new])

    return right_distances, right_data_equidist



def compute_centerline(left_data, right_data):
    """
    this is a simple way to compute the centerline of the track
    note: centerline != optimal line
    """

    right_distances, right_data_equidist = compute_equidistant(right_data)
    left_distances, left_data_equidist = compute_equidistant(left_data)
    
    centerline = []

    # need to find nearest points to each other and take the centerline for those
    for left_pt in left_data_equidist:
        # calc distance to all right boundary points and take the min dist point
        distances = np.sqrt(np.sum((right_data_equidist - left_pt)**2, axis=1))
        closest = np.argmin(distances)
        right_pt = right_data_equidist[closest]

        # get the middle = centerline
        x_center = 0.5*(left_pt[0] + right_pt[0])
        y_center = 0.5*(left_pt[1] + right_pt[1])

        centerline.append([x_center, y_center])

    # centerline might not be closed...
    if centerline[0] != centerline[-1]:
        centerline.append(centerline[0])

    return right_data_equidist, left_data_equidist, np.array(centerline)

def optimize_centerline(left_data, right_data):
    """
    finding the optimal line by minimizing curve's total curvature
    can compute total curvature because this is a closed curve (e.g., global theory of curves)
    constraints = stay in the boundaries
    """
    
    right_distances, right_data_equidist = compute_equidistant(right_data)
    left_distances, left_data_equidist = compute_equidistant(left_data)

    N = len(left_data)

    opti = ca.Opti()

    alpha = opti.variable(N) # the location of our point (0 = on left boundary, 1 = on right boundary)

    # get corresponding points
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    for left_pt in left_data_equidist:
        # calc distance to all right boundary points and take the min dist point
        distances = np.sqrt(np.sum((right_data_equidist - left_pt)**2, axis=1))
        closest = np.argmin(distances)
        right_pt = right_data_equidist[closest]

        left_x.append(left_pt[0])
        left_y.append(left_pt[1])
        right_x.append(right_pt[0])
        right_y.append(right_pt[1])

    left_x = np.array(left_x)
    left_y = np.array(left_y)
    right_x = np.array(right_x)
    right_y = np.array(right_y)
    # left_x = left_data[:, 0]
    # left_y = left_data[:, 1]
    # right_x = right_data[:, 0]
    # right_y = right_data[:, 1]

    x = left_x + alpha*(right_x - left_x) # our "optimized" x position based on what loc value is
    y = left_y + alpha*(right_y - left_y) 

    total_curvature = 0 # what we are minimizing (TODO: absolute value?)
    for i in range(N):
        if i == N-1:
           forward_idx = 0 
           prev_idx = i - 1
        else:
            prev_idx = i - 1 # for 0, this = -1 which should be fine
            forward_idx = i + 1

        # compute curvature at this point
        dx = (x[forward_idx] - x[prev_idx])/2
        dy = (y[forward_idx] - y[prev_idx])/2

        ddx = x[forward_idx] - 2*x[i] + x[prev_idx]
        ddy = y[forward_idx] - 2*y[i] + y[prev_idx]

        # squaring it so we don't have to worry about negative curvature (I think that's possible...)
        curvature_squared = ((dx*ddy - dy*ddx)**2)/((dx**2 + dy**2)**3 + 1e-6)
        total_curvature += curvature_squared

    opti.minimize(total_curvature)
    opti.subject_to(opti.bounded(0, alpha, 1))

    opti.set_initial(alpha, 0.5*np.ones(N))
    opti.solver("ipopt", {
        "ipopt.print_level": 0,
        'expand': True,
        'print_time': False
    }, {
        'max_iter': 3000,
        'tol': 1e-6
    })

    try:
        sol = opti.solve()
        new_curvature = sol.value(total_curvature)
        alpha_val = sol.value(alpha)
        x_val = sol.value(x)
        y_val = sol.value(y)
        optimized_line = np.column_stack([x_val, y_val])
        print(f"{bcolors.OKGREEN} + SUCCESS: i={i} {bcolors.OKGREEN}")
        return optimized_line, new_curvature
    except:
        print(f"{bcolors.FAIL} SOLVER FAILED: i={i} {bcolors.FAIL}")
        # just return the initial centerline guess
        alpha_init = 0.5*np.ones(N)
        x_init = left_x + alpha_init*(right_x - left_x)
        y_init = left_y + alpha_init*(right_y - left_y)
        return np.column_stack([x_init, y_init]), None
             


def smooth_line(centerline, smoothing=5.0, num_points=None):
    # set smoothing=0.0 for this function to just become a
    # function for getting curvature
    if  num_points is None:
        num_points = len(centerline)

    if np.allclose(centerline[0], centerline[-1]):
        centerline = centerline[:-1]

    # fit periodic spline
    tck, u = splprep([centerline[:, 0], centerline[:, 1]], s=smoothing, per=True)
    u_new = np.linspace(0, 1, num_points)
    x_smooth, y_smooth = splev(u_new, tck, der=0)

    dx, dy = splev(u_new, tck, der=1)
    ddx, ddy = splev(u_new, tck, der=2)
    curvature = (dx*ddy - dy*ddx)/(dx**2 + dy**2)**1.5 # calculate curvature at every point
    return np.column_stack([x_smooth, y_smooth]), curvature


def get_centerline(kml_file_left, kml_file_right, optimize=True):
    # global constants
    radius_earth = 6.371*(10**6) # earth's radius in meters
    
    # kml_file_left = "./tracks/pitt race karting left.kml"
    # kml_file_right = "./tracks/pitt race karting right.kml"

    coor_left = kml_to_array(kml_file_left)
    coor_right = kml_to_array(kml_file_right)
    
    # our reference coordinate will be the beginning of our left path:
    origin = coor_left[0]
    left_cartesian = to_cartesian(origin, coor_left)
    right_cartesian = to_cartesian(origin, coor_right)
    
    right_data, left_data, centerline = compute_centerline(left_cartesian, right_cartesian)
    centerline_smooth, centerline_curvature = smooth_line(centerline, smoothing=20.0, num_points=1000)

    total_curvature = np.sum(centerline_curvature)
    
    print("TOTAL CURVATURE before optimization: ", total_curvature)
    
    optimized_centerline, opt_curvature = optimize_centerline(left_data, right_data)
    print("OPTIMIZED CURVATURE: ", opt_curvature)
   
    # smoothing = 0.0 so that smooth_line just becomes a kappa(s) function
    _, optimized_centerline_curvature = smooth_line(optimized_centerline, smoothing=20.0, num_points=1000)
    
    # plot_optimal(left_data, right_data, optimized_centerline, show_original=True, original_line=centerline_smooth)
    return left_data, right_data, centerline_smooth, centerline_curvature, optimized_centerline, optimized_centerline_curvature
