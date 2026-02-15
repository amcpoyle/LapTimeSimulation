"""
File containing helper classes and functions that are
used for lap time simulation
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator
from scipy.spatial import cKDTree

def load_gg_data(velocity_values, version):
    ax_files = []
    ay_files = []
    v_files = []
    
    for v in velocity_values:
        ax_files.append(f"./data/ax_velocity_{v}_version{version}.npy")
        ay_files.append(f"./data/ay_velocity_{v}_version{version}.npy")
        v_files.append(f"./data/vlist_velocity_{v}_version{version}.npy")
    
    ax_vals = [] 
    ay_vals = [] 
    v_vals = [] 
    for ax_file, ay_file, v_file in zip(ax_files, ay_files, v_files):
        ax_data = list(np.load(ax_file))
        ay_data = list(np.load(ay_file))
        v_data = list(np.load(v_file))
    
        ax_vals = ax_vals + ax_data
        ay_vals = ay_vals + ay_data
        v_vals = v_vals + v_data
    
    df = pd.DataFrame({'ax': ax_vals, 'ay': ay_vals, 'v': v_vals})
    return df

def interpolate_common_pts(data, velocities, n_points=150):
    # we have generated 2d gg diagrams for discrete values of v
    # need to interpolate between these discrete planar diagrams to get
    # a 3d gg surface plot
    interpolated_data = {}
    for v in velocities:
        ax_original = data[data['v'] == v]['ax']
        ay_original = data[data['v'] == v]['ay']

        t_original = np.linspace(0, 1, len(ax_original))
        t_new = np.linspace(0, 1, n_points)
        f_ax = interp1d(t_original, ax_original, kind='cubic', fill_value='extrapolate')
        f_ay = interp1d(t_original, ay_original, kind='cubic', fill_value='extrapolate')

        ax_new = f_ax(t_new)
        ay_new = f_ay(t_new)

        interpolated_data[v] = {'ax': ax_new, 'ay': ay_new}
    return interpolated_data

class getPoints:
    # class that can take in (ay, v) and return ax from the 3d gg surface
    def __init__(self, df):
        accel_points = [] # (ay, v) pairs
        accel_values = [] # corresponding ax values

        brake_points = []
        brake_values = []

        velocities = list(df['v'].unique())

        for v in velocities:
            ax_data = df[df['v'] == v]['ax']
            ay_data = df[df['v'] == v]['ay']

            for ax, ay in zip(ax_data, ay_data):
                if ax >= 0:
                    accel_points.append([ay, v])
                    accel_values.append(ax)
                else:
                    brake_points.append([ay, v])
                    brake_values.append(ax)

        self.accel_points = np.array(accel_points)
        self.accel_values = np.array(accel_values)
        self.brake_points = np.array(brake_points)
        self.brake_values = np.array(brake_values)

        self.accel_interpolator = LinearNDInterpolator(self.accel_points, self.accel_values)
        self.brake_interpolator = LinearNDInterpolator(self.brake_points, self.brake_values)

        self.accel_tree = cKDTree(self.accel_points)
        self.brake_tree = cKDTree(self.brake_points)

        self.all_points = np.vstack([self.accel_points, self.brake_points])
        self.ay_min, self.ay_max = self.all_points[:, 0].min(), self.all_points[:, 0].max()
        self.v_min, self.v_max = self.all_points[:, 1].min(), self.all_points[:, 1].max()

    def query(self, ay, v, mode='accel', method='linear'):
        # get ax value for (ay, v) pair
        if mode == 'accel':
            interpolator = self.accel_interpolator
            tree = self.accel_tree
            values = self.accel_values
        elif mode == 'brake':
            interpolator = self.brake_interpolator
            tree = self.brake_tree
            values = self.brake_values
        else:
            raise ValueError("mode must be 'accel' or 'brake'")

        if method == 'linear':
            result = interpolator(ay, v)

            if np.isnan(result).any():
                if np.isscalar(ay):
                    _, idx = tree.query([ay, v])
                    result = values[idx]
                else:
                    points_query = np.column_stack([ay, v])
                    _, idx = tree.query(points_query)
                    result = values[idx]

            return result
        elif method == 'nearest':
            if np.isscalar(ay):
                _, idx = tree.query([ay, v])
                return values[idx]
            else:
                points_query = np.column_stack([ay, v])
                _, idx = tree.query(points_query)
                return values[idx]

    def get_bounds(self):
        return {'ay_min': self.ay_min, 'ay_max': self.ay_max,
                'v_min': self.v_min, 'v_max': self.v_max}

    def in_bounds(self, ay, v):
        return (self.ay_min <= ay <= self.ay_max and
                self.v_min <= v <= self.v_max)
