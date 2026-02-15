import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import pandas as pd
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator
from scipy.spatial import cKDTree
from helper import getPoints, interpolate_common_pts, load_gg_data


def plot_ggv(df, velocities):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Use a common angular parameterization for all velocities
    n_angles = 200
    common_angles = np.linspace(-np.pi, np.pi, n_angles)

    AY = np.zeros((len(velocities), n_angles))
    AX = np.zeros((len(velocities), n_angles))
    V = np.zeros((len(velocities), n_angles))

    for i, v in enumerate(velocities):
        ay_data = df[df['v'] == v]['ay'].values
        ax_data = df[df['v'] == v]['ax'].values

        # Compute angles from centroid
        center_ay = np.mean(ay_data)
        center_ax = np.mean(ax_data)
        angles = np.arctan2(ax_data - center_ax, ay_data - center_ay)

        # Sort by angle
        sort_idx = np.argsort(angles)
        angles_sorted = angles[sort_idx]
        ay_sorted = ay_data[sort_idx]
        ax_sorted = ax_data[sort_idx]

        # Interpolate to common angles (wrap around for closed curve)
        angles_extended = np.concatenate([angles_sorted - 2*np.pi, angles_sorted, angles_sorted + 2*np.pi])
        ay_extended = np.concatenate([ay_sorted, ay_sorted, ay_sorted])
        ax_extended = np.concatenate([ax_sorted, ax_sorted, ax_sorted])

        f_ay = interp1d(angles_extended, ay_extended, kind='linear')
        f_ax = interp1d(angles_extended, ax_extended, kind='linear')

        AY[i, :] = f_ay(common_angles)
        AX[i, :] = f_ax(common_angles)
        V[i, :] = v

    surf = ax.plot_surface(AY, AX, V, cmap='viridis', alpha=0.8, edgecolor='none')

    for i, v in enumerate(velocities):
        ax.plot(AY[i, :], AX[i, :], V[i, :],
                'r-', linewidth=1.5, alpha=1)

    ax.set_xlabel("ay [g]")
    ax.set_ylabel("ax [g]")
    ax.set_zlabel("Velocity [m/s]")
    ax.set_title("GGV Diagram")

    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()

    return surf, fig

