import pandas as pd
import numpy as np
from Car import Car
from Tire import Tire
from gg_multiplot import multiplot
from full_ocp4_original import run_gg 
import matplotlib.pyplot as plt
from helper import getPoints, interpolate_common_pts, load_gg_data
from ggv_3d import plot_ggv
from track_preprocessing import get_centerline, plot_optimal, plot_laptime
from optimize_lap import optimize
"""
Generating an individual plot
"""
# velocities = [1, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 40, 45, 50, 55]
velocities = [15]
g = 9.807

for v in velocities:
    ggv_fig, ay_norm, ax_norm, v_vals = run_gg(v, 3*g, N=200, save=False, version=0)
    ggv_fig.show()

