import numpy as np
import casadi as ca
from helper import getPoints
from scipy.interpolate import interp1d
from scipy.optimize import brentq

"""
NOT IPOPT optimization
1. Go point by point (at each length s) and compute: (v, (v**2)*kappa(s)). This is an upper-bound for what is possible at this point (upper bound on speed).
2. Forward pass: (accel limited) start at v0, compute ay = (v**2)*kappa(s), find the corresponding ax that matches this value (positive ax), integrate v_{i+1}^{2} = v_{i}^{2} + 2a_{x}*delta(s). This is the max speed we can reach at the next value.
3. Backward pass: (brake limited) start at v_end, compute ay and query ax min, backwards integrate v_{i-1}^{2} = v_{i}^{2} + 2a_{xmin}*delta(s), this is the min speed we can reach on the previous point.

cost = integral ds/dt
IPOPT optimization
"""

def optimize(centerline, centerline_curvature, surface, df):
    # finding the velocity upper bound for each point along our curve
    vertices = np.array(surface._vec).T
    ay_data = vertices[:,0]
    ax_data = vertices[:,1]
    v_data = vertices[:,2]

    get_pts = getPoints(df)


    unique_v = sorted(df['v'].unique())
    velocity_ay_max = {}

    for v in unique_v:
        v_data = df[df['v'] == v]
        velocity_ay_max[v] = np.max(np.abs(v_data['ay']))

    max_velocities = np.zeros(len(centerline_curvature))
    for i, kappa_i in enumerate(centerline_curvature):
        kappa_abs = abs(kappa_i)
        if kappa_abs <= 1e-9:
            max_velocities[i] = max(unique_v)
        else:
            best_v = min(unique_v)
            for v in unique_v:
                required_ay = (v**2)*kappa_abs
                if required_ay <= velocity_ay_max[v]:
                    best_v = v
            max_velocities[i] = best_v

    # Debug output
    track_length = sum(np.linalg.norm(centerline[i+1] - centerline[i]) for i in range(len(centerline)-1))
    print(f"Track length: {track_length:.1f} m")
    print(f"Curvature - min: {np.min(np.abs(centerline_curvature)):.6f}, max: {np.max(np.abs(centerline_curvature)):.6f}, mean: {np.mean(np.abs(centerline_curvature)):.6f}")
    print(f"Max velocities - min: {np.min(max_velocities):.1f}, max: {np.max(max_velocities):.1f}, mean: {np.mean(max_velocities):.1f}")

        # # get the max speed at this curvature
        # max_ay_values = np.zeros(len(unique_v))
        # # corresponding v values are just in v_vals (same order)
        # for j, v in enumerate(unique_v):
        #     max_ay = (v**2)*kappa_i
        #     max_ay_values[j] = max_ay

        # # get the idx of the max ay values
        # max_ay_idx = np.argmax(max_ay_values)
        # # get the corresponding v values
        # corr_v = unique_v[max_ay_idx]
        # max_velocities[i] = corr_v

    v_start = 5 # TODO: we don't have a gg plot for 0, could make one for less than 5 though
    forward_velocity = np.zeros(len(centerline))
    forward_velocity[0] = v_start

    # forward integrating (limited accel)
    for i in range(len(centerline) - 1):
        ds = np.linalg.norm(centerline[i+1] - centerline[i]) 
        kappa_i = centerline_curvature[i]
        # v = max_velocities[i]
        ay = (forward_velocity[i]**2)*kappa_i

        # query ggv for the corresponding ax value
        ax_max = get_pts.query(ay, forward_velocity[i], mode='accel')
        # forward integration
        v_next = forward_velocity[i]**2 + 2.0*ax_max*ds
        v_next = np.sqrt(max(v_next, 0.0))
        
        # our velocity constraint that we computed before
        forward_velocity[i+1] = min(v_next, max_velocities[i+1])

    forward_copy = forward_velocity.copy()
    forward_copy[-1] = forward_copy[0]

    # backward integrating (limited brake)
    for i in range(len(centerline) - 1, 0, -1):
        ds = np.linalg.norm(centerline[i] - centerline[i-1])
        kappa_i = centerline_curvature[i]
        ay = (forward_copy[i]**2)*kappa_i
        ax_min = get_pts.query(ay, forward_copy[i], mode='brake')
        
        v_prev = (forward_copy[i]**2) - 2.0*ax_min*ds
        v_prev = np.sqrt(max(v_prev, 0.0))

        forward_copy[i-1] = min(v_prev, max_velocities[i-1])

    # Debug output
    print(f"Final velocities - min: {np.min(forward_copy):.1f}, max: {np.max(forward_copy):.1f}, mean: {np.mean(forward_copy):.1f}")

    # compute lap time
    # T = \integral from 0 to L (over the length of the track) ds/v(s)
    lap_time = 0.0
    for i in range(len(centerline)-1):
        ds = np.linalg.norm(centerline[i+1] - centerline[i])
        lap_time += ds/max(forward_copy[i], 1e-3)

    print('LAP TIME: ', lap_time)

    return forward_copy, lap_time

def optimize3d(centerline, centerline_curvature, centerline_curvature_xy,
               centerline_slope, surface, df):
    # finding the velocity upper bound for each point along our curve
    vertices = np.array(surface._vec).T
    ay_data = vertices[:,0]
    ax_data = vertices[:,1]
    v_data = vertices[:,2]

    get_pts = getPoints(df)


    unique_v = sorted(df['v'].unique()) # issue I need to fix
    velocity_ay_max = {}

    for v in unique_v:
        v_data = df[df['v'] == v]
        velocity_ay_max[v] = np.max(np.abs(v_data['ay']))

    v_arr = np.array(unique_v)
    ay_max_arr = np.array([velocity_ay_max[v] for v in unique_v])
    ay_max_func = interp1d(v_arr, ay_max_arr, kind='linear', fill_value='extrapolate')

    v_min = min(unique_v)
    v_max = max(unique_v)

    # extrapolation for velocities beyond GGV data
    # use linear trend from last two velocity points at ay=0 to find true top speed
    v_second = unique_v[-2]
    ax_at_vmax = float(get_pts.query(0, v_max, mode='accel'))
    ax_at_v2 = float(get_pts.query(0, v_second, mode='accel'))
    ax_slope = (ax_at_vmax - ax_at_v2) / (v_max - v_second) # d(ax)/dv, should be negative

    # true top speed: where ax_max(ay=0) extrapolates to 0
    if ax_slope < 0 and ax_at_vmax > 0:
        v_top = v_max + ax_at_vmax / (-ax_slope)
    else:
        v_top = v_max
    print(f"GGV v_max: {v_max:.1f} m/s, extrapolated v_top: {v_top:.1f} m/s")

    def query_extrap(ay, v, mode='accel'):
        """Query GGV with linear extrapolation for v > v_max."""
        if v <= v_max:
            return float(get_pts.query(ay, v, mode=mode))
        # for v > v_max, query at v_max and shift ax by the extrapolated offset
        ax_base = float(get_pts.query(ay, v_max, mode=mode))
        if mode == 'accel':
            # shift accel boundary down as v increases (power/drag effect)
            ax_offset = ax_slope * (v - v_max)
            return max(ax_base + ax_offset, 0.0)
        else:
            # braking: keep v_max values (tire-limited, slight aero change is negligible)
            return ax_base

    # want to use a continuous range of best velocity
    max_velocities = np.zeros(len(centerline_curvature_xy))
    for i, kappa_i in enumerate(centerline_curvature_xy):
        kappa_abs = abs(kappa_i)
        if kappa_abs <= 1e-9:
            # max_velocities[i] = v_max # original: capped at GGV max
            max_velocities[i] = v_top
        else:
            # v^{2}*kappa = ay_max(v) => f(v) = ay_max(v) - v^{2}*kappa = 0
            f = lambda v: ay_max_func(v) - (v**2)*kappa_abs
            # if f(v_max) >= 0:  # original
            #     max_velocities[i] = v_max
            if f(v_top) >= 0:
                max_velocities[i] = v_top

            elif f(v_min) <= 0:
                max_velocities[i] = v_min
            else:
                # max_velocities[i] = brentq(f, v_min, v_max) # original
                max_velocities[i] = brentq(f, v_min, v_top)
            # old code that did not use a continuous range
            # best_v = min(unique_v)
            # for v in unique_v:
            #     required_ay = (v**2)*kappa_abs
            #     if required_ay <= velocity_ay_max[v]:
            #         best_v = v
            # max_velocities[i] = best_v

    # Debug output
    track_length = sum(np.linalg.norm(centerline[i+1] - centerline[i]) for i in range(len(centerline)-1))
    print(f"Track length: {track_length:.1f} m")
    print(f"Curvature - min: {np.min(np.abs(centerline_curvature_xy)):.6f}, max: {np.max(np.abs(centerline_curvature_xy)):.6f}, mean: {np.mean(np.abs(centerline_curvature_xy)):.6f}")
    print(f"Max velocities - min: {np.min(max_velocities):.1f}, max: {np.max(max_velocities):.1f}, mean: {np.mean(max_velocities):.1f}")

    v_start = 5 # TODO: maybe change this to close to 0?
    forward_velocity = np.zeros(len(centerline))
    forward_velocity[0] = v_start

    # forward integrating (limited accel)
    for i in range(len(centerline) - 1):
        ds = np.linalg.norm(centerline[i+1] - centerline[i]) 
        kappa_i = centerline_curvature_xy[i]
        # v = max_velocities[i]
        # TODO: ax should change with elevation changes (?)
        ay = (forward_velocity[i]**2)*kappa_i

        # query ggv for the corresponding ax value (with extrapolation)
        # ax_max = get_pts.query(ay, forward_velocity[i], mode='accel') # original
        ax_max = query_extrap(ay, forward_velocity[i], mode='accel')
        ax_eff = ax_max - 9.81*np.sin(centerline_slope[i]) # accounting for slope effect
        # forward integration
        v_next = forward_velocity[i]**2 + 2.0*ax_eff*ds
        v_next = np.sqrt(max(v_next, 0.0))
        
        # our velocity constraint that we computed before
        forward_velocity[i+1] = min(v_next, max_velocities[i+1])

    forward_copy = forward_velocity.copy()
    forward_copy[-1] = forward_copy[0]

    # backward integrating (limited brake)
    for i in range(len(centerline) - 1, 0, -1):
        ds = np.linalg.norm(centerline[i] - centerline[i-1])
        kappa_i = centerline_curvature_xy[i]
        ay = (forward_copy[i]**2)*kappa_i
        # ax_min = get_pts.query(ay, forward_copy[i], mode='brake') # original
        ax_min = query_extrap(ay, forward_copy[i], mode='brake')
        ax_eff = ax_min - 9.81*np.sin(centerline_slope[i-1])
        
        v_prev = (forward_copy[i]**2) - 2.0*ax_eff*ds
        v_prev = np.sqrt(max(v_prev, 0.0))

        forward_copy[i-1] = min(v_prev, max_velocities[i-1])

    # Debug output
    print(f"Final velocities - min: {np.min(forward_copy):.1f}, max: {np.max(forward_copy):.1f}, mean: {np.mean(forward_copy):.1f}")

    # compute lap time
    # T = \integral from 0 to L (over the length of the track) ds/v(s)
    lap_time = 0.0
    for i in range(len(centerline)-1):
        ds = np.linalg.norm(centerline[i+1] - centerline[i])
        lap_time += ds/max(forward_copy[i], 1e-3)

    print('LAP TIME: ', lap_time)

    return forward_copy, lap_time, max_velocities
