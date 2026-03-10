import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve, least_squares

def residuals(p, kappa, Fz, Fx_real, Fz0):
    Fx_predicted = mf_lon(kappa, Fz, p, Fz0)
    return Fx_predicted - Fx_real

def mf_lon(kappa, Fz, p, Fz0=1000):
    # p = params (e.g., pCx1, pDx1, etc)
    pCx1, pDx1, pEx1, pKx1, pKx3, pHx1 = p
    dfz = (np.abs(Fz) - Fz0)/Fz0
    Cx = pCx1
    Dx = np.abs(Fz)*pDx1  # pDx2 fixed at 0


    Ex = pEx1
    BCD = np.abs(Fz)*pKx1*np.exp(pKx3*dfz)

    Bx = BCD/(Cx*Dx)

    Shx = pHx1
    Svx = 0.0

    kappa_r = kappa + Shx
    Fx = Dx*np.sin(
        Cx*np.arctan(
            Bx*kappa_r - Ex*(Bx*kappa_r - np.arctan(Bx*kappa_r))
        )
    ) + Svx
    return Fx

lat_data = "./tire_data/cornering_run8.csv"
lon_data = "./tire_data/drivebrake_run72.csv"

lat_df = pd.read_csv(lat_data)
lon_df = pd.read_csv(lon_data)
lon_df = lon_df[np.abs(lon_df['SA']) < 0.5]

lat_df['SA'] = np.deg2rad(lat_df['SA'])
ref_load = 1000


"""
Longitudinal fit
"""

kappa_values = lon_df['SR'].values
Fz_values = lon_df['FZ'].values
Fx_values = lon_df['FX'].values

lon_params = ['pCx1', 'pDx1', 'pEx1', 'pKx1', 'pKx3', 'pHx1'] # note: lambda_mux = 1, pDx2 = 0
lon_params_init_guesses = [1.65, 2.5, -0.5, 15.0, -0.5, 0.0]

# bounds
lower = [1.0, 1.0, -2.0,   1.0, -5.0, -0.1]
upper = [2.0, 5.0,  1.0, 100.0,  5.0,  0.1]

lon_result = least_squares(
    residuals,
    lon_params_init_guesses,
    args=(kappa_values, Fz_values, Fx_values, ref_load),
    bounds=(lower, upper),
    method='trf',
    loss='soft_l1',
    max_nfev=1e5,
    verbose=2
)

lon_fitted_params = lon_result.x
for name, val in zip(lon_params, lon_fitted_params):
    print(f"{name}: {val:.6f}")


"""
Lateral fit
"""

def lat_residuals(p, alpha, Fz, Fy_real, Fz0):
    return mf_lat(alpha, Fz, p, Fz0) - Fy_real

def mf_lat(alpha, Fz, p, Fz0=1000):
    pCy1, pDy1, pEy1, pKy1, pKy2, pHy1 = p
    Cy = pCy1
    Dy = Fz * pDy1
    Ey = pEy1
    KyA = pKy1 * Fz0 * np.sin(2 * np.arctan(Fz / (pKy2 * Fz0)))
    By = KyA / (Cy * Dy)
    alpha_r = alpha + pHy1
    Fy = Dy * np.sin(Cy * np.arctan(By*alpha_r - Ey*(By*alpha_r - np.arctan(By*alpha_r))))
    return Fy

# filter: zero camber only for pure slip fit
lat_df = lat_df[np.abs(lat_df['IA']) < 0.5]

alpha_values = lat_df['SA'].values
Fz_lat_values = lat_df['FZ'].values
Fy_values = lat_df['FY'].values

lat_params = ['pCy1', 'pDy1', 'pEy1', 'pKy1', 'pKy2', 'pHy1']
lat_params_init_guesses = [1.3, 1.8, -0.5, 12.0, 2.0, 0.0]

# bounds
lat_lower = [1.0, 0.5, -3.0,  1.0, 0.5, -0.1]
lat_upper = [2.0, 5.0,  1.0, 50.0, 5.0,  0.1]

lat_result = least_squares(
    lat_residuals,
    lat_params_init_guesses,
    args=(alpha_values, Fz_lat_values, Fy_values, ref_load),
    bounds=(lat_lower, lat_upper),
    method='trf',
    loss='soft_l1',
    max_nfev=1e5,
    verbose=2
)

lat_fitted_params = lat_result.x
print("Lateral parameters:")
for name, val in zip(lat_params, lat_fitted_params):
    print(f"{name}: {val:.6f}")


