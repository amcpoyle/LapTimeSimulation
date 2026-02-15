import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from Car import Car
from Tire import Tire
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev

g = 9.807

class bcolors:
    FAIL = '\033[91m'
    OKGREEN = '\033[92m'
    WHITE = '\033[37M'

def smooth_with_spline(ay_values, ax_values):
    points = np.column_stack([ay_values, ax_values])
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    center = hull_points.mean(axis=0)
    angles = np.arctan2(hull_points[:,1] - center[1], hull_points[:,0] - center[0])
    sorted_idx = np.argsort(angles)
    hull_sorted = hull_points[sorted_idx]

    hull_closed = np.vstack([hull_sorted, hull_sorted[0]])

    tck, u = splprep([hull_closed[:,0], hull_closed[:,1]], s=0.1, per=True)
    u_new = np.linspace(0, 1, 200)
    ay_smooth, ax_smooth = splev(u_new, tck)
    return np.array(ay_smooth), np.array(ax_smooth)

def mf_fx_fy(tire, kappa, alpha, Fz):
    # calculate the coefs
    dfz = (Fz - tire['ref_load'])/tire['ref_load']

    Kx = Fz*tire['pKx1']*(1+tire['pKx3']*dfz)
    Ex = tire['pEx1']
    Dx = (tire['pDx1'] + tire['pDx2']*dfz)*tire['lambda_mux']
    Cx = tire['pCx1']
    Bx = Kx/(Cx*Dx + 1e-6)

    Ky = tire['ref_load']*tire['pKy1']*ca.sin(2*ca.atan(Fz/(tire['pKy2']*tire['ref_load'])))
    Ey = tire['pEy1']
    Dy = (tire['pDy1'] + tire['pDy2']*dfz)*tire['lambda_muy']
    Cy = tire['pCy1']
    By = Ky/(Cy*Dy + 1e-6)

    # magic formula
    eps = 1e-4
    sig_x = kappa/(1 + kappa)
    sig_y = alpha/(1 + kappa)
    sig = ca.sqrt((sig_x**2) + (sig_y**2) + eps)

    Fx = Fz*(sig_x/sig)*Dx*ca.sin(Cx*ca.arctan(Bx*sig - Ex*(Bx*sig - ca.arctan(Bx*sig))))
    Fy = Fz*(sig_y/sig)*Dy*ca.sin(Cy*ca.arctan(By*sig - Ey*(By*sig - ca.arctan(By*sig))))
    return Fx, Fy


def normal_loads(car, tire, ax, ay, u):
    rho_air = car['rho_air']
    CLfA = car['CLf']*car['A']
    CLrA = car['CLr']*car['A']
    vehicleMass = car['vehicleMass']
    roll_stiffness = car['roll_stiffness']
    h = car['h']
    trackwidth = car['trackwidth']
    a = car['a']
    b = car['b']
    roll_stiffness = 0.53
    global g

    FLf = 0.5*CLfA*rho_air*(u**2)
    FLr = 0.5*CLrA*rho_air*(u**2)

    Nfl = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) + vehicleMass*ay*(h/trackwidth)*roll_stiffness + 0.5*FLf
    Nfr = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*(roll_stiffness) + 0.5*FLf
    Nrl = 0.5*vehicleMass*g*(a/(a+b)) + 0.5*vehicleMass*ax*(h/(a+b)) + vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FLr
    Nrr = 0.5*vehicleMass*g*(a/(a+b)) + 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FLr

    Nmin = 10.0
    eps_smooth = 1.0  # 1 Newton smoothing tolerance
    Nfl = 0.5*(Nfl + Nmin + ca.sqrt((Nfl - Nmin)**2 + eps_smooth**2))
    Nfr = 0.5*(Nfr + Nmin + ca.sqrt((Nfr - Nmin)**2 + eps_smooth**2))
    Nrl = 0.5*(Nrl + Nmin + ca.sqrt((Nrl - Nmin)**2 + eps_smooth**2))
    Nrr = 0.5*(Nrr + Nmin + ca.sqrt((Nrr - Nmin)**2 + eps_smooth**2))

    return Nfl, Nfr, Nrl, Nrr

def qss_equations_core(alpha_val, states, car, tire, V, is_braking):
    """
    Core QSS equations - NO symbolic if statements!
    alpha_val: NUMERIC value of alpha at this grid point
    is_braking: boolean flag (True if alpha_val < 0)
    """
    global g
    vehicleMass = car['vehicleMass']
    h = car['h']
    wheelbase = car['wheelbase']
    a = car['a']
    b = car['b']
    trackwidth = car['trackwidth']
    CdA = car['Cd']*car['A']
    rho_air = car['rho_air']
    brake_ratio = car['gamma']
    accel_ratio = 1.0 # 50/50
    
    rho = states[0]
    delta = states[1]
    beta = states[2]
    kfl = states[3]
    kfr = states[4]
    krl = states[5]
    krr = states[6]

    ax = rho*g*ca.sin(alpha_val - beta)
    ay = rho*g*ca.cos(alpha_val - beta)

    u = V*ca.cos(beta)
    v = V*ca.sin(beta)

    omega = ay/V

    lambda_fl = delta - (v + omega*a)/(u + omega*trackwidth/2)
    lambda_fr = delta - (v + omega*a)/(u - omega*trackwidth/2)
    lambda_rl = -(v - omega*b)/(u + omega*trackwidth/2)
    lambda_rr = -(v - omega*b)/(u - omega*trackwidth/2)

    Nfl, Nfr, Nrl, Nrr = normal_loads(car, tire, ax, ay, u)

    fx_fl, fy_fl = mf_fx_fy(tire, kfl, lambda_fl, Nfl)
    fx_fr, fy_fr = mf_fx_fy(tire, kfr, lambda_fr, Nfr)
    fx_rl, fy_rl = mf_fx_fy(tire, krl, lambda_rl, Nrl)
    fx_rr, fy_rr = mf_fx_fy(tire, krr, lambda_rr, Nrr)

    FD = 0.5*CdA*rho_air*(u**2)

    # Equations of motion (11)-(13) - SAME for all conditions
    eqn1 = vehicleMass*ax - (fx_fl + fx_fr + fx_rl + fx_rr) + (fy_fl + fy_fr)*delta + FD
    eqn2 = vehicleMass*ay - (fy_fl + fy_fr + fy_rl + fy_rr) - (fx_fl + fx_fr)*delta
    eqn3 = (a*(fy_fl + fy_fr + (fx_fl + fx_fr)*delta) - b*(fy_rl + fy_rr) + 
            (trackwidth/2)*(fx_fl - fx_fr + fx_rl - fx_rr) - (trackwidth/2)*(fy_fl - fy_fr)*delta)

    # Drivetrain equations (15)-(16) - DIFFERENT for braking vs acceleration
    if is_braking:
        # BRAKING
        eqn4 = (fx_fl + fx_fr) - brake_ratio*(fx_rl + fx_rr)  # Brake ratio
        eqn5 = fx_fl - fx_fr  # Equal front brake forces
        eqn6 = fx_rl - fx_rr  # Equal rear brake forces
    else:
        # ACCELERATION (AWD)
        eqn4 = (fx_fl + fx_fr) - accel_ratio*(fx_rl + fx_rr)  # Torque distribution
        eqn5 = fx_fl - fx_fr  # Open diff front
        eqn6 = fx_rl - fx_rr  # Open diff rear

    Fscale = vehicleMass * g
    Mscale = Fscale * (a + b)
    g_eqn = ca.vertcat(eqn1/Fscale, eqn2/Fscale, eqn3/Mscale, eqn4/Fscale, eqn5/Fscale, eqn6/Fscale)
    
    # Return equations AND forces for power constraint
    return g_eqn, fx_fl, fx_fr, fx_rl, fx_rr, u

def solve(V, car, tire, N=181, u_max=10.0, epsilon=0.01):
    alpha_start = -np.pi/2
    alpha_end = np.pi/2
    # alpha_start = 0
    # alpha_end = 2*np.pi
    alpha_range = np.linspace(alpha_start, alpha_end, N)
    dalpha = alpha_range[1] - alpha_range[0]
    
    opti = ca.Opti()
    n_states = 7
    States = opti.variable(n_states, N)

    n_controls = 7
    Controls = opti.variable(n_controls, N)

    rho = States[0, :]
    delta = States[1, :]
    beta = States[2, :]
    kfl = States[3, :]
    kfr = States[4, :]
    krl = States[5, :]
    krr = States[6, :]

    # Cost function (maximize area with regularization)
    cost = 0
    for k in range(N):
        cost += -rho[k]**2 * dalpha
        for i in range(n_controls):
            cost += epsilon*Controls[i,k]**2 * dalpha
            
    opti.minimize(cost)

    # Differential equations (trapezoidal collocation)
    for k in range(N-1):
        for i in range(n_states):
            opti.subject_to(
                States[i, k+1] == States[i,k] + dalpha/2 * (Controls[i, k] + Controls[i, k+1])
            )

    # QSS equations at each grid point
    for k in range(N):
        states_k = States[:, k]
        alpha_k = alpha_range[k]  # NUMERIC value
        
        # Determine braking vs acceleration NUMERICALLY
        is_braking = alpha_k < 0
        
        # Get equations with correct conditional logic
        g_eq, fx_fl_k, fx_fr_k, fx_rl_k, fx_rr_k, u_k = qss_equations_core(
            alpha_k, states_k, car, tire, V, is_braking
        )
        
        # Apply ALL 6 QSS equations as EQUALITY constraints
        opti.subject_to(g_eq == 0)
        
        # Power constraint - SEPARATE from drivetrain equations
        # Only active during acceleration
        if not is_braking and alpha_k > 0.01:  # Small threshold
            P_max = car['P_max']
            # AWD: all 4 wheels driven
            opti.subject_to((fx_fl_k + fx_fr_k + fx_rl_k + fx_rr_k) * u_k <= P_max * 1000)

    # Bounds on controls
    for k in range(N):
        opti.subject_to(opti.bounded(-u_max, Controls[:, k], u_max))

    # Bounds on states
    for k in range(N):
        opti.subject_to(rho[k] >= 0)
        opti.subject_to(rho[k] <= 3)
        opti.subject_to(opti.bounded(-0.5, delta[k], 0.5))
        opti.subject_to(opti.bounded(-0.3, beta[k], 0.3))
        opti.subject_to(opti.bounded(-0.3, kfl[k], 0.3))
        opti.subject_to(opti.bounded(-0.3, kfr[k], 0.3))
        opti.subject_to(opti.bounded(-0.3, krl[k], 0.3))
        opti.subject_to(opti.bounded(-0.3, krr[k], 0.3))

    # Boundary conditions
    opti.subject_to(delta[0] == 0)
    opti.subject_to(beta[0] == 0)
    opti.subject_to(delta[-1] == 0)
    opti.subject_to(beta[-1] == 0)

    # Initial guess - V-dependent, physically motivated
    States_guess = np.zeros((n_states, N))

    # Estimate capabilities at this speed
    F_drag = 0.5 * car['Cd'] * car['A'] * car['rho_air'] * V**2
    F_downforce = 0.5 * (car['CLf'] + car['CLr']) * car['A'] * car['rho_air'] * V**2
    total_weight = car['vehicleMass'] * g + F_downforce
    mu_est = 1.5  # rough tire mu
    rho_lat_est = mu_est * total_weight / (car['vehicleMass'] * g)
    F_power = car['P_max'] * 1000 / max(V, 1.0)
    rho_accel_est = max((F_power - F_drag) / (car['vehicleMass'] * g), 0.2)
    rho_brake_est = mu_est * total_weight / (car['vehicleMass'] * g)

    # rho profile: asymmetric if power-limited
    for k in range(N):
        sa = np.sin(alpha_range[k])
        ca_val = np.cos(alpha_range[k])
        if sa >= 0:  # acceleration side
            rho_long = rho_accel_est
        else:  # braking side
            rho_long = rho_brake_est
        States_guess[0, k] = np.sqrt((rho_long * sa)**2 + (rho_lat_est * ca_val)**2) * 0.5

    # Steering angle: scales as ~L*ay/V^2 (bicycle model), zero at endpoints
    wheelbase = car['a'] + car['b']
    delta_scale = min(0.05, wheelbase * g / V**2)  # smaller at higher V
    States_guess[1, :] = delta_scale * np.cos(alpha_range)  # delta
    # Sideslip: small, peaks near alpha=0
    States_guess[2, :] = 0.005 * np.cos(alpha_range)  # beta
    # Slip ratios: scale with expected forces
    kappa_scale = min(0.02, 0.02 * 15.0 / max(V, 1.0))
    kappa_guess = kappa_scale * np.sin(alpha_range)
    States_guess[3, :] = kappa_guess  # kfl
    States_guess[4, :] = kappa_guess  # kfr
    States_guess[5, :] = kappa_guess  # krl
    States_guess[6, :] = kappa_guess  # krr

    Controls_guess = np.zeros((n_controls, N))

    opti.set_initial(States, States_guess)
    opti.set_initial(Controls, Controls_guess)

    # Solver options
    opts = {
        'ipopt.print_level': 5,
        'ipopt.max_iter': 3000,
        'ipopt.tol': 1e-6,
        'ipopt.acceptable_tol': 1e-4,
        'ipopt.acceptable_iter': 15,
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.mu_init': 1e-1,
        'ipopt.nlp_scaling_method': 'gradient-based',
        'print_time': True
    }
    opti.solver('ipopt', opts)

    try:
        sol = opti.solve()
        print(bcolors.OKGREEN + "SUCCESS" + bcolors.OKGREEN)

        States_opt = sol.value(States)
        Controls_opt = sol.value(Controls)
        cost_opt = sol.value(cost)

        print(f"\nOptimal cost: {cost_opt:.6f}")
        print(f"Max rho: {States_opt[0,:].max():.3f} g")
        print(f"Min rho: {States_opt[0,:].min():.3f} g")
        
        # Calculate actual accelerations
        rho_opt = States_opt[0, :]
        beta_opt = States_opt[2, :]
        ax_vals = rho_opt * g * np.sin(alpha_range - beta_opt)
        ay_vals = rho_opt * g * np.cos(alpha_range - beta_opt)
        print(f"Max longitudinal accel: {ax_vals.max()/g:.3f} g")
        print(f"Max braking: {-ax_vals.min()/g:.3f} g")
        print(f"Max lateral accel: {ay_vals.max()/g:.3f} g")
        
    except Exception as e:
        print(bcolors.FAIL + f"FAILED: {str(e)}" + bcolors.FAIL)
        States_opt = opti.debug.value(States)
        Controls_opt = opti.debug.value(Controls)
        cost_opt = opti.debug.value(cost)
        print(f"Debug - Max rho: {States_opt[0,:].max():.3f} g")

    solution = {
        'States': States_opt,
        'Controls': Controls_opt,
        'alpha': alpha_range,
        'cost': cost_opt,
        'V': V,
        'rho': States_opt[0, :],
        'delta': States_opt[1, :],
        'beta': States_opt[2, :],
        'kfl': States_opt[3, :],
        'kfr': States_opt[4, :],
        'krl': States_opt[5, :],
        'krr': States_opt[6, :]
    }
    return solution

def plot_results(solution, save, version):
    alpha = solution['alpha']
    rho = solution['rho']
    delta = solution['delta']
    beta = solution['beta']
    V = solution['V']
    print(V)


    global g
    
    ax_hat = rho*g*np.sin(alpha - beta)
    ay_hat = rho*g*np.cos(alpha - beta)

    # ay_norm = ay_hat/g
    # ax_norm = ax_hat/g
    ay_norm = ay_hat
    ax_norm = ax_hat

    # mirror the results (reverse order so the loop is continuous)
    ax_combined = np.concatenate((ax_norm, ax_norm[::-1], [ax_norm[0]]))
    ay_combined = np.concatenate((ay_norm, -ay_norm[::-1], [ay_norm[0]]))

    if save == True:
        if version is None:
            file_name_base = "velocity_{}".format(str(V))
            ay_file = "./data/ay_" + file_name_base + ".npy"
            ax_file = "./data/ax_" + file_name_base + ".npy"
            velocity_file = "./data/vlist_" + file_name_base + ".npy"
        else:
            file_name_base = "velocity_{}_version{}".format(str(V), str(version))
            ay_file = "./data/ay_" + file_name_base + ".npy"
            ax_file = "./data/ax_" + file_name_base + ".npy"
            velocity_file = "./data/vlist_" + file_name_base + ".npy"

        with open(ay_file, 'wb') as f:
            np.save(f, ay_combined)
        with open(ax_file, 'wb') as f:
            np.save(f, ax_combined)
        with open(velocity_file, 'wb') as f:
            np.save(f, np.array([V]*len(ax_combined)))
        
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Main GG diagram
    ax.plot(ay_combined, ax_combined, linewidth=2.5, color='#2E86AB', label='AWD Solution')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("ay/g")
    ax.set_ylabel("ax/g")
    ax.set_title(f"GG Diagram at V={V} m/s")
    ax.set_aspect('equal')
    ax.legend()
    
    
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    velocity_values = [45]
    car = Car("./VehicleParameters.xlsx")
    car.load_data()
    tire = Tire("./TireParameters.xlsx")
    tire.load_data()

    car_params = car.car_params
    tire_params = tire.tire_params
    
    # Set AWD torque split
    if 'gamma_accel' not in car_params:
        car_params['gamma_accel'] = 1.0  # 50/50 AWD
        # print("Using 50/50 AWD torque split")

    for V in velocity_values:
        print(bcolors.WHITE + f"Computing AWD GG Diagram at V = {V} m/s" + bcolors.WHITE)

        solution = solve(V=V, car=car_params, tire=tire_params)

        fig = plot_results(solution, True, 0)
        plt.savefig('gg_diagram_corrected.png', dpi=150, bbox_inches='tight')
        plt.show()
