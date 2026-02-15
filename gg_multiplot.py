import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

def multiplot(velocity_values):
    velocity_values = [5, 15, 25, 35]
    ax_files = []
    ay_files = []
    v_files = []
    
    for v in velocity_values:
        ax_files.append(f"./data/ax_velocity_{v}.npy")
        ay_files.append(f"./data/ay_velocity_{v}.npy")
        v_files.append(f"./data/vlist_velocity_{v}.npy")
    
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
    
    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(df[df['v'] == 5]['ay'], df[df['v'] == 5]['ax'], label='5 m/s')
    ax.plot(df[df['v'] == 15]['ay'], df[df['v'] == 15]['ax'], label='15 m/s')
    ax.plot(df[df['v'] == 25]['ay'], df[df['v'] == 25]['ax'], label='25 m/s')
    ax.plot(df[df['v'] == 35]['ay'], df[df['v'] == 35]['ax'], label='35 m/s')
    ax.set_xlabel('ay/g')
    ax.set_ylabel('ax/g')
    ax.set_title("GGV Diagrams")
    ax.legend()
    fig.show()
    plt.show()
    return fig
