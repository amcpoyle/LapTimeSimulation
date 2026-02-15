import pandas as pd
import numpy as np

class Car:
    car_params = {'h': None, 'wheelbase': None, 'trackwidth': None, 'a': None, 'b': None,
                  'vehicleMass': None, 'Cd': None, 'Cl': None, 'CLf': None, 'CLr': None,
                  'gamma': None, 'roll_stiffness': None, 'P_max': None,
                  'rho_air': None, 'A': None, 'vehicleName': None, 'mu': None,
                  'T_max': None, 'R': None, 'gear_ratio': None, 'LLTD': None, 'aeroDistro': None, 'weightDistro': None}
    # input constants for our tire model
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        car_df = pd.read_excel(self.data_path) # parameter name, value, description, unit format
        # assign parameters from csv file
        for cp in Car.car_params.keys():
            value = car_df[car_df['ParameterName'] == cp]['Value']
            if (cp == 'rearToe') | (cp == 'frontToe'):
                value = np.deg2rad(value.iloc[0])
                Car.car_params[cp] = value
            else:
                Car.car_params[cp] = value.iloc[0]
