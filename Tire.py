import pandas as pd
import numpy as np

class Tire:
    tire_params = {'ref_load': None, 'pCx1': None, 'pDx1': None, 'pDx2': None, 'pEx1': None,
                   'pKx1': None, 'pKx3': None, 'lambda_mux':  None, 'pCy1': None, 'pDy1': None,
                   'pDy2': None, 'pEy1': None, 'pKy1': None, 'pKy2': None, 'lambda_muy': None,
                   'mu': None}

    # input constants for our tire model
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        tire_df = pd.read_excel(self.data_path) # parameter name, value, description, unit format
        for cp in Tire.tire_params.keys():
            value = tire_df[tire_df['ParameterName'] == cp]['Value']
            Tire.tire_params[cp] = value.iloc[0]
