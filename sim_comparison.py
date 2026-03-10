import numpy as np
import pandas as pd
from plotting import gg_comp

# importing our centerline data from gps
data_path = "./data/validation_data/leads_testing_1.csv"
lap_num = 8
gg_comp(data_path, lap_num)
