import os
import math
import numpy as np
import pandas as pd



df = pd.read_csv('Kamal_mstdaccelerometerfinal_test.csv')
df['pitch'] = df.apply(lambda row: math.atan(row['accelerometer_X1_left']/math.sqrt(pow(row['accelerometer_X2_left'],2)+pow(row['accelerometer_X3_left'],2))) , axis=1)
df['roll'] = df.apply(lambda row: math.atan(-row['accelerometer_X1_left']/row['accelerometer_X3_left']) , axis=1)
df['norm'] = df.apply(lambda row: math.sqrt(pow(row['accelerometer_X1_left'],2)+pow(row['accelerometer_X2_left'],2)+pow(row['accelerometer_X3_left'],2)) , axis=1)

df = df.reindex_axis(['accelerometer_X1_left','accelerometer_X2_left','accelerometer_X3_left','X1_range','X2_range','X3_range','X1_power','X2_power','X3_power','X1_1_power','X2_1_power','X3_1_power','accelerometer_X1_right','accelerometer_X2_right','accelerometer_X3_right','pitch','roll','norm_left','norm_right','posture'], axis=1)
df.to_csv('training.csv')
