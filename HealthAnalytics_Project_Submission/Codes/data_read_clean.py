from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from functools import reduce

user_name = 'Kamal'
sensor = ['accelerometer'] # accelerometer, gyroscope

path_for_data  = '/Users/sonali/Desktop/heath_analytics_project/'+user_name+'_Sessions/'
directory_list = os.listdir(path_for_data)
cleanup_nums = {"posture":     {"sitting": 0, "walking": 1, "standing": 2, "laying_down": 3}}


overall_df = None
newdf = None
cnt = 1
for individual_session in directory_list:
	if not individual_session.startswith('.') and not individual_session.startswith('_'):
		posture_file_path = os.path.join(path_for_data, individual_session,"data")
		files = os.listdir(posture_file_path)
		frames = []
		for data_file in files:
			if data_file.endswith('gz'):
				sensor_name = data_file.split('.')[-4]
				if sensor_name in sensor:
					final_path = os.path.join(posture_file_path, data_file)
					df = pd.read_csv(final_path, header=None, compression='gzip')
					nrows, ncols = df.shape
					old_cols = []
					for i in range(1, ncols-2):
						name = sensor_name+"_X"+str(i)
						old_cols+=[name]
					cols = ["timestamp"] + old_cols + ["confidence", "posture"]
					df.columns = cols

					df.drop(df.index[list(np.arange(0,1250))], inplace=True)
					df.drop(df.index[list(np.arange(-1250,0))], inplace=True)
					df["timestamp"] = df["timestamp"]/1000
					df["timestamp"] = df["timestamp"].astype(int)
					newdf_mean = df.groupby(['posture', 'timestamp'])[["accelerometer_X1", "accelerometer_X2", "accelerometer_X3"]].mean().\
					reset_index() # for every posture, for every timestamp find mean of the specified columns
					df_fft = newdf_mean.groupby(['posture', np.arange(len(newdf_mean)) // 5])[["accelerometer_X1", "accelerometer_X2", "accelerometer_X3"]]. \
					apply(lambda x: pd.Series(pow(np.abs(np.fft.fft(x))[0], 2))).reset_index()
					# Calculate the energies
					df_fft.columns=["posture","level_1","accelerometer_X1", "accelerometer_X2", "accelerometer_X3"]
					df_fft1 = newdf_mean.groupby(['posture', np.arange(len(newdf_mean)) // 5])[
						["accelerometer_X1", "accelerometer_X2", "accelerometer_X3"]]. \
						apply(lambda x: pd.Series(pow(np.abs(np.fft.fft(x))[0], 2))).reset_index()
					# Calculate the energies
					df_fft1.columns = ["posture", "level_1", "accelerometer_X1", "accelerometer_X2", "accelerometer_X3"]
					df_min = newdf_mean.groupby(['posture',np.arange(len(newdf_mean))//5])[["accelerometer_X1", "accelerometer_X2", "accelerometer_X3"]].\
					min().reset_index()

					df_max = newdf_mean.groupby(['posture', np.arange(len(newdf_mean)) // 5])[["accelerometer_X1", "accelerometer_X2", "accelerometer_X3"]]. \
					max().reset_index()
					newdf_mean = newdf_mean.groupby(['posture',np.arange(len(newdf_mean))//5])[["accelerometer_X1", "accelerometer_X2", "accelerometer_X3"]].\
					mean().reset_index() # for every posture for every 5 rows find mean of every 5 rows => 5 sec

					newdf_mean["X1_range"]=df_max["accelerometer_X1"]-df_min["accelerometer_X1"]
					newdf_mean["X2_range"] = df_max["accelerometer_X2"] - df_min["accelerometer_X2"]
					newdf_mean["X3_range"] = df_max["accelerometer_X3"] - df_min["accelerometer_X3"]
					newdf_mean["X1_power"]=df_fft["accelerometer_X1"]
					newdf_mean["X2_power"] = df_fft["accelerometer_X2"]
					newdf_mean["X3_power"] = df_fft["accelerometer_X3"]
					newdf_mean["X1_1_power"] = df_fft1["accelerometer_X1"]
					newdf_mean["X2_1_power"] = df_fft1["accelerometer_X2"]
					newdf_mean["X3_1_power"] = df_fft1["accelerometer_X3"]

					newdf_std = df.groupby(['posture', 'timestamp'])[["accelerometer_X1", "accelerometer_X2", "accelerometer_X3"]].mean().\
					reset_index() # for every posture, for every timestamp find mean of the specified columns
					newdf_std = newdf_std.groupby(['posture',np.arange(len(newdf_std))//5])[["accelerometer_X1", "accelerometer_X2", \
					"accelerometer_X3"]].std().fillna(newdf_std.groupby(['posture',np.arange(len(newdf_std))//5]).last()).reset_index() # for every posture for every 5 rows find mean of every 5 rows => 5 sec
					newdf_mean = newdf_mean.drop(['posture'], axis = 1)

					dfs = [newdf_mean, newdf_std]
					newdf = reduce(lambda left,right: left.join(right, lsuffix="_left", rsuffix="_right"), dfs)
					newdf.rename(columns={'posture_right': 'posture'}, inplace=True)

		if newdf is not None:
			if overall_df is None:
				overall_df = newdf
			else:
				overall_df = overall_df.append(newdf)
			print(cnt)
			cnt+=1
overall_df = overall_df.reset_index()
cols = list(overall_df.columns)
cols.pop(cols.index('posture'))
overall_df = overall_df[cols+['posture']]
overall_df = overall_df.drop(['level_1_left', 'level_1_right','index'], axis=1)
overall_df = overall_df[(overall_df['posture'].astype('str')=="sitting") | (overall_df['posture'].astype('str')=="walking") |
(overall_df['posture'].astype('str')=="standing") | (overall_df['posture'].astype('str')=="laying_down")]
overall_df.replace(cleanup_nums, inplace=True)
overall_df['posture'] = overall_df['posture'].astype(int)
overall_df.to_csv(user_name+"_mstd"+sensor[0]+"final.csv", index=False)

