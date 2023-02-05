import numpy as np
import pandas as pd

#Load data in numpy-array and pandas-df
np_data = np.genfromtxt("qcp/physical-dataset-1/archive/spg.csv", delimiter=",")
df_data = pd.read_csv("qcp/physical-dataset-1/archive/spg.csv")
#Get infos about the set
df_data.columns
df_data.shape
df_data.info
df_data.describe
#Save both datatype-sets
np.save("qcp/physical-dataset-1/data", np_data)
df_data.to_pickle("qcp/physical-dataset-1/data.pkl")