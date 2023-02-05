import numpy as np
import pandas as pd

#Load data into a numpy array
data = [np.genfromtxt("qcp/archive/Sample_"+str(i)+".csv", delimiter=",") for i in range(1,5)]
data.append(np.genfromtxt("qcp/archive/Sample_5_corrected.csv", delimiter=","))
for i in range(5):
    print(np.shape(data[i]))
#Convert data into a pandas array

df = pd.DataFrame(data, columns = ['samples'])

showcase = df['samples'].iloc[2]
showcase

print("EndeGelaende")