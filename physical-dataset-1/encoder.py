# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:39:15 2023

@author: samuel
"""

#%%
# #Import Scetion
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow import random as tf_random
from tensorflow import split, concat
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#Possible losses/metrics: MeanSquaredError, MeanRelativeError, MeanSquaredLogarithmicError, MeanAbsoluteError
#from tensorflow.keras.metrics import Accuracy,  MeanSquaredError as acc, mse
#from tensorflow.keras.metrics import MeanRelativeError, MeanAbsoluteError as mre, mae
#from tensorflow.keras.losses import MeanSquaredError as loss_mse
#from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import plot_model, split_dataset
from tensorflow.keras.backend import clear_session

#%%     
#Load and split data
#np_data = np.load("qcp/physical-dataset-1/data.npy")
#df_data = pd.read_pickle("qcp/physical-dataset-1/data.pkl")
np_data = np.load("data.npy")
np_data = np.nan_to_num(np_data, nan=0.0)
df_data = pd.read_pickle("data.pkl")
powerOutput = np_data[:,-1].reshape(len(np_data),1)
inputData   = np_data[:,0:20]
x_train, x_test, y_train, y_test = train_test_split(inputData, powerOutput, test_size=0.33)

#%%
#Create a simple ANN
clear_session()
layer_1 = Dense(20, input_shape=(x_train.shape[1],), activation="linear")
layer_2 = Dense(40, activation="relu")
layer_3 = Dense(10, activation="selu")
layer_4 = Dense(1, activation="linear")
ann = Sequential([layer_1, layer_2, layer_3, layer_4])
ann.summary()

#%%
#Training
#losses = 'mean_squared_error', 'mean_absolute_error', 'mean_squared_logarithmic_error', 'mean_absolute_percentage_error'
ann.compile(optimizer = 'rmsprop', loss='mean_absolute_percentage_error', metrics=['mean_squared_error', 'mean_absolute_error','mean_squared_logarithmic_error', 'mean_absolute_percentage_error'])
ann_history = ann.fit(x_train, y_train, epochs=10, batch_size=5, validation_split=0.25)
#%%
#Plot definitions
def plot_metrics(history):
    n = len(history.history.keys())//2
    fig,axs = plt.subplots(1,n, figsize=(18,5))

    for i,[key,val] in enumerate(history.history.items()):
        axs[i%n].plot(history.history[key], lw=4, label=key.replace("_", " "))
    
    for ax in axs:
        #ax.set_yscale("log")
        ax.set_xlabel("epoch", fontsize=16)
        ax.legend(fontsize=14)
plot_metrics(ann_history)
#%%

#Evaluation
evaluation = ann.evaluate(x_test[:-1], y_test[:-1])
plot_metrics(evaluation)