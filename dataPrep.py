# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:08:42 2023

@author: samuel
"""
# Header: imports
# -----------------------------
import numpy as np
from sklearn import preprocessing as sp
from sklearn.decomposition import PCA
# -----------------------------

# load data in numpy array
data = np.genfromtxt('data/airfoil_self_noise.dat')
y = np.reshape(data[:,-1], (len(data),1))
np.save('data/y.npy',y)
# Print stochastic features of the data
means = np.mean(data,axis=0)
stds  = np.std(data, axis=0)
maxs  = np.max(data, axis=0)
mins  = np.min(data, axis=0)
print(f'Means of all features:-----\n {means}\n')
print(f'Std-dev of all features:-----\n {stds}\n')
print(f'Maxima of all features:-----\n {maxs}\n')
print(f'Minima of all features:-----\n {mins}\n')
stochasticParamsRaw = np.array([means, stds, maxs, mins])
# -----------------------------

# Range scale data and save fitting params for inverse transform
rangeScaler = sp.MinMaxScaler()
rangeScaler.fit(data[:,:-1])
paramsRS = rangeScaler.get_params()
np.save('data/paramsRS.npy',paramsRS)
dataRS = rangeScaler.transform(data[:,:-1])
dataRS = np.concatenate((dataRS, y),axis=1)
means = np.mean(dataRS,axis=0)
stds  = np.std(dataRS, axis=0)
maxs  = np.max(dataRS, axis=0)
mins  = np.min(dataRS, axis=0)
stochasticParamsRS = np.array([means, stds, maxs, mins])

# Standard scale data and save fitting params for inverse transform
standardScaler = sp.StandardScaler()
standardScaler.fit(data[:,:-1])
paramsStd = standardScaler.get_params()
np.save('data/paramsStd.npy',paramsStd)
dataStd = standardScaler.transform(data[:,:-1])
dataStd = np.concatenate((dataStd, y),axis=1)
means = np.mean(dataStd,axis=0)
stds  = np.std(dataStd, axis=0)
maxs  = np.max(dataStd, axis=0)
mins  = np.min(dataStd, axis=0)
stochasticParamsStd = np.array([means, stds, maxs, mins])

# PCA reduced data, currently using range scale data
latent_dim = 3
pca = PCA(n_components=latent_dim,)
dataPcaRS = pca.fit_transform(dataRS[:,:-1])
dataPcaRS = np.concatenate((dataPcaRS, y),axis=1)
means = np.mean(dataPcaRS,axis=0)
stds  = np.std(dataPcaRS, axis=0)
maxs  = np.max(dataPcaRS, axis=0)
mins  = np.min(dataPcaRS, axis=0)
stochasticParamsPcaRS = np.array([means, stds, maxs, mins])
# -----------------------------

# Save all dataset
np.save('data/rawData.npy',data)
np.save('data/dataRS.npy', dataRS)
np.save('data/dataStd.npy',dataStd)
np.save('data/dataPcaRS.npy',dataPcaRS)

# Save all stochastic parameters
np.save('data/stochasticParamsRaw.npy',stochasticParamsRaw)
np.save('data/stochasticParamsRS.npy',stochasticParamsRS)
np.save('data/stochasticParamsStd.npy',stochasticParamsStd)
np.save('data/stochasticParamsPcaRS.npy',stochasticParamsPcaRS)