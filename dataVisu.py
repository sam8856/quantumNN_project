# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:09:43 2023

@author: samuel
"""
# Header: imports
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
# -----------------------------

# Load data
rawData = np.load('data/rawData.npy')

dataRS = np.load('data/dataRS.npy')
dataStd = np.load('data/dataStd.npy')
# -----------------------------

# Plot data
all_data = [dataRS[:,i] for i in range(6)]
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))


# plot violin plot
axs[0].violinplot(all_data,
                  showmeans=False,
                  showmedians=True)
axs[0].set_title(r'Violin plot')

# plot box plot
axs[1].boxplot(all_data)
axs[1].set_title(r'Box plot')

# adding horizontal grid lines
for ax in axs:
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(all_data))],
                  labels=[r'Frequency', r'Angle of attack', r'Chord length',
                          r'Free-stream velocity', r'Suctions Side disp.', 
                          r'Sound pressure level'],
                  rotation = 25)
    ax.set_xlabel(r'Features')
    ax.set_ylabel(r'Range scaled values')

plt.show()
fig.savefig('Boxplot.png')

n_bins = 25
fig2, axs = plt.subplots(2,3, figsize=(15,10))
axs[0,0].hist(rawData[:,0], bins=n_bins, label=r'Frequency')
plt.xlabel('feature range')
plt.ylabel('samples in bin')
plt.legend()
axs[0,1].hist(rawData[:,1], bins=n_bins, label=r'Angle of attack')
plt.xlabel('feature range')
plt.ylabel('samples in bin')
plt.legend()
axs[0,2].hist(rawData[:,2], bins=n_bins, label=r'Chord length')
plt.xlabel('feature range')
plt.ylabel('samples in bin')
plt.legend()
axs[1,0].hist(rawData[:,3], bins=n_bins, label=r'Free-stream velocity')
plt.xlabel('feature range')
plt.ylabel('samples in bin')
plt.legend()
axs[1,1].hist(rawData[:,4], bins=n_bins, label=r'Suctions Side disp.')
plt.xlabel('feature range')
plt.ylabel('samples in bin')
plt.legend()
axs[1,2].hist(rawData[:,5], bins=n_bins, label=r'Sound pressure level')
plt.xlabel('feature range')
plt.ylabel('samples in bin')
plt.legend()
fig2.savefig('Histogram.png')