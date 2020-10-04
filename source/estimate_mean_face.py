#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimates the Mean Face shape from a large dataset of facial landmarks with 
multiple faces, each one with multiple expressions and headposes. Then it 
stores the shape as a numpy array.

@author: Vasileios Vonikakis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_landmark_matrix, get_procrustes, plot_landmarks


plt.close('all')
df_data = pd.read_csv('../data/landmark_dataset.csv')
ls_coord_all = []

for i in range(len(df_data)):
    print(
        'Processing face', i, '/', len(df_data), 
          ' [', round((100*i)/len(df_data), 2), '%]'
          )
    
    ls_coord = list(df_data.iloc[i, 6:].values)
    landmarks = get_landmark_matrix(ls_coord)
    landmarks_standard = get_procrustes(landmarks)
    ls_coord_all.append( 
        list(landmarks_standard[:,0]) + 
        list(landmarks_standard[:,1]) 
        )

coord_all = np.array(ls_coord_all)
coord_all_mean = np.mean(coord_all, axis=0)
landmarks_mean_face = get_landmark_matrix(coord_all_mean)
plot_landmarks(landmarks_mean_face, axis=None, title='Mean face')

np.save(
        '../data/landmarks_mean_face.npy', 
        landmarks_mean_face, 
        allow_pickle=True, 
        fix_imports=True
        )
