#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimates Frontalization Weights from a large dataset of facial landmarks with 
multiple faces, each one with multiple expressions and headposes. 

 - Fills 2 matrices A and Y, as described in the paper:
V. Vonikakis, S. Winkler. (2020). Identity Invariant Facial Landmark 
Frontalization for Facial Expression Analysis. Proc. ICIP2020, October 2020.

 - Then runs a Least Squares optimization.

@author: Vasileios Vonikakis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_landmark_matrix, get_procrustes, mirror_landmarks
from utils import plot_landmarks, frontalize_landmarks


plt.close('all')


# load data
df_data = pd.read_csv('../data/landmark_dataset.csv')
landmarks_mean_face = np.load('../data/landmarks_mean_face.npy')

# initializations
ls_subjects = list(set(df_data['subject_id'].values))  # find unique subjects
ls_matrix_A = []  # feature values
ls_matrix_Y = []  # ground truth values



for i,subject in enumerate(ls_subjects):  # going through all the subjects
    print('Processing subject', i, '/', len(ls_subjects), 
          ' [', round((100*i)/len(ls_subjects), 2), '%]')

    df_subject = df_data[df_data['subject_id'] == subject]  # subject's data
    
    # find unique expressions of this subject
    ls_expressions = list(set(df_subject['expression'].values))
    
    
    for expression in ls_expressions:  # go through all the expressions
        print('     expression:', expression)
        
        df_subject_expression = df_subject[
            df_subject['expression'] == expression
            ]
        
        # find the frontal pose
        df_frontal = df_subject_expression[
            (df_subject_expression['pose_pitch'] == 0) & 
            (df_subject_expression['pose_yaw'] == 0)
            ]
        
        if len(df_frontal) == 0:
            print('Problem! No frontal view found!')
        else:
            # frontal face found
            
            # processing frontal face
            ls_coord = list(df_frontal.iloc[0, 6:].values)
            landmarks_frontal_raw = get_landmark_matrix(ls_coord)
            landmarks_frontal = get_procrustes(
                landmarks_frontal_raw, 
                template_landmarks=landmarks_mean_face  # identity invariance
                )  
            
            
            # accross all headposes of the same expression of this subject
            for j in range(len(df_subject_expression)):
                
                # processing non-frontal faces
                ls_coord = list(df_subject_expression.iloc[j, 6:].values)
                landmarks_pose_raw = get_landmark_matrix(ls_coord)
                landmarks_pose = get_procrustes(
                    landmarks_pose_raw, 
                    template_landmarks=None  # no identity invariance
                    )  
                
                # filling matrices
                
                # normal landmakrs
                ls_matrix_A.append( 
                    list(landmarks_pose[:,0]) + 
                    list(landmarks_pose[:,1]) 
                    )
                ls_matrix_Y.append( 
                    list(landmarks_frontal[:,0]) + 
                    list(landmarks_frontal[:,1]) 
                    )
                
                # mirrored landmarks
                landmarks_frontal_flipped = mirror_landmarks(landmarks_frontal)
                landmarks_pose_flipped = mirror_landmarks(landmarks_pose)
                ls_matrix_A.append( 
                    list(landmarks_pose_flipped[:,0]) + 
                    list(landmarks_pose_flipped[:,1]) 
                    )
                ls_matrix_Y.append( 
                    list(landmarks_frontal_flipped[:,0]) + 
                    list(landmarks_frontal_flipped[:,1]) 
                    )
            
            
        
# prepare matrices
matrix_A = np.array(ls_matrix_A)
matrix_Y = np.array(ls_matrix_Y)
interception = np.ones((len(matrix_A),1)) 
matrix_A_1 = np.hstack( (matrix_A, interception) )  # adding interception

# Least Squares fit
ls_fit = np.linalg.lstsq(a=matrix_A_1, b=matrix_Y, rcond=None)
#TODO: implement ridge regression (did not want to use SKlearn as dependency!)

frontalization_weights = ls_fit[0]
np.save(
        '../data/frontalization_weights.npy', 
        frontalization_weights, 
        allow_pickle=True, 
        fix_imports=True
        )



# simple test on a ramdom face from the dataset to show the frontalization
ls_coord = list(df_data.iloc[30010, 6:].values)
landmarks = get_landmark_matrix(ls_coord)
plot_landmarks(landmarks, axis=None, title='Original')
landmarks_frontal = frontalize_landmarks(ls_coord, frontalization_weights)
plot_landmarks(landmarks_frontal, axis=None, title='Frontalized')


