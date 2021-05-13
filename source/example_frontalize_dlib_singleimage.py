#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstrates facial landmark frontalization in a static image. Loads the image,
detects faces, extracts landmarks using DLIB and frontalizes them.

@author: Vasileios Vonikakis
"""


import numpy as np
import dlib
import matplotlib.pyplot as plt
from utils import plot_landmarks, frontalize_landmarks, get_landmark_array
import imageio


plt.close('all')

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat") 
frontalization_weights = np.load('../data/frontalization_weights.npy')
image = imageio.imread('../data/faces.jpg')  # load image

faces = detector(image)  # detect faces

for i, face in enumerate(faces):
    
    landmarks_raw = predictor(image, face)  # detect landmarks
    landmarks = get_landmark_array(landmarks_raw)
    landmarks_frontal = frontalize_landmarks(landmarks, frontalization_weights)
    
    if landmarks is not None:
        
        # initialize new image
        fig = plt.figure(figsize=(7,3))
        
        plt.subplot(1,3,1)
        plt.title('Detected face')
        x1 = landmarks_raw.rect.left()
        y1 = landmarks_raw.rect.top()
        x2 = x1 + landmarks_raw.rect.width()
        y2 = y1 + landmarks_raw.rect.height()
        plt.imshow(image[y1:y2, x1:x2, :])
        plt.axis(False)
        
        plt.subplot(1,3,2)
        plt.title('Original landmarks')
        
        plt.subplot(1,3,3)
        plt.title('Frontalized landmarks')
        
        plt.suptitle('Face ' + str(i+1))
        plt.tight_layout()
        axes = fig.get_axes()
        
        plot_landmarks(landmarks, axis=axes[1])
        plot_landmarks(landmarks_frontal, axis=axes[2])

        plt.show()
