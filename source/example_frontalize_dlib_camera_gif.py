#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstrates real-time facial landmark frontalization in a video stream from 
a camera. Captures frames, detects faces, extracts landmarks using DLIB and 
frontalizes them.

@author: Vasileios Vonikakis
"""


import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt
from utils import plot_landmarks, frontalize_landmarks, get_landmark_array
from array2gif import write_gif


cap = cv2.VideoCapture(0)  # camera object
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat") 
frontalization_weights = np.load('../data/frontalization_weights.npy')

# initialize image
fig = plt.figure()
plt.subplot(1,2,1)
plt.title('Original landmarks')
plt.subplot(1,2,2)
plt.title('Frontalized landmarks')
# plt.suptitle('Original vs frontalized landmarks \n (press q to quit)')
plt.tight_layout()
plt.show()
axes = fig.get_axes()

frames = []

f = 0
while(f < 50):  # for 300 frames
    print('frame', f)

    

    
    # numpy mage from figure plot
    ax = fig.gca()
    ax.axis('off')
    fig.tight_layout(pad=0)
    ax.margins(0)  # To remove the huge white borders
    fig.canvas.draw()
    image_1d = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image_1d.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(image)

    
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    ret, frame = cap.read()  # capture a frame 
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(image)  # detect faces
    
    for face in faces:
        landmarks_raw = predictor(image, face)  # detect landmarks
        landmarks = get_landmark_array(landmarks_raw)
        landmarks_frontal = frontalize_landmarks(
            landmarks, 
            frontalization_weights
            )
        
        if landmarks is not None:
            axes[0].clear()
            plot_landmarks(
                landmarks, 
                axis=axes[0], 
                title='Original'
                )
            axes[1].clear()
            plot_landmarks(
                landmarks_frontal, 
                axis=axes[1], 
                title='Frontalized'
                )
            plt.pause(0.001)
        
    f += 1


# When everything done, release resources
cap.release()
cv2.destroyAllWindows()

print('Saving gif...')
write_gif(frames[10:], 'test.gif', fps=5)