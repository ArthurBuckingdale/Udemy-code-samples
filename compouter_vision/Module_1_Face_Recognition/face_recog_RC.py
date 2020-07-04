# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:10:12 2020

@author: ryanc
"""

#the purpose of this script is to learn how to use open CV for face detection. 
#I'm uncertain if it has the libraries that are needed for computer vision in 
#a drone for example. I'm almost for sure going to do that in MATLAB but 
#we'll see how this goes. It's definitely not the best algo. But it will give me 
#some character and some programming skill in python.

import cv2

#loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#defining the function that will do the detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0),2)
        
    return frame

#doing some face recognition with a webcam(i don't have one)
video_capture = cv2.VideoCapture(0)
while True:
    _,frame=video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas=detect(gray,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

        