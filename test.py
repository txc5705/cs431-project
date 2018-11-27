#!/usr/bin/env python3
# Author: axl1439

import cv2
import numpy as np
import os

import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EYE_CASCADE = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, 'haarcascade_eye.xml'))
FACE_CASCADE = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml'))

def main():
    cap = cv2.VideoCapture(os.path.join(BASE_DIR, 'data/face2.mp4'))
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error reading video')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Frame', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
        cv2.imshow('p', frame)
        # green = frame[:,:,1]
        # cv2.imshow('G', green)
        # integral_image = cv2.integral(frame)
        # cv2.imshow('Integral', integral_image)
        if (cv2.waitKey(0) & 255) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
