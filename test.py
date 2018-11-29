#!/usr/bin/env python3
# Author: axl1439

import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import os

import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EYE_CASCADE = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, 'haarcascade_eye.xml'))
FACE_CASCADE = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml'))


# integral_image = cv2.integral(frame)
# cv2.imshow('Integral', integral_image)

def test_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error opening webcam')
        return
    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)

def main():
    cap = cv2.VideoCapture(os.path.join(BASE_DIR, 'data/face2.mp4'))
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error reading video')
    subplot = plt.subplot()
    window = []  
    def update(i):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = frame[y:y + h, x:x + w]
            green = np.sum(face[:,:,1])
            window.append(np.sum(face[:,:,1]))
            subplot.plot(np.arange(len(window)), window, 'b')
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)
    
    animation = anim.FuncAnimation(plt.gcf(), update, interval=16)
    plt.show()

if __name__ == '__main__':
    # test_webcam()
    main()
