from argparse import _ActionsContainer
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time

kernel_5 = np.ones((5,5),np.uint8)

def get_image_top(image, frame_top, frame_bottom, frame_left, frame_right):
    image_top =  image[frame_top:frame_bottom, frame_left:frame_right]
    return image_top

def detect_red(image):
    height = image.shape[0]
    width = image.shape[1]  
    resized_img = cv2.resize(image, (int(height),int(width)), interpolation=cv2.INTER_LINEAR) #shrinking the image for better performance
    hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    

    mask1 = cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255]))
    mask2 = cv2.inRange(hsv, np.array([160,100,100]), np.array([180,255,255]))
    mask = cv2.bitwise_or(mask1, mask2)

    morphologyEx_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_5,iterations=1)

    mask_image, contours, hierarchy =  cv2.findContours(morphologyEx_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    return len(contours)

def detect_green(image):
    height = image.shape[0]
    width = image.shape[1]  
    resized_img = cv2.resize(image, (int(height),int(width)), interpolation=cv2.INTER_LINEAR) #shrinking the image for better performance
    hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array([40,100,100]), np.array([70,255,255]))

    morphologyEx_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_5,iterations=1)

    mask_image, contours, hierarchy =  cv2.findContours(morphologyEx_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    return len(contours)

