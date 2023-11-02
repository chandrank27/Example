import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from Color_detection import *
from Lane_detection_test import *
from picarx import Picarx

px = Picarx()
px.set_dir_servo_angle(-5)
#px.set_camera_tilt_angle(0)
px.set_cam_pan_angle(0)

