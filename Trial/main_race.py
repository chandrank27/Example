import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from Color_detection import *
from Lane_detection_test import *
from picarx import Picarx

px = Picarx()
px.set_dir_servo_angle(-5)
px.set_camera_tilt_angle(0)
px.set_cam_pan_angle(30)
steering_angle_state = 0


def set_steering_angle(steering_angle, sa_max = 40):

    #This function is used to control the steering servo
    
    steering_angle -= 5
     
    if abs(steering_angle) > sa_max: 
        if steering_angle < 0:
            px.set_dir_servo_angle(-sa_max)
        else:
            px.set_dir_servo_angle(sa_max)
    else:
        px.set_dir_servo_angle(steering_angle)
    
    return steering_angle

def steering_angle_from_lane_detection(image):
    hsv = convert_to_HSV(image)
    edges = detect_mask(hsv)
    roi = region_of_interest(edges)
    line_segments = detect_line_segments(roi)
    lane_lines = average_slope_intercept(image,line_segments)
    steering_angle, no_line  = get_steering_angle(image, lane_lines)
    line_image = display_lines(image, lane_lines)
    heading_image = display_heading_line(line_image, steering_angle+5)
    return (steering_angle+5), no_line

with PiCamera() as camera:
    camera.resolution = (640, 480)  
    camera.framerate = 12
    rawCapture = PiRGBArray(camera, size=camera.resolution)  
    time.sleep(3)
    race_start = False
    #px.forward(30)
    
    i = 0
    
    for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True): # use_video_port=True
        img = frame.array
                
        steering_angle, no_line = steering_angle_from_lane_detection(img)
        steering_angle -= 80
        
        time.sleep(0.05)
        
        if not no_line:
            steering_angle_state = set_steering_angle(steering_angle = steering_angle)
        else:
            steering_angle_state = set_steering_angle(steering_angle = 0)
            '''
            if steering_angle_state < 0:
                steering_angle_state = set_steering_angle(steering_angle = 30)
                time.sleep(0.5)
            else:
                steering_angle_state = set_steering_angle(steering_angle = -30)
                time.sleep(0.5)
            '''

        rawCapture.truncate(0)

        i += 1          
        if i == 10:
            px.forward(30)
            
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            px.forward(0)
            break

    print('quit ...') 
    cv2.destroyAllWindows()
    camera.close()
