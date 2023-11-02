import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from Color_detection import *
from Lane_detection_test_2 import *
from picarx import Picarx
import matplotlib.pyplot as plt

px = Picarx()
px.set_camera_tilt_angle(0)
px.set_cam_pan_angle(0)
px.set_dir_servo_angle(-5)
steering_angle_state = 0

def steering_angle_from_lane_detection(image):
    hsv = convert_to_HSV(image)
    edges = detect_edges(hsv)
    #warp_img = perspective_warp(image)
    #edges = detect_mask(hsv)
    #edges = detect_edge_sobel(image)
    roi = region_of_interest(edges)
    line_segments = detect_line_segments(roi)
    lane_lines, lines_count = average_slope_intercept(image,line_segments)
    steering_angle, no_line  = get_steering_angle(image, lane_lines, lines_count)
    line_image = display_lines(image, lane_lines)
    heading_image = display_heading_line(line_image, steering_angle+5)
    return (steering_angle+5), roi, heading_image, no_line


j = 0
cam_pan_angle = [50, 25, 0, -25, -50, -25, 0, 25]
        
with PiCamera() as camera:
    camera.resolution = (640, 480)  
    camera.framerate = 12
    rawCapture = PiRGBArray(camera, size=camera.resolution)  
    time.sleep(3)
    race_start = False
 
    for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True): # use_video_port=True
        img = frame.array
                
        steering_angle, filter_image1, filter_image2, no_line = steering_angle_from_lane_detection(img)
        
        print(steering_angle)
        
        '''
        if not no_line:
            if j == 0:
                print("Line visible")
            else:
                print("Line Found")
                px.set_cam_pan_angle(-3)
                j = 0
        
                
        else:
            print("Line not visible, Searching ...")
            px.set_cam_pan_angle(cam_pan_angle[j%8])
            time.sleep(2)
            j +=1
        '''

        cv2.imshow("video", filter_image2)  # OpenCV image show
        
        rawCapture.truncate(0)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            px.forward(0)
            break

    print('quit ...') 
    cv2.destroyAllWindows()
    camera.close()