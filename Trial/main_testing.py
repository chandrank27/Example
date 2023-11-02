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
px.set_cam_pan_angle(-3)
steering_angle_state = 0


def set_steering_angle(steering_angle, sa_max = 30):

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
    edges = detect_mask(image)
    roi = region_of_interest(edges)
    line_segments = detect_line_segments(roi)
    lane_lines = average_slope_intercept(image,line_segments)
    steering_angle, no_line  = get_steering_angle(image, lane_lines)
    line_image = display_lines(image, lane_lines)
    heading_image = display_heading_line(line_image, steering_angle+5)
    return (steering_angle+5), roi, heading_image, no_line

with PiCamera() as camera:
    
    camera.resolution = (320, 240)  
    camera.framerate = 12
    rawCapture = PiRGBArray(camera, size=camera.resolution)  
    time.sleep(3)
    race_start = True
    px.forward(30)
    
    steering_angles = [-5]
    
    i = 0
    j = 0
    cam_pan_angle = [50, 25, 0, -25, 50]
    
    for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True): # use_video_port=True
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            px.forward(0)
            break
        
        img = frame.array
        
        cropped_img = get_image_top(img, 0, int(img.shape[0]/4), int(img.shape[1]/3), int(2*img.shape[1]/3))
        #cv2.imshow("video", cropped_img)
        reds = detect_red(cropped_img)
        greens = detect_green(cropped_img)
        
        if not race_start and reds > 0 and greens == 0:           
            print("Ready to Race!!")
        if not race_start and greens > 0 and reds == 0:
            race_start = True
            px.set_camera_tilt_angle(0)
            time.sleep(0.2)
            px.forward(30)
            print("Start!!")
        
        if race_start:
            steering_angle, filter_image1, filter_image2, no_line = steering_angle_from_lane_detection(img)
            steering_angle -= 80
        
            time.sleep(0.05)
                       
            if not no_line:
                if j == 0:
                    px.forward(30)
                    steering_angles.append(steering_angle)
                    steering_angles = steering_angles[1:]
                else:
                    px.forward(15)
                    steering_angle_state = set_steering_angle(steering_angle = cam_pan_angle[(j-1)%4])
                    time.sleep(1)
                    px.forward(0)
                    steering_angle_state = set_steering_angle(steering_angle = -5)
                    px.set_cam_pan_angle(-3)
                    px.forward(30)
                    j = 0
                    
                    
            else:
                if j == 0:
                    px.forward(0)
                    steering_angle_state = set_steering_angle(steering_angle = -5)
                    px.backward(15)
                    time.sleep(1)
                    px.backward(0)
                    
                px.set_cam_pan_angle(cam_pan_angle[j%5])
                j += 1
                #px.forward(30)
                                
            #if state != 0:
            #    steering_angle_state = state
            
            print(steering_angles)
            steering_angle_state = set_steering_angle(steering_angle = steering_angles[0])

            #cv2.imshow("video", filter_image2)  # OpenCV image show
            #cv2.imwrite('images/imageR'+str(i).zfill(5) +'.jpg', filter_image1)
            #cv2.imwrite('images/imageH'+str(i).zfill(5) +'.jpg', filter_image2)
            i += 1
                      
            distance = px.ultrasonic.read()
            #print("distance: ",distance)
            if distance > 0 and distance < 300:
                if distance < 10:
                    px.forward(0)
                    steering_angle_state = set_steering_angle(steering_angle = -5)
                    px.backward(30)
                    time.sleep(1)
                    px.forward(0)
        
        rawCapture.truncate(0)
                

    print('quit ...') 
    cv2.destroyAllWindows()
    camera.close()