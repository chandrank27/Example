import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from Color_detection import *
from Lane_detection_test import *
from picarx import Picarx

cam_pan_correction = 0

px = Picarx()

def set_steering_angle(steering_angle, sa_max = 25):

    #This function is used to control the steering servo   
    if abs(steering_angle) > sa_max: 
        if steering_angle < 0:
            steering_angle = -sa_max
            steering_angle -= 5
            px.set_dir_servo_angle(-sa_max)

        else:
            steering_angle = sa_max
            steering_angle -= 5
            px.set_dir_servo_angle(sa_max)

    else:
        steering_angle -= 5
        px.set_dir_servo_angle(steering_angle)
    
    print(steering_angle)
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
    return (steering_angle+5), roi, heading_image, no_line

px.set_dir_servo_angle(-5)
px.set_camera_tilt_angle(0)
px.set_cam_pan_angle(cam_pan_correction)

steering_angle_state = -5
i = 0
j = 0
cam_pan_angle = [50, 25, 0, -25, -50, -25, 0, 25]
collision_recovery_mode = 0
no_lane_mode = 0
steering_angles = [-5]

with PiCamera() as camera:
    
    camera.resolution = (320, 240)  
    camera.framerate = 12
    rawCapture = PiRGBArray(camera, size=camera.resolution)  
    time.sleep(3)
    race_start = True
        
    for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True): # use_video_port=True
        
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            px.forward(0)
            break
        
        img = frame.array
        rawCapture.truncate(0)
        
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
            px.forward(30)
            steering_angle, filter_image1, filter_image2, no_line = steering_angle_from_lane_detection(img)         
            distance = px.ultrasonic.read()
            
            cv2.imwrite('images/imageR'+str(i).zfill(5) +'.jpg', filter_image1)
            cv2.imwrite('images/imageH'+str(i).zfill(5) +'.jpg', filter_image2)
            #cv2.imshow("video", filter_image2)  # OpenCV image show
            i += 1

            if distance > 0 and distance < 10:
                if collision_recovery_mode == 0:
                    print('Commencing Collision Recovery...')
                    px.forward(0)
                    px.set_dir_servo_angle(-5)
                    steering_angle_state = -5  
                    time.sleep(0.2)
                    px.backward(15)
                    time.sleep(1)
                    px.backward(0)
                    time.sleep(1)
                    collision_recovery_mode = 1
                    continue
                
            if collision_recovery_mode == 1:
                if no_line:
                    print('Searching lane ...')
                    px.set_cam_pan_angle(cam_pan_angle[j%8] + cam_pan_correction)
                    j += 1
                    continue
                else:
                    print('Lane found ...')
                    ('Collision Recovery complete ...')                  
                    steering_angle_state = set_steering_angle(steering_angle = cam_pan_angle[(j-1)%8] - 5)
                    
                    px.forward(15)
                    time.sleep(1)
                    px.forward(0)
                    
                    px.set_cam_pan_angle(cam_pan_correction)
                    px.set_dir_servo_angle(-5)
                    steering_angle_state = -5
                    time.sleep(1)
                    px.forward(30)
                    
                    j = 0
                    collision_recovery_mode = 0
                    

            if no_line:
                #no_lane_mode = 1
                print("No Lane detected. Continuing course ...")
                px.forward(0)
                time.sleep(0.5)
                px.forward(30)
                continue
            else:
                steering_angles.append(steering_angle)
                steering_angles = steering_angles[1:]
                steering_angle_state = set_steering_angle(steering_angle = steering_angles[0])
                time.sleep(0.05)
                

    print('quit ...') 
    cv2.destroyAllWindows()
    camera.close()