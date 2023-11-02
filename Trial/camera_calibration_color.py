import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from Color_detection import *
from Lane_detection_test import *
from picarx import Picarx
import matplotlib.pyplot as plt

px = Picarx()
px.set_camera_tilt_angle(10)
px.set_cam_pan_angle(-3)
px.set_dir_servo_angle(-5)
steering_angle_state = 0

with PiCamera() as camera:
    camera.resolution = (320, 240)  
    camera.framerate = 12
    rawCapture = PiRGBArray(camera, size=camera.resolution)  
    time.sleep(3)
    race_start = False
 
    for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True): # use_video_port=True
        img = frame.array
                
          # OpenCV image show
        #cv2.imwrite('images/imageR'+str(i).zfill(5) +'.jpg', filter_image1)
        #cv2.imwrite('images/imageH'+str(i).zfill(5) +'.jpg', filter_image2)

        cropped_img = get_image_top(img, 0, int(img.shape[0]/4), int(img.shape[1]/3), int(2*img.shape[1]/3))
        cv2.imshow("video", cropped_img)
        reds = detect_red(cropped_img)
        greens = detect_green(cropped_img)
        
        #print(f'Reds: {reds}')
        #print(f'Greens: {greens}')
        
        if not race_start and reds > 0 and greens == 0:           
            print("Ready to Race!!")
        if not race_start and greens > 0 and reds == 0:
            race_start = True
            print("Start!!")

        
        rawCapture.truncate(0)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            px.forward(0)
            break

    print('quit ...') 
    cv2.destroyAllWindows()
    camera.close()