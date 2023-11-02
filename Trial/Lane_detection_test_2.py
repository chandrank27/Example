
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def convert_to_HSV(frame):
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  return hsv

def detect_edge_sobel(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def perspective_warp(img,
                     dst_size=(640,480),
                     src=np.float32([(0.05,0.6),(0.95,0.7),(0,0.66),(1,0.8)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped


def detect_edges_canny(frame):
    lower_black = np.array([0, 0, 0], dtype = "uint8") # lower limit of blue color
    upper_black = np.array([180, 255, 80], dtype="uint8") # upper limit of blue color
    mask = cv2.inRange(frame,lower_black,upper_black) # this mask will filter out everything but blue

    # detect edges
    edges = cv2.Canny(mask, 50, 150) 
    return edges


def detect_mask(frame):
    lower_black = np.array([0, 0, 0], dtype = "uint8") # lower limit of blue color
    upper_black = np.array([180, 255, 80], dtype="uint8") # upper limit of blue color
    mask = cv2.inRange(frame,lower_black,upper_black) # this mask will filter out everything but blue
        
    return mask

def detect_edges(frame):
    dilation_kernel = np.ones((7,7), np.uint8)
    erosion_kernel = np.ones((3,3), np.uint8)
    lower_black = np.array([0, 0, 0], dtype = "uint8") # lower limit of blue color
    upper_black = np.array([180, 255, 80], dtype="uint8") # upper limit of blue color
    mask = cv2.inRange(frame,lower_black,upper_black) # this mask will filter out everything but blue
    dilated_mask = cv2.dilate(mask, dilation_kernel, iterations = 1)
    eroded_mask = cv2.erode(dilated_mask, erosion_kernel, iterations = 1)
    # detect edges
    edges = cv2.Canny(eroded_mask, 50, 150, apertureSize = 3, L2gradient = True)
    
    return edges

def region_of_interest(edges):
    height, width = edges.shape # extract the height and width of the edges frame
    mask = np.zeros_like(edges) # make an empty matrix with same dimensions of the edges frame

    # only focus lower half of the screen
    # specify the coordinates of 4 points (lower left, upper left, upper right, lower right)
    polygon = np.array([[
        (0, height), 
        (0,  2*height/3),
        (width , 2*height/3),
        (width , height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, (255,255,255)) # fill the polygon with blue color 
    cropped_edges = cv2.bitwise_and(edges, mask) 
    return cropped_edges

def detect_line_segments(cropped_edges):
    rho = 1  
    theta = np.pi / 180  
    min_threshold = 5
    line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold, 
                                    np.array([]), minLineLength=5, maxLineGap=0)
    
    #for line_segment in line_segments:
    #    for x1, y1, x2, y2 in line_segment:
    #        cv2.line(cropped_edges, (x1, y1), (x2, y2), (255, 0, 0), 5)
            
    return line_segments

def average_slope_intercept(frame, line_segments):
    lane_lines = {'left':[], 'right':[], 'cleft':[], 'cright':[]}
    lines_count = [1,1,1,1]

    if line_segments is None:
        print("no line segment detected")
        return lane_lines

    height, width,_ = frame.shape
    left_fit = []
    right_fit = []
    central_left_fit = []
    central_right_fit = []
    boundary = 1/3

    right_region_boundary = width * (1 - boundary) 
    left_region_boundary = width * boundary
    centre_boundary = width * 0.5

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                #print("skipping vertical lines (slope = infinity)")
                continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            
            if slope < -0.15:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
                elif x1 < centre_boundary and x2 < centre_boundary:
                    central_left_fit.append((slope, intercept))
                    
            elif slope > 0.15:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))
                elif x1 > centre_boundary and x2 > centre_boundary:
                    central_right_fit.append((slope, intercept))
                    

    print(f'Left fit: {len(left_fit)}')
    print(f'Right fit: {len(right_fit)}')
    print(f'Central left fit: {len(central_left_fit)}')
    print(f'Central right fit: {len(central_right_fit)}')
    
    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        #print(left_fit_average)
        lane_lines['left'] = make_points(frame, left_fit_average)[0]

    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        #print(right_fit_average)
        lane_lines['right'] = make_points(frame, right_fit_average)[0]
        
    if len(central_left_fit) > 0:
        central_left_fit_average = np.average(central_left_fit, axis=0)
        #print(left_fit_average)
        lane_lines['cleft'] = make_points(frame, central_left_fit_average)[0]

    if len(central_right_fit) > 0:
        central_right_fit_average = np.average(central_right_fit, axis=0)
        #print(right_fit_average)
        lane_lines['cright'] = make_points(frame, central_right_fit_average)[0]

    lines_count = [len(left_fit), len(right_fit), len(central_left_fit), len(central_right_fit)]
    #lines_count = [1, 1, 1, 1]
    
    return lane_lines, lines_count

def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(2*y1 / 3)  # make points from middle of the frame down

    if slope == 0: 
        slope = 0.1    

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [[x1, y1, x2, y2]]

def get_steering_angle(frame, lane_lines, lines_count):
    height, width, _ = frame.shape
    no_line_left = False
    no_line_right = False
        
    left_angles = []
    right_angles = []
    
    left_weights = 0
    right_weights = 0
        
    #print(lane_lines)
    if len(lane_lines['left']) == 0 and len(lane_lines['cleft']) == 0:  
        left_angles.append(0)
        left_weights = 1
        no_line_left = True

    if len(lane_lines['right']) == 0 and len(lane_lines['cright']) == 0:  
        right_angles.append(0)
        right_weights = 1
        no_line_left = False
        
    if len(lane_lines['left']) > 0:
        x1, _, x2, _ = lane_lines['left']
        x_offset = int(x2 - x1)
        y_offset = int(height / 3)
        left_angles.append(lines_count[0] * math.atan(x_offset / y_offset) / 8)
        left_weights += lines_count[0]
        
    if len(lane_lines['right']) > 0:
        x1, _, x2, _ = lane_lines['right']
        x_offset = int(x2 - x1)
        y_offset = int(height / 3)
        right_angles.append(lines_count[1] * math.atan(x_offset / y_offset) / 8)
        right_weights += lines_count[1]
        
    if len(lane_lines['cleft']) > 0:
        x1, _, x2, _ = lane_lines['cleft']
        x_offset = int(x2 - x1)
        y_offset = int(height / 3)
        left_angles.append(lines_count[2] * math.atan(x_offset / y_offset) * 3)
        left_weights += lines_count[2]
        
    if len(lane_lines['cright']) > 0:
        x1, _, x2, _ = lane_lines['cright']
        x_offset = int(x2 - x1)
        y_offset = int(height / 3)
        right_angles.append(lines_count[3] * math.atan(x_offset / y_offset) * 3)
        right_weights += lines_count[3]

    left_angle = sum(left_angles) / left_weights
    print(left_angle)
    right_angle = sum(right_angles) / right_weights
    print(right_angle)
    angle_to_mid_radian = (left_angle + right_angle)/2
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  
    steering_angle = angle_to_mid_deg
    
    no_line = no_line_left or no_line_right
    
    return steering_angle, no_line

'''
The following two functions are only for visualising the lane detection
They do not contribute to the function of the PiCar
'''

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6): # line color (B,G,R)
    line_image = np.zeros_like(frame)
    
    for region in lines:
        if len(lines[region]) == 4:
                x1, y1, x2, y2 = lines[region]
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)

    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)  
    return line_image

def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5):

    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
                       
    steering_angle += 90

    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 3 / math.tan(steering_angle_radian))
    y2 = int(height / 3)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)

    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image