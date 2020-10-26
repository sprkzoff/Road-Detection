import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image) :
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny_image = cv2.Canny(blur,50,150)
    return canny_image

def region_of_interest(image) :
    h = image.shape[0]
    w = image.shape[1]
    polygons = np.array([[(20,h),(w-20,h),(w//2,h//2-50)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    mask_image = cv2.bitwise_and(image,mask)
    return mask_image

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None :
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def make_coor(image,line_params) :
    slope,intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines) :
    left_fit = []
    right_fit = []
    for line in lines :
        x1,y1,x2,y2 = line.reshape(4)
        params = np.polyfit((x1,x2),(y1,y2),1)
        slope = params[0]
        intercept = params[1]
        if slope < 0 :
            left_fit.append((slope,intercept))
        else :
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit,axis=0)
    left_line =  make_coor(image,left_fit_average)
    right_line = make_coor(image,right_fit_average)
    return np.array([left_line,right_line])

cap = cv2.VideoCapture("./datasets/test2.mp4")
while(cap.isOpened()): 
    _, frame = cap.read()
    lane_image = frame
    canny_image = canny(lane_image)
    res = region_of_interest(canny_image)
    thres = 100
    lines = cv2.HoughLinesP(res,2,np.pi/180,thres,np.array([]),minLineLength=40,maxLineGap=5)
    average_lines = average_slope_intercept(lane_image,lines)
    print(average_lines)
    print()
    average_line_image = display_lines(lane_image,average_lines)
    combine_average = cv2.addWeighted(lane_image, 0.8, average_line_image, 1, 1)
    all_point = []
    for i in range(average_lines.shape[0]) :
        x1,y1,x2,y2 = average_lines[i]
        combine_average = cv2.circle(combine_average, (x1,y1), 5, (0,0,255), -1)
        combine_average = cv2.circle(combine_average, (x2,y2), 5, (0,0,255), -1)
        if i==0 :
            all_point.append((x1,y1))
            all_point.append((x2,y2))
            cv2.putText(combine_average,"L("+str(x2)+","+str(y2)+")",(x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2)
        else :
            all_point.append((x2,y2))
            all_point.append((x1,y1))
            cv2.putText(combine_average,"R("+str(x2)+","+str(y2)+")",(x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2)
    cnt = np.array( all_point )
    cv2.drawContours(combine_average, [cnt], 0, (0,255,0), -1)
    cv2.imshow('Road tracking',combine_average)
    cv2.waitKey(1)

