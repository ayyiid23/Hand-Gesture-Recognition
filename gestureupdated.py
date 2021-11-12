import cv2
import numpy as np
import math
import time
import datetime

# cap is the pointer used for video
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, img = cap.read()
     # reads the frame given by the webcam
    cv2.rectangle(img,(300,300),(100,100),(0,255,0),0)
     # we take a subscreen in rectangular form
    crop_img = img[100:300, 100:300]
     # crop the image into the rectangle

    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # converted to grayscale (used for contour extraction)

    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)
    # apply Gaussian Blur to remove noise and details from the image

    # Otsu's Thresholding: for changing into binary 
    _, thresh1 = cv2.threshold(blurred, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # shows the thresholded image
    #cv2.imshow('Thresholded', thresh1)

    contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
            cv2.CHAIN_APPROX_NONE)

    # finding contour with maximum area
    max_area = -1
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i
    cnt=contours[ci]

    # create a bounding rectangle around the contour
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0)

    # find convex hull
    hull = cv2.convexHull(cnt)

    # draw the contours around the hand
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing,[cnt],0,(0,255,0),0)
    cv2.drawContours(drawing,[hull],0,(0,0,255),0)

    # find convex hull
    hull = cv2.convexHull(cnt,returnPoints = False)

    # find convexity defects and draw contours on the defects
    defects = cv2.convexityDefects(cnt,hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)

    # applying Cosine Rule to find angle for all defects (between fingers)
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        # find length of all the sides of triangle (of the defect)
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        
        # apply cosine rule
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        
        # ignore angles>90 and highlight the rest with dots
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img,far,1,[255,0,0],-1)
       
        # draw line from start to end i.e. the convex points (finger tips)
        dist = cv2.pointPolygonTest(cnt,(int(far[0]), int(far[1])),1)
        cv2.line(crop_img,start,end,[0,255,0],2)
        cv2.circle(crop_img,far,5,[0,0,255],-1)

    if count_defects == 1:
        cv2.putText(img,"FAN SPEED: 2", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 2:
        cv2.putText(img,"FAN SPEED: 3", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 3:
        cv2.putText(img,"FAN SPEED: 4", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 4:
        cv2.putText(img,"FAN SPEED: 5", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    else:
        cv2.putText(img,"FAN SPEED: 1", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    
    #cv2.imshow('drawing', drawing)
    #cv2.imshow('end', crop_img)

    # show images
    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img))
    # cv2.imshow('Contours', all_img)

    k = cv2.waitKey(10)
    if k == 27:
        break