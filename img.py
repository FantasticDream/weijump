# -*- coding: utf-8 -*-
import cv2
import numpy as np
#找到棋子
def self_detect(img):
    #切片图片
    region_upper=int(img.shape[0]*0.3)
    region_lower=int(img.shape[0]*0.7)
    region = img[region_upper:region_lower]
    #rgb转hsv
    hsv_img = cv2.cvtColor(region,cv2.COLOR_BGR2HSV)
    #黑色色区区分
    color_lower = np.int32([105,25,45])
    color_upper = np.int32([135,125,130])
    color_mask = cv2.inRange(hsv_img,color_lower,color_upper)
    #返回轮廓本身和每条轮廓对应的属性（寻找轮廓图像，轮廓检索模式，轮廓近似方法）
    contours = cv2.findContours(color_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    if len(contours)>0:
        max_contour = max(contours,key=cv2.contourArea)
        max_contour = cv2.convexHull(max_contour)
        rect = cv2.boundingRect(max_contour)
        x,y,w,h=rect
        center_coord=(x+int(w/2),y+h+region_upper-20)
        cv2.circle(img,center_coord,5,(0,255,0),-1)
        return center_coord
    # cv2.imshow('region',img)
#找出目标地点
def goal_detect(img,body_position):
    region_upper=int(img.shape[0]*0.3)
    region_lower=int(img.shape[0]*0.7)
    if body_position[0]<(img.shape[1]/2.0):
        region_left=body_position[0]+30
        region_right = img.shape[1]-30
    else:
        region_left=30
        region_right = body_position[0]-30
    region = img[region_upper:region_lower,region_left:region_right]
    # cv2.imshow('test',region)

    #轮廓处理,提取边缘--重点
    edge_list=[0,0,0,0]
    for i in range(3):
        region_gray=cv2.cvtColor(region,cv2.COLOR_BGR2HSV)[:,:,i]
        #region_gray=cv2.equalizeHist(region_gray)
        edge_list[i]=cv2.Canny(region_gray,100,160)
    
    region_gray=cv2.cvtColor(region,cv2.COLOR_BGR2GRAY)
    edge_list[3] = cv2.Canny(region_gray, 100, 160,apertureSize=5)
    edge_list[1]=np.bitwise_or(edge_list[0],edge_list[1])
    edge_list[2]=np.bitwise_or(edge_list[2],edge_list[1])
    edge_final=np.bitwise_or(edge_list[3],edge_list[2])
    # cv2.imshow('test2',edge_final)



    contours= cv2.findContours(edge_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]  # cv2.CHAIN_APPROX_NONE

    #max_contour = max(contours, key=cv2.contourArea)
    #max_contour = cv2.convexHull(max_contour)

    y=99999
    for contour in contours:
        sorted_top=sorted(contour,key=lambda contour:contour[0][1])
        if sorted_top[0][0][1]<y:
            raw_x = x = sorted_top[0][0][0]
            raw_y = y = sorted_top[0][0][1]

    print((int(x+ region_left), int(y+ region_upper)))


    mask = np.zeros((region_lower-region_upper+ 2, region_right-region_left + 2), np.uint8)
    mask=cv2.floodFill(region, mask, (raw_x, raw_y+16), [255,25,255])[2]
    cv2.circle(img, (int(x + region_left), int(y + region_upper)), 5, (255, 0, 5), -1)

    
    M = cv2.moments(mask)
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])

    if y<raw_y or abs(x-raw_x)>40:
        x=raw_x;y=raw_y
        y += region_upper
        x += region_left
        y = (-abs(x-body_position[0])/pow(3,0.5)+body_position[1])
        print("TESTTEST")

    #cv2.imshow("test",mask)
    #cv2.imshow("edge", edge_final)
    else:
        y += region_upper
        x += region_left

    #y = (-abs(x-body_position[0])/pow(3,0.5)+body_position[1])
    # cv2.circle(img, (int(x),int(y)), 5, (0,0,255), -1)
    cv2.imshow('dealed',img)
    return [x, y]




img = cv2.imread('test.png')
#压缩图像减少系统性能
img = cv2.resize(img,(720,1080))
#测试图片类型<type 'numpy.ndarray'>
#print(type(img))

#cv2.imshow('test',img)
pos = self_detect(img)
goal_detect(img,pos)
#如果不用waitKey会让操作系统不刷新这个窗口误认为这个窗口卡死
cv2.waitKey(0)
