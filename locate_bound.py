# -*- coding:utf-8 -*-
import sys
import cv2
import time
import numpy as np

#根据边缘的投影直方图，定位板子的左右边界
def locate_PCBA_bound(gray_im, rsz_width=300):
    #缩小图像，长宽比不变
    rsz_w = rsz_width
    rsz_h = int(gray_im.shape[0]*rsz_w/gray_im.shape[1])
    scale = (rsz_w+0.0)/gray_im.shape[1]
    rsz_im = cv2.resize(gray_im,(rsz_w,rsz_h))
    
    #canny边缘检测
    smooth_im = cv2.GaussianBlur(rsz_im, (3,3),0)
    canny_im = cv2.Canny(smooth_im, 50, 150)
    
    #形态学处理
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))     
    #morph = cv2.morphologyEx(canny_im, cv2.MORPH_GRADIENT, kernel)
    
    #二值化
    ret,binary = cv2.threshold(canny_im,127,255,cv2.THRESH_OTSU)
    
    #垂直方向上的投影直方图
    bin_sum = binary.sum(axis=0)
    #大于投影均值一半的列，认为是可能属于板子的列
    cmp = bin_sum < bin_sum.mean()/2
    
    #找最大连通列，返回连通列的左右边界值
    flag = 0
    begin = -1
    end = -1
    seg = list()
    for i in range(len(cmp)):
        if cmp[i]== False:
            if flag==0:
                begin = i
                flag = 1
            else:
                end = i             
        if cmp[i]== True or i==len(cmp)-1:
            flag = 0
            if begin > 0:
                seg.append((begin,end,end-begin))
                begin = end = -1
    left = -1
    right = -1
    np_seg = np.array(seg)
    max_idx = np.argmax(np_seg[:,2])
    #最大连通列，大于图像宽的1/8，才认为有效 
    if seg[max_idx][2] > len(cmp)/8:
        left = int(seg[max_idx][0] / scale)
        right = int(seg[max_idx][1] / scale)
    return left, right
    
if __name__ == "__main__":
    if len(sys.argv) == 2:
        image_path = sys.argv[1]
    else:
        image_path = 'C:\\Users\\LG\\Desktop\\Image_20190426214058.jpg'
    
    gray_im = cv2.imread(image_path,0)
    color_im = cv2.imread(image_path,1)
    
    tim = time.time()
    left, right = locate_PCBA_bound(gray_im) #输入为灰度图像
    print("Done in : ", time.time()-tim, " seconds")
    
    #绘制左右边界
    cv2.line(color_im, (left, 0), (left, color_im.shape[0]), (0, 255, 0), 16)
    cv2.line(color_im, (right, 0), (right, color_im.shape[0]), (0, 255, 0), 16)
    cv2.namedWindow('test',0)
    cv2.imshow('test',color_im)
    cv2.waitKey(0)
