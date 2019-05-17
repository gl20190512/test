# -*- coding:utf-8 -*-
import sys
import cv2
import time
import numpy as np

#根据分块特征，获取图像方向向量（类似于LBP特征）
#输入为灰度图像
def LBP_dirction(gray_im): 
    width = gray_im.shape[1]
    height = gray_im.shape[0]
    block_w = int(width/3)
    block_h = int(height/3)
     
    median_mx = np.zeros([3,3])
    dir_vec = list()
    
    for i in range(3):
        for j in range(3):
            b_x = j * block_w
            b_y = i * block_h
            block = gray_im[b_y:b_y+block_h,b_x:b_x+block_w]
            median_mx[i,j] = np.max(block)
            
    center_median = median_mx[1,1]   
    
    """
    0-1-2  从(0,0)块开始，顺时针编码方向 
    |   |
    7   3
    |   |
    6-5-4 
    """
    dir_vec.append(center_median<median_mx[0,0])
    dir_vec.append(center_median<median_mx[0,1])
    dir_vec.append(center_median<median_mx[0,2])
    dir_vec.append(center_median<median_mx[1,2])
    dir_vec.append(center_median<median_mx[2,2])
    dir_vec.append(center_median<median_mx[2,1])
    dir_vec.append(center_median<median_mx[2,0])
    dir_vec.append(center_median<median_mx[1,0])
    return dir_vec

def compare_direction_LBP(gray_img1, gray_img2):
    dir1 = LBP_dirction(gray_img1) 
    dir2 = LBP_dirction(gray_img2)
    print(dir1,dir2)
    return dir1 == dir2
    
if __name__ == "__main__":
    if len(sys.argv) == 3:
        image_path1 = sys.argv[1]
        image_path2 = sys.argv[2]
    else:
        image_path1 = '1.jpg'
        image_path2 = '3.jpg'
    
    gray_im1 = cv2.imread(image_path1,0)
    gray_im2 = cv2.imread(image_path2,0)
    
    tim = time.time()
    is_same_dir = compare_direction_LBP(gray_im1,gray_im2) #输入为灰度图像
    print("Is it the same direction? ", is_same_dir)
    print("Done in : ", time.time()-tim, " seconds")
    
    #绘制左右边界
    cv2.namedWindow('im1',0)
    cv2.imshow('im1',gray_im1)
    cv2.namedWindow('im2',0)
    cv2.imshow('im2',gray_im2)
    cv2.waitKey(0)
