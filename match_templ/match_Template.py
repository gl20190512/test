# -*- coding:utf-8 -*-
import sys
import cv2
import time
import numpy as np

def PCBA_matchTemplate(image, templ, rsz_scale=1.0):
    #缩小图像后再匹配
    if rsz_scale != 1.0:
        rsz_im = cv2.resize(image, None, fx=rsz_scale, fy=rsz_scale)
        rsz_tpl = cv2.resize(templ, None, fx=rsz_scale, fy=rsz_scale)
    else:
        rsz_im = image
        rsz_tpl = templ
        
    cv2.namedWindow("img",1)
    cv2.imshow("img", rsz_im)
    cv2.namedWindow("templ",1)
    cv2.imshow("templ", rsz_tpl)
    
    #模板匹配，找到模板在图像中的位置:(x,y,width,height)   
    th, tw = rsz_tpl.shape[:2]
    result = cv2.matchTemplate(rsz_im, rsz_tpl, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print("min_val, max_val: ", min_val, max_val)
    
    #返回坐标位置(x,y,width,height)  
    return int(min_loc[0]/rsz_scale), int(min_loc[1]/rsz_scale), \
           int(tw/rsz_scale), int(th/rsz_scale)
    
if __name__ == "__main__":
    if len(sys.argv) == 3:
        image_path = sys.argv[1]
        templ_path = sys.argv[2]
    else:
        image_path = 'Image_20190426214058.jpg'
        templ_path = 'templ2.jpg'
    
    image = cv2.imread(image_path) #图像
    templ = cv2.imread(templ_path) #模板
    
    tim = time.time()
    x, y, width, height = PCBA_matchTemplate(image,templ,0.1) 
    print("Done in : ", time.time()-tim, " seconds")
    
    #绘制模板位置
    cv2.rectangle(image, (x, y),  (x+width, y+height), [0, 255, 0],15)
    cv2.namedWindow("result",0)
    cv2.imshow("result", image)
    cv2.waitKey(0)

