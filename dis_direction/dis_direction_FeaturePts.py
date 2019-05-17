# coding:utf-8
import cv2
import time
import numpy as np
import math

DEBUG = True

#计算向量之间的夹角
#vec:(begin_pt,end_pt)
def _angle(vec1, vec2):
    dx1 = vec1[1][0] - vec1[0][0]
    dy1 = vec1[1][1] - vec1[0][1]
    dx2 = vec2[1][0] - vec2[0][0]
    dy2 = vec2[1][1] - vec2[0][1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

#根据提取的特征点，判断两个图像的方向是否一致
#输入为灰度图像
#如果方向一致，  返回 1; 
#    方向不一致，返回 0; 
#    无法判断，  返回 None
def compare_direction_FeaturePts(gray_img1, gray_img2, angle_thres=30): 
   
    #创建ORB特征检测器和描述符
    orb = cv2.ORB_create()
    #对两幅图像检测特征和描述符
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    #转换成numpy array
    pts2f_1 = cv2.KeyPoint_convert(kp1)
    pts2f_2 = cv2.KeyPoint_convert(kp2)

    #获取flann匹配器
    FLANN_INDEX_LSH=6
    indexParams=dict(algorithm=FLANN_INDEX_LSH, 
                 table_number = 12,
                 key_size = 20,
                 multi_probe_level = 2)
    searchParams=dict(checks=100)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    #进行匹配
    matches = flann.knnMatch(des1, des2, k=2)

    #去除空匹配
    matches = filter(None,matches)
    #最少需要2组匹配点
    if len(matches) < 2:
        return None

    #根据匹配距离排序,从小到大
    matches = sorted(matches, key = lambda x:x[0].distance)
    #取最匹配(距离最小)的两个点对
    mt_pts_1 = [pts2f_1[matches[0][0].queryIdx], \
                pts2f_1[matches[3][0].queryIdx]]
    mt_pts_2 = [pts2f_2[matches[0][0].trainIdx], \
                pts2f_2[matches[3][0].trainIdx]]
    
    #计算向量之间的夹角（两个向量分别是由每幅图中的两个点构成）
    angle = _angle(mt_pts_1, mt_pts_2)
    print('Angle between vectors: ',angle)
    
    if DEBUG:
        #绘制匹配
        img3 = np.zeros((max(img1.shape[0],img2.shape[0]), \
                        img1.shape[1]+img2.shape[1]),dtype=np.uint8)
        img3[:img1.shape[0],:img1.shape[1]] = img1
        img3[:img2.shape[0],img1.shape[1]:] = img2
        img3 = cv2.cvtColor(img3,cv2.COLOR_GRAY2BGR)
        line1 = np.array([mt_pts_1[0],mt_pts_2[0]+[img1.shape[1],0]],\
                         np.int32).reshape((-1, 1, 2))
        line2 = np.array([mt_pts_1[1],mt_pts_2[1]+[img1.shape[1],0]],\
                         np.int32).reshape((-1, 1, 2))
        cv2.polylines(img3, [line1,line2], True, (0, 255, 255))
        cv2.namedWindow("matches",1)
        cv2.imshow("matches", img3)
        #'''
    
    #根据向量的夹角判别方向是否一致
    if angle > angle_thres:
        return False
    else:
        return True

if __name__ == "__main__":
    image_path1 = '13.jpg'
    image_path2 = '14.jpg'
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)#输入为灰度图像
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    
    tim = time.time()
    is_same_dir = compare_direction_FeaturePts(img1,img2)
    print("Done in : ", time.time()-tim, " seconds")
    print("Is it the same direction? ", is_same_dir)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

