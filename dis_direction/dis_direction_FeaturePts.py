# coding:utf-8
import cv2
import time
import numpy as np
import math

DEBUG = True

#��������֮��ļн�
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

#������ȡ�������㣬�ж�����ͼ��ķ����Ƿ�һ��
#����Ϊ�Ҷ�ͼ��
#�������һ�£�  ���� 1; 
#    ����һ�£����� 0; 
#    �޷��жϣ�  ���� None
def compare_direction_FeaturePts(gray_img1, gray_img2, angle_thres=30): 
   
    #����ORB�����������������
    orb = cv2.ORB_create()
    #������ͼ����������������
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    #ת����numpy array
    pts2f_1 = cv2.KeyPoint_convert(kp1)
    pts2f_2 = cv2.KeyPoint_convert(kp2)

    #��ȡflannƥ����
    FLANN_INDEX_LSH=6
    indexParams=dict(algorithm=FLANN_INDEX_LSH, 
                 table_number = 12,
                 key_size = 20,
                 multi_probe_level = 2)
    searchParams=dict(checks=100)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    #����ƥ��
    matches = flann.knnMatch(des1, des2, k=2)

    #ȥ����ƥ��
    matches = filter(None,matches)
    #������Ҫ2��ƥ���
    if len(matches) < 2:
        return None

    #����ƥ���������,��С����
    matches = sorted(matches, key = lambda x:x[0].distance)
    #ȡ��ƥ��(������С)���������
    mt_pts_1 = [pts2f_1[matches[0][0].queryIdx], \
                pts2f_1[matches[3][0].queryIdx]]
    mt_pts_2 = [pts2f_2[matches[0][0].trainIdx], \
                pts2f_2[matches[3][0].trainIdx]]
    
    #��������֮��ļнǣ����������ֱ�����ÿ��ͼ�е������㹹�ɣ�
    angle = _angle(mt_pts_1, mt_pts_2)
    print('Angle between vectors: ',angle)
    
    if DEBUG:
        #����ƥ��
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
    
    #���������ļн��б����Ƿ�һ��
    if angle > angle_thres:
        return False
    else:
        return True

if __name__ == "__main__":
    image_path1 = '13.jpg'
    image_path2 = '14.jpg'
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)#����Ϊ�Ҷ�ͼ��
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    
    tim = time.time()
    is_same_dir = compare_direction_FeaturePts(img1,img2)
    print("Done in : ", time.time()-tim, " seconds")
    print("Is it the same direction? ", is_same_dir)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

