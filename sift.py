# coding:utf-8
import cv2
import numpy as np

image_path1 = 'D:\\git_folder\\dis_direction\\1.jpg'
image_path2 = 'D:\\git_folder\\dis_direction\\3.jpg'

gray1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
gray2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

sift=cv2.SIFT()
kp1, des1=sift.detectAndCompute(gray1,None)
kp2, des2=sift.detectAndCompute(gray2,None)

#FLANN匹配参数
FLANN_INDEX_KDTREE=0
indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
searchParams=dict(checks=50)
flann=cv2.FlannBasedMatcher(indexParams,searchParams)
matches=flann.knnMatch(des1,des2,k=2)
matchesMask=[[0,0] for i in range(len(matches))]
for i,(m,n) in enumerate(matches):
    if m.distance<0.7*n.distance:
        matchesMask[i]=[1,0]
        
drawParams=dict(matchColor=(0,255,0),
                singlePointColor=(255,0,0),
                matchesMask=matchesMask,
                flags=0)
resultImage=cv2.drawMatchesKnn(gray1,kp1,gray2,kp2,matches,None,**drawParams)
cv2.namedWindow('Flann',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Flann',400,600)
cv2.imshow('Flann',resultImage)
cv2.waitKey(0)
cv2.destroyAllWindows()