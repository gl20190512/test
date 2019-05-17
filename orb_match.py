# coding:utf-8
import os
import sys
import glob
import cv2
import time
import numpy as np

#最少匹配点对数
MIN_MATCH_COUNT = 10

if __name__ == "__main__":
    im_names = []
    templ_name = 'D:\\PCB\\20190311\\orig\\Image_20190311091726271.jpg'
    if len(sys.argv) > 1:
        for i in range(1,len(sys.argv)):
            im_names.append(sys.argv[i])
    else:
        #image_path='D:\\PCB\\20190327-2\\20190327机型3\\机型3正确照'
        image_path='D:\\PCB\\20190311\\orig'
        im_names = glob.glob(os.path.join(image_path,"*.jpg"))
    
    #读入模板图像    
    templ = cv2.imread(templ_name,cv2.IMREAD_GRAYSCALE)
    #缩放图像
    templ = cv2.resize(templ,None,fx=0.25,fy=0.25)

    # 创建ORB特征检测器和描述符
    orb = cv2.ORB_create()
    # 对模板图像检测特征和描述符
    templ_kp, templ_des = orb.detectAndCompute(templ, None)

    for im_name in im_names:
        img = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)
        tim = time.time()
        #缩放图像
        img = cv2.resize(img,None,fx=0.25,fy=0.25)
        # 对测试图像检测特征和描述符
        img_kp, img_des = orb.detectAndCompute(img, None)
        # knn匹配器
        mt = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
        matches = mt.knnMatch(templ_des,img_des,k=2)
        print("Done in : ", time.time()-tim, " seconds")
        
        #初步去除误匹配
        good = []
        for m,n in matches:
            if m.distance < 0.9*n.distance:
                good.append(m)
        #采样RANSAC方法进一步去除误匹配
        if len(good) > MIN_MATCH_COUNT: #最少匹配数量
            src_pts = np.float32([ templ_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ img_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            matches = [[x] for i,x in enumerate(good) if matchesMask[i]==1]
            print("%d perfect matches are found - %f" % (sum(matchesMask),(sum(matchesMask)+0.0)/len(good)))
        else:
            print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        
        #print(matches)
        #绘制匹配
        img3 = cv2.drawMatchesKnn(templ, templ_kp, img, img_kp, matches, None, flags=2)
        cv2.namedWindow("matches",0)
        cv2.imshow("matches", img3)
        key = cv2.waitKey()
        if key == 27:
            break
    cv2.destroyAllWindows()




