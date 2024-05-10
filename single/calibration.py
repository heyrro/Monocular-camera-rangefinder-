import cv2
import numpy as np
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

w = 9
h = 6


objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
objp = 2 * objp
objpoints = []
imgpoints = []

def get_internal_reference(path):
    images = glob.glob(path + '/*.jpg')
    for frame in images:
        img = cv2.imread(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (w,h),None)   #在灰度图像 gray 中找到棋盘格角点

        if ret:
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)  #亚像素级别的角点精确化
            objpoints.append(objp)#将世界坐标系中的棋盘格角点的理论坐标（objp）加入 objpoints 和 imgpoints 中。
            # 这些点用于后续相机标定的计算。
            imgpoints.append(corners)#图像坐标系中检测到的角点坐标（corners）加入  imgpoints 中
            cv2.drawChessboardCorners(img, (w,h), corners, ret)
        # cv2.imshow('draw',img)
        # cv2.waitKey(0)
    #利用棋盘格角点的理论坐标和图像坐标进行相机标定。
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # ret表示的是重投影误差；mtx是相机的内参矩阵；dist表述的相机畸变参数；
    # rvecs表示标定棋盘格世界坐标系到相机坐标系的旋转参数：rotation vectors，需要进行罗德里格斯转换；
    # tvecs表示translation vectors，主要是平移参数。
    #返回相机矩阵 mtx 和畸变参数 dist
    return ret, mtx, dist, rvecs, tvecs
