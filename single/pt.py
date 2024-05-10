import cv2
import numpy as np
import os
import time
from calibration import get_internal_reference
from vp import FilterLines, GetLines, GetVanishingPoint
from distance import cal_distance, get_py
from detect import Yolov5

def camera_calibration(path):  # path为棋盘标定内参的图片路径
    if os.path.exists(path):
        ret, mtx, dist, rvecs, tvecs = get_internal_reference(path)
    else:
        os.makedirs(path)
        cap = cv2.VideoCapture(0)
        count = 0
        i = 1
        EXTRACT_FREQUENCY = 10
        while len(os.listdir(path)) < 25:
            _, frame = cap.read()
            if frame is None:
                break
            if count % EXTRACT_FREQUENCY == 0:
                save_path = 'E:/project/single/{}/{}.jpg'.format(path, i)
                cv2.imwrite(save_path, frame)
                cv2.imshow('calibration_img', frame)
                cv2.waitKey(1000)
                i += 1
            count += 1
        cv2.destroyAllWindows()
        ret, mtx, dist, rvecs, tvecs = get_internal_reference(path)
    return ret, mtx, dist, rvecs, tvecs #(mtx是相机的内参矩阵；dist表述的相机畸变参数；)

ret, mtx, dist, rvecs, tvecs = camera_calibration('calibration images')
cap = cv2.VideoCapture('E:/project/data/2.mp4')
yolo = Yolov5()
while True:
    _, frame1 = cap.read()
    frame = cv2.resize(frame1, (800, 600))
    start = time.time()
    # dst_frame = cv2.resize(frame, (800, 600))
    # h, w = frame.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    # dst_frame = cv2.undistort(frame1, mtx, dist, None, newcameramtx)
    # x, y, w, h = roi
    # dst_frame = dst_frame[y:y+h, x:x+w]

    lines = GetLines(frame)
    vp = GetVanishingPoint(lines)  #消失点坐标
    # if vp is None:
    #     continue
    for line in lines:
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
    cv2.circle(frame, (int(vp[0]), int(vp[1])), 3, (0, 0, 255), 2)
    img, cor = yolo.run(frame)
    for xywh in cor:
        cenx = int(xywh[0])
        ceny = int(xywh[1])
        width = int(xywh[2])
        height = int(xywh[3])
        print(cenx, ceny, width, height)
        distance = cal_distance(int(vp[0]), int(vp[1]), mtx, cenx, ceny+height/2)
        print(distance)
        cv2.putText(img, '%.2fm' % distance, (cenx-int(width/2), ceny-int(height/2)-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)

    end = time.time()
    print(end - start)

    cv2.imshow('start', frame1)
    cv2.imshow('final', img)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()