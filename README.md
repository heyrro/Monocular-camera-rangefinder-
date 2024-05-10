# Monocular-camera-rangefinder-
single文件里calibration images是标定棋盘格的图像，data里面是需要识别的车辆视频。yolo文件是下载的权重参数。
pt.py为主代码，运行他即可。
其中，calibration.py是相机标定的代码，detect.py是yolov5检测车辆的代码；distance.py为计算距离；vp.py为计算消失点。
