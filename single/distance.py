import math
import numpy as np

h_image_width = 400
h_image_height = 400
H = 5


def get_py(vp_x, vp_y, K):
    p_infinity = np.array([vp_x, vp_y, 1])#消失点
    K_inv = np.linalg.inv(K)  # K的逆矩阵，k为内参
    r3 = K_inv @ p_infinity  # 数组相乘
    r3 /= np.linalg.norm(r3)  # r3的行列式
    pitch = np.arcsin(r3[1])
    yaw = -np.arctan2(r3[0], r3[2])
    return pitch, yaw

def cal_distance(vp_x, vp_y, mtx, x, y):  # 输入目标框底部中心点x,y坐标
    pitch, yaw = get_py(vp_x, vp_y, mtx)
    pitch = math.fabs(float(pitch))
    hfov = math.atan(400 / mtx[0][0])  # 弧度值-半水平视场角
    vfov = math.atan(300 / mtx[1][1])  # 弧度值-半垂直视场角
    if y > h_image_height/2:
        dis_y = H / math.tan(pitch + math.atan((y - h_image_height / 2) * 2 * math.tan(vfov) / h_image_height))
    else:
        dis_y = H / math.tan(pitch - math.atan((h_image_height / 2 - y) * 2 * math.tan(vfov) / h_image_height))
    dis_x = math.sqrt(H**2 + dis_y**2) * math.fabs(x - h_image_width) * math.tan(hfov) / h_image_width
    distance = math.sqrt(dis_x**2 + dis_y**2)
    return distance
