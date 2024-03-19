import cv2
import numpy as np
import glob

def get_camera_intrinsic(h, w, image_path_list):
    # h, w: 棋盘格角点规格

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的在世界坐标系位置， 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    '''
    objp:
    [[0. 0. 0.]
    [1. 0. 0.]
    [2. 0. 0.]
    ...
    [4. 4. 0.]
    [5. 4. 0.]
    [6. 4. 0.]]
    '''
    objp = np.zeros((h * w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)

    obj_points = []
    img_points = []
  
    for image_path in image_path_list:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (h, w), None)
        print(ret)

        if ret:
            obj_points.append(objp)

            # 专门用来获取棋盘图上内角点的精确位置的， 即在原角点的基础上寻找亚像素角点
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria) 
            
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)

            # cv2.drawChessboardCorners(img, (h, w), corners, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(1000)

    print(len(img_points))
    # cv2.destroyAllWindows()

    # calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    np.set_printoptions(suppress=True)
    print("ret:", ret)
    print("mtx:\n", mtx.reshape(1,-1).tolist()) # 内参数矩阵
    print("dist:\n", dist.tolist())  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    # print("rvecs:\n", len(rvecs))  # 旋转向量  # 外参数
    # print("tvecs:\n", tvecs ) # 平移向量  # 外参数
    # print("-----------------------------------------------------")

    # 用标定结果把空间三维坐标点映射回像素坐标系 再和 标定图片上的亚像素角点坐标 做对比
    total_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i],img_points2, cv2.NORM_L2)/len(img_points2)
        total_error += error
    # 单位是像素
    print("average error: ", total_error/len(img_points2))

def main():
    image_path_list = glob.glob("/home/wenzihao/Desktop/WandCalibration/testData/checkerboard/*.jpg")
    get_camera_intrinsic(5, 4, image_path_list)

if __name__ == '__main__':
    main()

