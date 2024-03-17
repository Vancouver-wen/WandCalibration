import cv2
import numpy as np
import glob

class IntrinsicCalibration(object):
    def __init__(
            self,
            height,
            width,
        ) -> None:
        # height, width: 棋盘格角点规格
        self.height=height
        self.width=width
        # 获取标定板角点的在世界坐标系位置， 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
        self.objp = np.zeros((self.height * self.width, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.height, 0:self.width].T.reshape(-1, 2)
        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
        self.criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    def __call__(self, image_path_list):
        obj_points = []
        img_points = []
        for image_path in image_path_list:
            img = cv2.imread(image_path)
            img_width, img_height = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (self.height, self.width), None)
            if ret:
                obj_points.append(self.objp)
                # 专门用来获取棋盘图上内角点的精确位置的， 即在原角点的基础上寻找亚像素角点
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), self.criteria) 
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)
        # calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
        return {
            "image_size":[img_width,img_height],
            "K":np.squeeze(mtx).tolist(),
            "dist":np.squeeze(dist).tolist()
        }