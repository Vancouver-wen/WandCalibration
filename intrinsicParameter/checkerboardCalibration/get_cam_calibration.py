import os
import sys
from threading import Lock

import cv2
import numpy as np
import glob
from tqdm import tqdm

class IntrinsicCalibration(object):
    def __init__(
            self,
            height,
            width,
            image_path
        ) -> None:
        # height, width: 棋盘格角点规格
        self.height=height
        self.width=width
        # 获取标定板角点的在世界坐标系位置， 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
        self.objp = np.zeros((self.height * self.width, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.height, 0:self.width].T.reshape(-1, 2)
        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
        self.criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        # debug path
        self.lock=Lock()
        self.debug_path=os.path.join(image_path,"vis_corners")
        self.debug_number=0
    
    def get_corners(self,image_path):
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (self.height,self.width), None)
        # print(ret,corners)
        if ret:
            # 专门用来获取棋盘图上内角点的精确位置的， 即在原角点的基础上寻找亚像素角点
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), self.criteria) 
            if [corners2]:
                return ret,self.objp,corners2
            else:
                return ret,self.objp,corners
        else:
            return None,None,None

    def __call__(self, image_path_list,cam_index=0):
        obj_points = []
        img_points = []
        for image_path in tqdm(image_path_list,position=cam_index):
            img = cv2.imread(image_path)
            # if img is None:  # 可能存在部分图片损坏的情况
            #     import pdb;pdb.set_trace()
            img_height, img_width = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (self.height,self.width), None)
            # print(ret,corners)
            if ret:
                obj_points.append(self.objp)
                # 专门用来获取棋盘图上内角点的精确位置的， 即在原角点的基础上寻找亚像素角点
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), self.criteria) 
                if [corners2]:
                    with self.lock:
                        self.vis_corners(img,corners2)
                    img_points.append(corners2)
                else:
                    img_points.append(corners)
        # calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=obj_points, 
            imagePoints=img_points, 
            imageSize=size, 
            cameraMatrix=None, 
            distCoeffs=None,
            flags=cv2.CALIB_USE_LU # LU is much faster than SVD
        )
        return {
            "image_size":[img_width,img_height],
            "K":np.squeeze(mtx).tolist(),
            "dist":np.squeeze(dist).tolist()
        }
    
    def vis_corners(self,image,corners):
        if self.debug_number>=30:
            return 
        for step,corner in enumerate(corners):
            image=cv2.circle(
                img=image,
                center=np.squeeze(corner.astype(np.int64)),
                radius=3,
                color=(0,255,0),
                thickness=-1
            )
            image=cv2.putText(
                img=image,
                text=str(step),
                org=np.squeeze(corner.astype(np.int64)),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1,
                color=(0,0,255),
                thickness=2
            )
        if not os.path.exists(self.debug_path):
            os.mkdir(self.debug_path)
        cv2.imwrite(os.path.join(self.debug_path,f"{self.debug_number}.jpg"),image)
        self.debug_number+=1