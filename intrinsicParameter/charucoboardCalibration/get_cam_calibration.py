import time
from threading import Lock

import os
import numpy as np
import cv2
from tqdm import tqdm

class IntrinsicCalibration(object):
    def __init__(
            self,
            height,
            width,
            square_length,
            markser_length,
            image_path,
            aruco_dict=cv2.aruco.DICT_4X4_1000
        ) -> None:
        self.dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.board = cv2.aruco.CharucoBoard((height, width), square_length, markser_length, self.dictionary)
        self.params = cv2.aruco.DetectorParameters()
        # debug path
        self.lock=Lock()
        self.debug_path=os.path.join(image_path,"vis_corners")
        if not os.path.exists(self.debug_path):
            os.mkdir(self.debug_path)
        self.debug_number=0

    def __call__(self, image_path_list,cam_index=0):
        all_charuco_corners = []
        all_charuco_ids = []
        # start=time.time()
        for image_file in tqdm(image_path_list,position=cam_index):
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            size = image.shape[::-1]
            img_height, img_width = image.shape[:2]
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
                image, 
                self.dictionary, 
                parameters=self.params
            )
            if marker_ids is None:
                continue
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, 
                marker_ids, 
                image, 
                self.board
            )
            if charuco_retval>=6: # 解DLT问题需要的最少解
                # print({
                #     "corners":charuco_corners,
                #     "ids":charuco_corners
                # })
                assert len(charuco_corners)==len(charuco_ids),"len(charuco_corners) != len(charuco_ids)"
                with self.lock:
                    self.vis_corners(image,charuco_corners,charuco_ids)
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
        # print(f"detect corners time: {time.time()-start}")
        # import pdb;pdb.set_trace()
        # start=time.time()
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_charuco_corners, 
            charucoIds=all_charuco_ids, 
            board=self.board, 
            imageSize=size, 
            cameraMatrix=None, 
            distCoeffs=None,
            flags=cv2.CALIB_USE_LU # LU is much faster than SVD
        )
        # print(f"calibrate camera charuco time: {time.time()-start}")
        # import pdb;pdb.set_trace()
        return {
            "image_size":[img_width,img_height],
            "K":np.squeeze(camera_matrix).tolist(),
            "dist":np.squeeze(dist_coeffs).tolist()
        }
    
    def forward(self,img_path):
        image = cv2.imread(img_path)
        img_width, img_height = image.shape[:2]
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, self.dictionary, parameters=self.params)
        if marker_ids is None:
            return None,None,None
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, self.board)
        return charuco_retval, charuco_corners, charuco_ids
    
    def vis_corners(self,image,corners,ids):
        if self.debug_number>=30:
            return 
        image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        for step,corner in list(zip(ids,corners)):
            # import pdb;pdb.set_trace()
            image=cv2.circle(
                img=image,
                center=np.squeeze(corner.astype(np.int64)),
                radius=3,
                color=(0,255,0),
                thickness=-1
            )
            image=cv2.putText(
                img=image,
                text=str(step.item()),
                org=np.squeeze(corner.astype(np.int64)),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1,
                color=(0,0,255),
                thickness=2
            )
        cv2.imwrite(os.path.join(self.debug_path,f"{self.debug_number}.jpg"),image)
        self.debug_number+=1