import os
import numpy as np
import cv2

class IntrinsicCalibration(object):
    def __init__(
            self,
            height,
            width,
            square_length,
            markser_length,
            aruco_dict=cv2.aruco.DICT_4X4_1000
        ) -> None:
        self.dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.board = cv2.aruco.CharucoBoard((height, width), square_length, markser_length, self.dictionary)
        self.params = cv2.aruco.DetectorParameters()

    def __call__(self, image_path_list):
        all_charuco_corners = []
        all_charuco_ids = []
        for image_file in image_path_list:
            image = cv2.imread(image_file)
            img_width, img_height = image.shape[:2]
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, self.dictionary, parameters=self.params)
            if marker_ids is None:
                continue
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, self.board)
            if charuco_retval>3:
                print({
                    "corners":charuco_corners,
                    "ids":charuco_corners
                })
                assert len(charuco_corners)==len(charuco_ids),"len(charuco_corners) != len(charuco_ids)"
                
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, self.board, image.shape[:2], None, None)
        return {
            "image_size":[img_height,img_width],
            "K":np.squeeze(camera_matrix).tolist(),
            "dist":np.squeeze(dist_coeffs).tolist()
        }