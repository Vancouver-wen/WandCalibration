import os
import numpy as np
import cv2

# ------------------------------
# ENTER YOUR REQUIREMENTS HERE:
ARUCO_DICT = cv2.aruco.DICT_4X4_1000
SQUARES_VERTICALLY = 5
SQUARES_HORIZONTALLY = 4
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.022
# ...
PATH_TO_YOUR_IMAGES = '/home/wenzihao/Desktop/WandCalibration/testData/charucoboard'
# ------------------------------

def calibrate_and_save_parameters():
    # Define the aruco dictionary and charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()

    # Load PNG images from folder
    image_files = [os.path.join(PATH_TO_YOUR_IMAGES, f) for f in os.listdir(PATH_TO_YOUR_IMAGES) if f.endswith(".jpg")]
    image_files.sort()  # Ensure files are in order

    all_charuco_corners = []
    all_charuco_ids = []

    for image_file in image_files:
        image = cv2.imread(image_file)
        # image=cv2.flip(image,1)
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)
        if marker_ids is None:
            continue
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)
        if charuco_retval>3:
            print({
                "corners":charuco_corners,
                "ids":charuco_corners
            })
            assert len(charuco_corners)==len(charuco_ids),"len(charuco_corners) != len(charuco_ids)"
            
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)

    # Calibrate camera
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, image.shape[:2], None, None)
    print("ret:", retval)
    print("mtx:\n", camera_matrix.reshape(1,-1).tolist()) # 内参数矩阵
    print("dist:\n", dist_coeffs.tolist())  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    # # Save calibration data
    # np.save('camera_matrix.npy', camera_matrix)
    # np.save('dist_coeffs.npy', dist_coeffs)

    # # Iterate through displaying all the images
    # for image_file in image_files:
    #     image = cv2.imread(image_file)
    #     undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    #     cv2.imshow('Undistorted Image', undistorted_image)
    #     cv2.waitKey(0)

    # cv2.destroyAllWindows()

calibrate_and_save_parameters()
exit(0)

def detect_pose(image, camera_matrix, dist_coeffs):
    # Undistort the image
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

    # Define the aruco dictionary and charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()

    # Detect markers in the undistorted image
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(undistorted_image, dictionary, parameters=params)

    # If at least one marker is detected
    if len(marker_ids) > 0:
        # Interpolate CharUco corners
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, undistorted_image, board)

        # If enough corners are found, estimate the pose
        if charuco_retval:
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None)

            # If pose estimation is successful, draw the axis
            if retval:
                cv2.drawFrameAxes(undistorted_image, camera_matrix, dist_coeffs, rvec, tvec, length=0.1, thickness=15)
    return undistorted_image


def main():
    # Load calibration data
    camera_matrix = np.load['camera_matrix.npy']
    dist_coeffs = np.load['dist_coeffs.npy']

    # Iterate through PNG images in the folder
    image_files = [os.path.join(PATH_TO_YOUR_IMAGES, f) for f in os.listdir(PATH_TO_YOUR_IMAGES) if f.endswith(".png")]
    image_files.sort()  # Ensure files are in order

    for image_file in image_files:
        # Load an image
        image = cv2.imread(image_file)

        # Detect pose and draw axis
        pose_image = detect_pose(image, camera_matrix, dist_coeffs)

        # Show the image
        cv2.imshow('Pose Image', pose_image)
        cv2.waitKey(0)

main()