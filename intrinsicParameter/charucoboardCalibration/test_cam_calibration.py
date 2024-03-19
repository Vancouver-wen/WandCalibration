import os
import sys
sys.path.append('./')
import time

import numpy as np
import cv2
import glob

from .get_cam_calibration import IntrinsicCalibration


if __name__=="__main__":
    myIntrinsicCalibration=IntrinsicCalibration(
        height=4,
        width=5,
        square_length=0.2,
        markser_length=0.15
    )
    image_folder_path="/home/wenzihao/Desktop/WandCalibration/imageCollectWTT/board/cam2"
    image_path_list=glob.glob(os.path.join(image_folder_path,'*'))
    cv2.namedWindow("image",cv2.WINDOW_GUI_NORMAL)
    for image_path in image_path_list:
        rets,corners,ids=myIntrinsicCalibration.forward(img_path=image_path)
        if rets is not None and rets>0:
            image=cv2.imread(image_path)
            print(corners,ids)
            for corner,id in list(zip(corners,ids)):
                # import pdb;pdb.set_trace()
                cv2.circle(
                    img=image,
                    center=np.squeeze(corner.astype(np.int32)).tolist(),
                    radius=5,
                    color=(0,0,255),
                    thickness=-1
                )
                cv2.putText(
                    img=image,
                    text=str(id.item()),
                    org=np.squeeze(corner.astype(np.int32)).tolist(),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=2,
                    color=(0,0,255),
                    thickness=2
                )
            cv2.imshow("image",image)
            if cv2.waitKey(10) & 0xFF==ord('q'):
                break
            time.sleep(10)
            
            

