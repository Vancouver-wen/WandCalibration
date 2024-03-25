import os
import sys

import numpy as np
import cv2
from joblib import Parallel,delayed

from extrinsicParameter.poleDetection.blobDetection import get_cam_list
from utils.imageConcat import show_multi_imgs

def vis_one_frame(
        pole,
        frame_path
    ):
    frame=cv2.imread(frame_path)
    if pole is None:
        return frame
    else:
        corners,ids=pole
        for corner,id in list(zip(corners,ids)):
            point=corner.astype(np.int32)
            # import pdb;pdb.set_trace()
            frame=cv2.circle(img=frame,center=point,radius=10,color=(0,255,0),thickness=-1)
            cv2.putText(frame,str(id.item()),point,cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),2)
        return frame
def vis_each_pole(
        pole_list,
        frame_list
    ):
    assert len(pole_list)==len(frame_list),"len(pole_list) != len(frame_list)"
    frames_with_keypoints=Parallel(n_jobs=len(pole_list),backend="threading")(
        delayed(vis_one_frame)(pole,frame_path)
        for pole,frame_path in list(zip(pole_list,frame_list))
    )
    image=show_multi_imgs(
        scale=1,
        imglist=frames_with_keypoints,
        order=(int(len(frames_with_keypoints)/3+0.99),3),
        border=2
    )
    return image

def vis_pole(
        cam_num,
        image_path,
        pole_lists,
        show=False
    ):
    debug_path=os.path.join(image_path,"vis_poles")
    debug_num=0
    if not os.path.exists(debug_path):
        os.mkdir(debug_path)
    
    frame_lists=get_cam_list(image_path,cam_num)
    assert len(pole_lists)==len(frame_lists),"len(pole_lists) != len(frame_lists)"
    if show:
        cv2.namedWindow("image",cv2.WINDOW_GUI_NORMAL)
    for pole_list,frame_list in list(zip(pole_lists,frame_lists)):
        if show or debug_num<30:
            image=vis_each_pole(pole_list,frame_list)
        if debug_num<30:
            cv2.imwrite(os.path.join(debug_path,f"{debug_num}.jpg"),image)
            debug_num+=1
        if show:
            cv2.imshow("image",image)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
    if show:
        cv2.destroyAllWindows()


if __name__=="__main__":
    pass