import os
import sys

import cv2
import numpy as np
 
def click_test(self):
    # 图片路径
    img = cv2.imread('/home/wenzihao/Desktop/WandCalibration/imageCollect/empty/cam1/0-1709712857753782849.jpeg')
    a = []
    b = []
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            a.append(x)
            b.append(y)
            cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", img)
            print(x,y)
    cv2.namedWindow("image",cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    if cv2.waitKey(0) & 0xFF == ord('q'): 
        exit(0)
    # _,R,T=cv2.solvePnP(objp,corners,mtx,dist)
    # print('所求结果：')
    # print("旋转向量",R)
    # print("平移向量",T)

def click_table(
        cam_0_param,
        image_path,
        table_config
    ):
    # 获取 世界坐标系的四个点
    width,length,height=table_config['width'],table_config['length'],table_config['height']
    quadrant_1=[width/2,height/2,height]
    quadrant_2=[-width/2,height/2,height]
    quadrant_3=[-width/2,-height/2,height]
    quadrant_4=[width/2,-height/2,height]
    objectPoints=np.array([quadrant_1,quadrant_2,quadrant_3,quadrant_4])
    # 获取 图像中的四个点
    img = cv2.imread(image_path)
    a = []
    b = []
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            a.append(x)
            b.append(y)
            cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", img)
            # print(x,y)
    cv2.namedWindow("image",cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    if cv2.waitKey(0) & 0xFF == ord('q'): 
        cv2.destroyAllWindows() 
    imagePoints=np.array([a,b]).T.astype(np.float32)
    # 获取 相机参数
    K=np.array(cam_0_param['K'])
    dist=np.array(cam_0_param['dist'])
    # 解 PnP
    _,R_vec,T_vec=cv2.solvePnP(
        objectPoints=objectPoints,
        imagePoints=imagePoints,
        cameraMatrix=K,
        distCoeffs=dist
    )
    R_mat = cv2.Rodrigues(R_vec)[0]
    return R_mat,np.squeeze(T_vec)
    
    
 

if  __name__=="__main__":
    click_test()