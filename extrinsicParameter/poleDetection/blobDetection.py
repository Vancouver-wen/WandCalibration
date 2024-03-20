import time
import os

import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm
from joblib import Parallel,delayed

class SimpleBlobDetection():
    """ 
    SimpleBlobDetector从图像中提取blobs的算法流程如下:
    根据阈值步距“thresholdStep”递增,从最小阈值minThreshold“(包含)到最大阈值maxThreshold(排除)计算几个阈值,第一个阈值minThreshold,第二个是minThreshold+ thresholdStep,…以此类推。将这些阈值分别应用于源图像转换为几张二值图像。
    通过findContours算子从每幅二值图像中提取连通分量并计算它们中心位置。
    由团块之间的最小距离minDistBetweenBlobs参数控制。将几个二值图像的团块中心坐标进行分组。闭合中心形成一组。
    从组中,估计斑点的最终中心和它们的半径,并返回点的位置和大小。
    最后对返回的blob执行特征过滤:
    1. 颜色过滤:使用blobColor = 0提取亮色斑点,使用blobColor = 255提取暗色斑点。将二值化图像斑点中心的灰度值和blobColor比较 。如果它们不一致,则将该斑点过滤掉。
    2. 面积过滤:提取面积在minArea(包含)和maxArea(不包含)之间的blob。
    3. 圆度过滤:提取的圆度介于minCircularity(包含)和maxCircularity(不包含)之间的Blob。
    4. 惯性比过滤:提取惯量介于minInertiaRatio(包含)和maxInertiaRatio(不包含)之间的blob
    5. 凸性过滤:提取凸性介于minConvexity(包含)和maxConvexity(不包含)之间的blob。
    """
    def __init__(
            self,
            minThreshold  =   235,
            maxThreshold  =   255,
            thresholdStep = 1,
            filterByColor=True,
            blobColor=255,
            minRepeatability=2,
            minDistBetweenBlobs=10,
            filterByArea  =   True,
            minArea  =   10,
            maxArea=50,
            filterByCircularity  =   True,
            minCircularity  =   0.8,
            filterByConvexity  =   True,
            minConvexity  =   0.4,
            filterByInertia  =   True,
            minInertiaRatio  =   0.1,
            color="white"
        ) -> None:
        """
        float thresholdStep
        float minThreshold
        float maxThreshold
        size_t minRepeatability
        float minDistBetweenBlobs
        bool filterByColor
        uchar blobColor
        bool filterByArea
        float minArea, maxArea
        bool filterByCircularity
        float minCircularity, maxCircularity
        bool filterByInertia
        float minInertiaRatio, maxInertiaRatio
        bool filterByConvexity
        float minConvexity, maxConvexity
        """
        # Setup SimpleBlobDetector parameters.
        self.params  =  cv2.SimpleBlobDetector_Params()
        # Change thresholds
        self.params.minThreshold  =   minThreshold
        self.params.maxThreshold  =   maxThreshold
        self.params.thresholdStep = thresholdStep
        # Color
        self.params.filterByColor=filterByColor
        self.params.blobColor=blobColor
        # Distance
        self.params.minRepeatability=minRepeatability
        self.params.minDistBetweenBlobs=minDistBetweenBlobs
        # Filter by Area.
        self.params.filterByArea  =  filterByArea 
        self.params.minArea  =  minArea
        self.params.maxArea=maxArea
        # Filter by Circularity -> 几边形
        self.params.filterByCircularity  =  filterByCircularity
        self.params.minCircularity  =  minCircularity 
        # Filter by Convexity -> 凸度
        self.params.filterByConvexity  =  filterByConvexity 
        self.params.minConvexity  = minConvexity 
        # Filter by Inertia -> 椭圆度
        self.params.filterByInertia  =  filterByInertia 
        self.params.minInertiaRatio  =  minInertiaRatio  
        # 创建检测器
        self.detector = cv2.SimpleBlobDetector_create(self.params)
        self.color=color
    def frame_pre_process(self,frame):
        if self.color=="white":
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # print(f"frame.shape: {frame.shape}")
        elif self.color=="red":
            red_channel=frame[:,:,2].astype(np.int64)-frame[:,:,0].astype(np.int64)-frame[:,:,1].astype(np.int64)
            red_channel=np.clip(red_channel,a_min=0,a_max=255).astype(np.uint8)
            frame=red_channel
            # print(f"frame.shape: {frame.shape}")
            # exit(0)
        else:
            support_list=["white","red"]
            raise NotImplementedError(f"we only support {support_list}")
        return frame
    def __call__(self, frame, drawKeypoints=True):
        frame=self.frame_pre_process(frame)
        keypoints = self.detector.detect(frame)
        if drawKeypoints==True:
            with_keypoints = cv2.drawKeypoints(
                frame, 
                keypoints, 
                np.array([]),
                (0,0,255), 
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            return keypoints,with_keypoints
        else:
            return keypoints

def get_cam_list(root_path,cam_num):
    temps=[]
    for cam_index in range(cam_num):
        img_folder=os.path.join(root_path,f'cam{cam_index+1}')
        img_names=natsorted(os.listdir(img_folder))
        img_paths=[os.path.join(img_folder,img_name) for img_name in img_names]
        temps.append(img_paths)
    results=[]
    max_len=min([len(temps[cam_index]) for cam_index in range(cam_num)])
    # print(f'=> 最大长度为:{max_len} .. ')
    for index in range(max_len):
        temp=[temps[cam_index][index] for cam_index in range(cam_num)]
        results.append(temp)
    # print(f'=> 总frame长度为:{len(results)} .. ')
    return results

def get_image_list(frame_path,blobDetector):
    frame=cv2.imread(frame_path)
    keypoints=blobDetector(frame,drawKeypoints=False)
    pts=[keypoint.pt for keypoint in keypoints]
    for pt in pts:
        # import pdb;pdb.set_trace()
        cv2.circle(frame, (int(pt[0]),int(pt[1])), 8, (0,0,255),5)
    with_keypoints = cv2.drawKeypoints(
        frame, 
        keypoints, 
        np.array([]),
        (0,0,255), 
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return keypoints,with_keypoints


if __name__=="__main__":
    # 根据分辨率计算 area
    resolution=(2856,1606)
    minAreaRatio=5.0e-6
    maxAreaRatio=6.5e-4
    minArea=resolution[0]*resolution[1]*minAreaRatio
    maxArea=resolution[0]*resolution[1]*maxAreaRatio

    mySimpleBlobDetection=SimpleBlobDetection(
        minThreshold  =   240,
        maxThreshold  =   255,
        thresholdStep = 1,
        filterByColor=True,
        blobColor=255,
        minRepeatability=2,
        minDistBetweenBlobs=10,
        filterByArea  =   True,
        minArea  =  minArea,
        maxArea=maxArea,
        filterByCircularity  =   True,
        minCircularity  =   0.2,
        filterByConvexity  =   True,
        minConvexity  =   0.2,
        filterByInertia  =   True,
        minInertiaRatio  =   0.1
    )

    root_path="./archery/0306/blob"
    num_cam=4

    # root_path="./wtt/0306/blob"
    # num_cam=6

    frame_lists=get_cam_list(root_path,num_cam)

    for frame_list in frame_lists:
        # time.sleep(0.03)
        temps=Parallel(n_jobs=len(frame_list),backend="threading")(
            delayed(get_image_list)(frame_path,mySimpleBlobDetection)
            for frame_path in frame_list
        )
        temps=list(zip(*temps))
        # import pdb;pdb.set_trace()
        all_keypoints=list(temps[0])
        all_frames=list(temps[1])