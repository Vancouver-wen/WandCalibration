import os
import sys
import json

import numpy as np
import cv2
import glob
from natsort import natsorted
from loguru import logger

from extrinsicParameter.poleDetection.blobDetection import get_cam_list
from extrinsicParameter.refinePose.multiViewTriangulate import easy_multi_view_triangulate
from utils.imageConcat import show_multi_imgs