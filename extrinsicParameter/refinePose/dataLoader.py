import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import math

import cv2
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset,DataLoader

class BoundAdjustmentDataset(Dataset):
    def __init__(self,list_len):
        self.num=list_len

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return idx
    