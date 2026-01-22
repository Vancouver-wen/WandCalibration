import os
import glob
import base64
import io
import json
from math import log

import cv2
import yaml
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from easydict import EasyDict
from natsort import natsorted
from loguru import logger
from markdown_strings import header, table, code_block,image,bold

class CalibrationQuality(object):
    def __init__(self,n_bins=10):
        # config={
        #     'board_type': 'chessboard',  # or 'charuco'
        #     'board_size': [7, 10],  # width, height in squares
        #     'square_size': 15.0,  # mm
        #     'marker_length': 0.05,
        #     'marker_separation': 0.01,
        #     'n_bins': 10,
        # }
        self.n_bins=n_bins
        # self.config = EasyDict(config)

        self.corners_list = []  # list of detected corners
        self.x_bins = np.zeros(self.n_bins)
        self.y_bins = np.zeros(self.n_bins)
        self.size_bins = np.zeros(self.n_bins)
        self.skew_bins = np.zeros(self.n_bins)
        self.xy_heatmap = np.zeros((self.n_bins,self.n_bins))
        self.quality_score = 0.0

        self.prev_coverage = { 'x': 0, 'y': 0, 'size': 0, 'skew': 0 }
        
        self.setup_gui()

    def setup_gui(self):
        # Visualizations
        self.fig_xy, self.ax_xy = plt.subplots()
        self.fig_size, self.ax_size = plt.subplots()
        self.fig_skew, self.ax_skew = plt.subplots()
        
    # 计算凸包面积
    def compute_convex_hull_area(self,points):
        if len(points) < 3:
            return 0
        hull = ConvexHull(points)
        return hull.volume  # 对于2D，是面积

    # 计算最小外接矩形的长宽比
    def compute_min_rect_aspect_ratio(self,points):
        if len(points) < 3:
            return 1.0
        rect = cv2.minAreaRect(points)
        width, height = rect[1]
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1.0
        return aspect_ratio

    # 映射长宽比到[0,1]
    def map_aspect_ratio_to_01(self,aspect_ratio, min_ar=0.2, max_ar=5.0):
        if aspect_ratio < min_ar:
            aspect_ratio = min_ar
        elif aspect_ratio > max_ar:
            aspect_ratio = max_ar
        log_ar = log(aspect_ratio)
        log_min = log(min_ar)
        log_max = log(max_ar)
        return (log_ar - log_min) / (log_max - log_min)

    def compute_coverages(self, new_corners=None, img_shape=None, temp=False):
        # 如果temp，只计算添加new_corners后的临时覆盖
        x_bins = self.x_bins.copy() if not temp else np.zeros(self.n_bins)
        y_bins = self.y_bins.copy() if not temp else np.zeros(self.n_bins)
        size_bins = self.size_bins.copy() if not temp else np.zeros(self.n_bins)
        skew_bins = self.skew_bins.copy() if not temp else np.zeros(self.n_bins)
        xy_heatmap = self.xy_heatmap.copy() if not temp else np.zeros((self.n_bins, self.n_bins))

        corners_to_process = self.corners_list + [new_corners] if new_corners is not None else self.corners_list

        for corners in corners_to_process:
            if corners is None:
                continue
            points = corners.reshape(-1, 2)
            h, w = img_shape[:2] if img_shape else (480, 640)  # 默认大小，实际应从图像获取

            # X coverage
            x_norm = points[:, 0] / w
            x_idx = np.floor(x_norm * self.n_bins).astype(int).clip(0, self.n_bins - 1)
            x_bins[x_idx] += 1

            # Y coverage
            y_norm = points[:, 1] / h
            y_idx = np.floor(y_norm * self.n_bins).astype(int).clip(0, self.n_bins - 1)
            y_bins[y_idx] += 1

            # XY heatmap
            for xi, yi in zip(x_idx, y_idx):
                xy_heatmap[yi, xi] += 1  # 注意y是行，x是列

            # Size coverage
            area = self.compute_convex_hull_area(points)
            norm_area = area / (h * w)
            size_idx = np.floor(norm_area * self.n_bins).astype(int).clip(0, self.n_bins - 1)
            size_bins[size_idx] += 1

            # Skew coverage
            ar = self.compute_min_rect_aspect_ratio(points)
            mapped_ar = self.map_aspect_ratio_to_01(ar)
            skew_idx = np.floor(mapped_ar * self.n_bins).astype(int).clip(0, self.n_bins - 1)
            skew_bins[skew_idx] += 1

        x_cov = np.sum(x_bins > 0) / self.n_bins
        y_cov = np.sum(y_bins > 0) / self.n_bins
        size_cov = np.sum(size_bins > 0) / self.n_bins
        skew_cov = np.sum(skew_bins > 0) / self.n_bins

        if not temp:
            self.x_bins = x_bins
            self.y_bins = y_bins
            self.size_bins = size_bins
            self.skew_bins = skew_bins
            self.xy_heatmap = xy_heatmap
            self.prev_coverage = {'x': x_cov, 'y': y_cov, 'size': size_cov, 'skew': skew_cov}

        return x_cov, y_cov, size_cov, skew_cov

    def update_metrics(self,img_shape):
        x_cov, y_cov, size_cov, skew_cov = self.compute_coverages(img_shape=img_shape)
        self.quality_score = (x_cov + y_cov + size_cov + skew_cov) / 4
        logger.info(f"X Score: {x_cov * 100}")
        logger.info(f"Y Score: {y_cov * 100}")
        logger.info(f"Size Score: {size_cov * 100}")
        logger.info(f"Skew Score: {skew_cov * 100}")
        logger.info(f"Quality Score: {self.quality_score:.2f}")
        
        # 更新可视化
        self.ax_xy.clear()
        self.ax_xy.imshow(self.xy_heatmap, cmap='hot', interpolation='nearest')
        self.ax_xy.set_title("XY Heatmap")
        buf = io.BytesIO()
        self.fig_xy.savefig(buf,format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        xy_img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        self.ax_size.clear()
        self.ax_size.bar(range(self.n_bins), self.size_bins)
        self.ax_size.set_title("Size Distribution")
        buf = io.BytesIO()
        self.fig_size.savefig(buf,format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        size_img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        self.ax_skew.clear()
        self.ax_skew.bar(range(self.n_bins), self.skew_bins)
        self.ax_skew.set_title("Skew Distribution")
        buf = io.BytesIO()
        self.fig_skew.savefig(buf,format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        skew_img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        report=""
        report+=(header("camera intrinsic calibration quality report",3)+'\n')
        report+=(f"X Score: {x_cov * 100}"+'\n')
        report+=(f"Y Score: {y_cov * 100}"+'\n')
        report+=(f"Size Score: {size_cov * 100}"+'\n')
        report+=(f"Skew Score: {skew_cov * 100}"+'\n')
        report+=(f"Quality Score: {self.quality_score:.2f}"+'\n')
        report+=(bold(f"XY Heatmap Distribution:")+'\n')
        report+=(image('','data:image/png;base64'+','+xy_img_base64)+'\n')
        report+=(bold(f"Size Heatmap Distribution:")+'\n')
        report+=(image('','data:image/png;base64'+','+size_img_base64)+'\n')
        report+=(bold(f"Skew Heatmap Distribution:")+'\n')
        report+=(image('','data:image/png;base64'+','+skew_img_base64)+'\n')
        
        return report

    def evaluate_and_suggest(self):
        suggestions = []
        if np.any(self.x_bins == 0):
            suggestions.append("Need more images covering left/right edges.")
        if np.any(self.y_bins == 0):
            suggestions.append("Need more images covering top/bottom edges.")
        if np.any(self.size_bins == 0):
            low_size = np.where(self.size_bins == 0)[0] < self.n_bins / 2
            suggestions.append("Need more close-up (large) images." if not low_size.all() else "Need more distant (small) images.")
        if np.any(self.skew_bins == 0):
            low_skew = np.where(self.skew_bins == 0)[0] < self.n_bins / 2
            suggestions.append("Need more tilted images." if not low_skew.all() else "Need more frontal images.")
        
        msg = "; ".join(suggestions) if suggestions else "Data quality is good."
        logger.info("Suggestions: "+msg)
        
    def evaluate_quality(self,all_corners,img_shape):
        self.corners_list=all_corners
        report=self.update_metrics(img_shape)
        self.evaluate_and_suggest()
        return report

if __name__ == "__main__":
    app = CalibrationQuality()
    app.evaluate_quality()
    
"""
https://pypi.org/project/markdown-strings/

pip install markdown_strings

from markdown_strings import header, table, code_block

with open("mark_down.md", 'w', encoding="utf8") as file:
    file.write(header("一级目录", 1) + "\n")
    file.write("这是一个文本\n")
    file.write(code_block("import pandas as pd\npd.DataFrame()", "python") + "\n")
    file.write(table([["列1", "1-1", "1-2", "1-3"],
                      ["列2", "2-1", "2-2", "2-3"],
                      ["列3", "3-1", "3-2", "3-3"],
                      ["列4", "4-1", "4-2", "4-3"]]))

"""