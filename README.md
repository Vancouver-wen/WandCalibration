<p align="center">
<img src="assets/挥杆标定图标.png" style="height: 13em" alt="Kawi the Wand-Calibration" />
</p>

<div align="center">

 |[English](https://github.com/Vancouver-wen/WandCalibration) | [中文简体](assets/README_CN.md) |
</div>

---
<p align="center">
<b>Code and Data</b> For SenseTime IRDC Intelligent Sports Team 2024 Patent 
</br>
<a href="https://github.com/Vancouver-wen/WandCalibration"> Wand-Calibration: Calibrate Multi-Camera-System With 3 Marker Wand
</a>
    </br>
    </br>
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Python-3.9-1f425f.svg?color=purple">
    </a>
    <a href="https://copyright.princeton.edu/policy">
        <img alt="License" src="https://img.shields.io/badge/License-AVGP-blue">
    </a>
    <a href="https://badge.fury.io/py/swebench">
        <img src="https://badge.fury.io/py/swebench.svg">
    </a>
</p>

https://github.com/user-attachments/assets/2ed1ab6d-a460-4aef-8a22-444395619660

## 📰 News
* **[Jul. 26, 2024]**: Our Wand-Calibration has been used in the [Table-Tennis](https://h.xinhuaxmt.com/vh512/share/12129448) and [Archery](https://mp.weixin.qq.com/s/ZxIvB2N_dKBc4UrcW5A73A) events at the [Paris Olympics 🔗](https://olympics.com/en/paris-2024)!
* **[Jun. 15, 2024]**: Wand-Calibration has been used in our business collaboration with [Snow51](https://snowhowchina.com/cn/snow-51/)! ([bilibili 🔗](https://www.bilibili.com/video/BV1avJVeKEFL))
* **[Jun. 12, 2024]**: We have completed tests in various scenarios and released the [Usage Method](https://github.com/Vancouver-wen/WandCalibration) for Wand Calibration! ([bilibili 🔗](https://www.bilibili.com/video/BV1HQgcebEx8))
* **[May. 16, 2024]**: The initial development of the [Wand-Calibration](https://www.bilibili.com/video/BV13rJVeuE1L) was completed!

## 👋 Overview
Wand Calibration is a tool used for multi-camera joint calibration, which achieves the joint calibration of multiple cameras by swinging a calibration wand. Given a set of captured calibration images, this tool can return precise camera intrinsic and extrinsic parameters. This is particularly important for applications such as large-format high-precision positioning and measurement, scene stitching, and 3D human pose estimation.

![GIF展示](assets/animation.gif)


## 🚀 Set Up
The Wand Calibration code can be run on Windows, Ubuntu, and Mac.
You first need to have an [Anaconda](https://www.anaconda.com/) Python environment, and then follow these steps:
```bash
conda create -n wandcalibration python=3.9
conda activate wandcalibration
conda install pyqt==5.12.3 --verbose
pip install -r requirements.txt --verbose
python main.py --config "config file path" 
# default config path in config/cfg_uni.yaml
```

## 💽 Usage
> [!WARNING]
> 1. The collected images must be frame-synchronized.
>
> 2. Bundle Adjustment is very sensitive to the initial values of the intrinsic parameters. The accuracy of the intrinsic parameters directly affects the upper limit of bundle adjustment.
>
> 3. If you wish to construct your own calibration data, please follow the steps outlined in [**_bluePrint/Material.md_**](https://github.com/Vancouver-wen/WandCalibration/blob/main/bluePrint/Material.md).


Use the prepared intrinsic calibration data and extrinsic calibration data with the following command:
```bash
cd WandCalibration
# download file from https://drive.google.com/file/d/196Ow0GzzVFBvj4z0CCTwVlLGJvAHSYbq/view?usp=sharing
unzip imageCollect.zip
```
The well-organized directory structure should look like this:
```bash
|-- WandCalibration # root path
    |-- config
    |-- ...
    |-- intrinsicParameter
    |-- extrinsicParameter
    |-- imageCollect
    |   |-- board
    |   |   |-- cam1
    |   |   |   |-- image1.[jpg png]
    |   |   |   |-- ...
    |   |   |   |-- imagek.[jpg png]
    |   |-- empty
    |   |   |-- cam1
    |   |   |   |-- image1.[jpg png]
    |   |   |   |-- ...
    |   |   |   |-- imagek.[jpg png]
    |   |-- pole
    |   |   |-- cam1
    |   |   |   |-- image1.[jpg png]
    |   |   |   |-- ...
    |   |   |   |-- imagek.[jpg png]
    |   |-- wand
    |   |   |-- cam1
    |   |   |   |-- image1.[jpg png]
    |   |   |   |-- ...
    |   |   |   |-- imagek.[jpg png]
```

## ⬇️ Downloads
| Datasets | Google Drive | Baidu Netdisk |
| - | - | - |
| Indoor Tabletennis | [imageCollect.zip](https://drive.google.com/file/d/196Ow0GzzVFBvj4z0CCTwVlLGJvAHSYbq/view?usp=sharing) | [imageCollect](https://pan.baidu.com/s/1SihJdx6WulFQqCobCZn3_w?pwd=vwys) |

## 🍎 Issues
Welcome to provide your valuable suggestions!

## 💫 Contributions
We would love to hear from the broader CV, Machine Learning, and Software Engineering research communities, and we welcome any contributions, pull requests, or issues!
To do so, please either file a new pull request or issue and fill in the corresponding templates accordingly. We'll be sure to follow up shortly!

Contact person: [Zihao Wen](https://github.com/Vancouver-wen) and [Terry Liu](https://github.com/TerryLiu007) (Email: 1052951572@qq.com, sdjnltr@gmail.com).

## ✍️ Citation
If you find our work helpful, please use the following citations.
```
@software{repository:wand-calibration,
  author = {ZihaoWen, TerryLiu},
  title = {Wand Calibration},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Vancouver-wen/WandCalibration}}
}
```

## 🪪 License
AVGP V3. Check `LICENSE` File.
