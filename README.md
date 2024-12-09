<p align="center">
<img src="assets/挥杆标定图标.png" style="height: 13em" alt="Kawi the Wand-Calibration" />
</p>

<div align="center">

 |[English](https://github.com/Vancouver-wen/WandCalibration) | [中文简体](docs/README_CN.md) |
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
* **[Jun. 26, 2024]**: Our Wand-Calibration has been used in the [Table-Tennis](https://h.xinhuaxmt.com/vh512/share/12129448) and [Archery](https://mp.weixin.qq.com/s/ZxIvB2N_dKBc4UrcW5A73A) events at the Paris Olympics!
* **[Jun. 15, 2024]**: Wand-Calibration has been used in our business collaboration with [Snow51](https://snowhowchina.com/cn/snow-51/)! ([bilibili 🔗](https://www.bilibili.com/video/BV1avJVeKEFL))
* **[Jun. 12, 2024]**: We have completed tests in various scenarios and released the [Usage Method](https://github.com/Vancouver-wen/WandCalibration) for Wand Calibration! ([bilibili 🔗](https://www.bilibili.com/video/BV1HQgcebEx8))
* **[May. 16, 2024]**: The initial development of the [Wand-Calibration](https://www.bilibili.com/video/BV13rJVeuE1L) was completed!

##### 注意事项：
1. 本仓库用于校准有共视区域的多相机系统
2. 需保证采集的图像是帧同步的
3. Bundle Adjustment 对内参初值非常敏感,内参的精度直接关系到捆绑调整的上限

##### 环境配置 python=3.9
conda update -n base -c defaults conda
1. conda install pyqt==5.12.3 --verbose
2. pip install -r requirements.txt --verbose

##### 使用方法:
1. 根据配置文件制作T型杆与L型杆
    1. SolidWorks 制作图纸位于 bluePrint/3MarkerWand.STEP
    2. 根据 bluePrint/Material.md 购买其他配件

2. 采集
    1. 内参图像
    2. 空场图片
    3. 采集挥动T型杆图像
    4. 采集静止的L型杆图像
3. 将采集的图片整理成如下结构
    ```
    - imageCollect
        - board
            - cam*
                - [image]
        - empty
            - cam*
                - [image]
        - pole
            - cam*
                - [image]
        - wand
            - cam*
                - [image]
    ```
4. 执行命令
    ```
    python main.py --config "config file path"
    ```


## ⬇️ Downloads
| Datasets | Models |
| - | - |
| [🤗 SWE-bench](https://huggingface.co/datasets/princeton-nlp/SWE-bench) | [🦙 SWE-Llama 13b](https://huggingface.co/princeton-nlp/SWE-Llama-13b) |
| [🤗 "Oracle" Retrieval](https://huggingface.co/datasets/princeton-nlp/SWE-bench_oracle) | [🦙 SWE-Llama 13b (PEFT)](https://huggingface.co/princeton-nlp/SWE-Llama-13b-peft) |
| [🤗 BM25 Retrieval 13K](https://huggingface.co/datasets/princeton-nlp/SWE-bench_bm25_13K) | [🦙 SWE-Llama 7b](https://huggingface.co/princeton-nlp/SWE-Llama-7b) |
| [🤗 BM25 Retrieval 27K](https://huggingface.co/datasets/princeton-nlp/SWE-bench_bm25_27K) | [🦙 SWE-Llama 7b (PEFT)](https://huggingface.co/princeton-nlp/SWE-Llama-7b-peft) |
| [🤗 BM25 Retrieval 40K](https://huggingface.co/datasets/princeton-nlp/SWE-bench_bm25_40K) | |
| [🤗 BM25 Retrieval 50K (Llama tokens)](https://huggingface.co/datasets/princeton-nlp/SWE-bench_bm25_50k_llama)   | |


## ✍️ Citation
If you find our work helpful, please use the following citations.
```
@inproceedings{
    jimenez2024swebench,
    title={{SWE}-bench: Can Language Models Resolve Real-world Github Issues?},
    author={Carlos E Jimenez and John Yang and Alexander Wettig and Shunyu Yao and Kexin Pei and Ofir Press and Karthik R Narasimhan},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=VTF8yNQM66}
}
```

## 🪪 License
AVGP. Check `LICENSE.md`.
