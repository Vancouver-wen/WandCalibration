<p>
<img src="assets/挥杆标定图标.png" style="height: 13em" alt="Kawi the Wand-Calibration" />
</p>
[English](docs/README_ENGLISH.md) | [中文简体](https://github.com/Vancouver-wen/WandCalibration) |

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
