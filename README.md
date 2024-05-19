# WandCalibration
calibratie multi cameras with 3 marker wand 

#### 注意事项：
1. 本仓库用于校准有共视区域的多相机系统
2. 需保证采集的图像是帧同步的
3. bundle adjustment 对内参初值非常敏感,内参的精度直接关系到捆绑调整的 upper bound
4. 优化的耗时与图片组的数量成正比,图片组的数量尽量不要超过2000

##### 环境配置
1. conda install pyqt==5.12.3
2. pip install -r requirements.txt

###### 使用方法:
1. 根据配置文件制作T型杆与L型杆
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