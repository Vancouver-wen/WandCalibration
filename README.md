# WandCalibration
calibratie multi cameras with 3 marker wand 

##### 注意事项：
1. 本仓库用于校准有共视区域的多相机系统
2. 需保证采集的图像是帧同步的
3. bundle adjustment 对内参初值非常敏感,内参的精度直接关系到捆绑调整的 upper bound

##### 环境配置 python=3.9
conda update -n base -c defaults conda
1. conda install pyqt==5.12.3 --verbose
2. pip install -r requirements.txt --verbose

##### 使用方法:
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

##### TODO List:
1. 使用dataloader控制单次反向传播的数量,进行batch size的实验,稳定不同机器上的实验结果。迭代次数就设置为800次dataloader
2. 增强labelme功能的鲁棒性(无标注文件,空标注,标注数量不够),使用torch进行优化,结合pnp与icp的优势
3. SGD优化器 + cos预热退火