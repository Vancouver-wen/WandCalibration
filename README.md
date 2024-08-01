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
1. vis_spread通过颜色来区分可见视野数量
2. vis_reproj_all展示所有重投影的pole,通过颜色展示其reproj_error与mean_pixel_error的比值
3. world pose 使用matplotlib绘制gif图片,使之旋转起来,同时要避免matplotlib内存泄漏
4. 给labelme的enhance mode添加初始化