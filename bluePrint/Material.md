挥杆标定法

代码位置:
193:
/mnt/locals/osg/volume1/wenzihao/optitrack

标准化器材展示
使用方法视频
https://ones.ainewera.com/wiki/api/wiki/editor/JNwe8qUX/XBUyqCL3/resources/f1ER88dfAgUgcS4XL_r0FJWpUisNO8msVDjNAJ03y6o.mp4?token=W.ofEhDdPUGjj6qIP_ZMxJ7W-NVd232OfKx0mpjVaJfqsDN6fQBUrxHfVW_2T8FTk
细节展示




标准化采购器材（已采购）：




【淘宝】https://m.tb.cn/h.g0stCLRSFBMTDmu?tk=9j17WrtP4Ov CZ3456 「304不锈钢杯头内六角穿墙螺丝长螺杆加长螺栓对穿丝杆M2M3M4M5M20」 点击链接直接打开 或者 淘宝搜索直接打开 6份(共60个)                                                                                                                                                                                     
              
【淘宝】https://m.tb.cn/h.g0sucglrsnQa7lO?tk=hazdWrtOdtw CZ8908 「304不锈钢螺母六角螺帽螺丝帽大全M1M2M3M4M5M6M8M10M12M14M16M33」 点击链接直接打开 或者 淘宝搜索直接打开 6份(共60个)                                                                                                                                                                                     
              
【淘宝】https://m.tb.cn/h.g0ZvdbBvwwY9wVN?tk=RkbuWrtman3 CZ3460 「LED彩色灯泡低压12V24V36V节能红色7W9W灯笼户外防水婚庆E27螺口」 点击链接直接打开 或者 淘宝搜索直接打开 6个                                                                                                                                                                                              
              
【淘宝】https://m.tb.cn/h.g0ZvdbBvwwY9wVN?tk=RkbuWrtman3 CZ3460 「LED彩色灯泡低压12V24V36V节能红色7W9W灯笼户外防水婚庆E27螺口」 点击链接直接打开 或者 淘宝搜索直接打开 6个                                                                                                                                                                                              
              
【淘宝】https://m.tb.cn/h.g0sE0FAvgHtmwol?tk=eOWuWrtNn6g CZ0016 「灯头卡式全牙半牙e27台灯螺口灯座div吊灯落地灯具配件带M10牙杆」 12个                                                                                                                                                                                                                                    
              
【淘宝】https://m.tb.cn/h.gZFqj9kEJVGXrPR?tk=2ItOWI9KhNu HU7632 「电源线扣护线套机箱电源线固定线扣塑料卡线扣电线保护套线卡6N-4」 点击链接直接打开 或者 淘宝搜索直接打开 1份（100粒）                                                                                                                                                                                      
              
【淘宝】https://m.tb.cn/h.gb9eEfYECpm1GOH?tk=KFTgWI9EZL7 ZH4920 「环保尼龙方管堵头三层盖封塑料堵头圆型内塞橡胶凳子静音家居垫脚」 点击链接直接打开 或者 淘宝搜索直接打开 6份（共60个）                                                                                                                                                                                     
              
【淘宝】https://m.tb.cn/h.gb9eEfYECpm1GOH?tk=KFTgWI9EZL7 ZH4920 「环保尼龙方管堵头三层盖封塑料堵头圆型内塞橡胶凳子静音家居垫脚」 点击链接直接打开 或者 淘宝搜索直接打开 6份（共60个）                                                                                                                                                                                     
              
【淘宝】https://m.tb.cn/h.gckDCHG?tk=mF4VWtjEv9A CZ0001 「18650锂电池组3.7V7.4V12V14.8V24V大容量灯具电池音箱电源长条款」 点击链接直接打开 或者 淘宝搜索直接打开 2个                                                                                                                                                                                                     
              
【淘宝】https://m.tb.cn/h.g1ttHI2?tk=u3m8WtQYHDK CZ0015 「电子秤开关配件通用带线免焊三档船型小台灯饮水机电子称电源开关」 点击链接直接打开 或者 淘宝搜索直接打开 5个                                                                                                                                                                                                     
              
【淘宝】https://m.tb.cn/h.gZXV4KR?tk=fmIiWrtpETK CZ3456 「304不锈钢定制铁板铝板折弯焊接攻牙激光切割定做机箱钣金件加工」 点击链接直接打开 或者 淘宝搜索直接打开 2套(每套中包含一个T型包含一个L型) 联系该店铺的客服，只需要对接暗号"原丽杰"店家就知道图纸了（附件的两个图纸已经和商家沟通过了，可以制作） 价格是350一根，总共采购4套（2个T型杆，2个L型杆） 一定要记得备注黑色喷漆！
见如上两个附件

室外实验结果：
实验流程：
场地: mhub室外或者archery室外
采集 aruco cube 标定图案 -> 现有的标定方法 
采集 charuco board 标定图案 -> multical
采集 3 marker wand 标定图案 -> 自研挥杆标定
采集 篮球传球(击地/横传/投篮)图案 通过重投影误差比较三种标定方法的精度
实验结果：
使用灰色展示篮球检测模型的效果(包括检测框 检测分数 以及检测框中心点)
使用蓝色blue展示aruco cube相机标定结果对[1]的结果进行三角化再重投影的结果
使用绿色green展示charuco board相机标定结果对[1]的结果进行三角化再重投影的结果
使用红色red展示3 marker wand相机标定结果对[1]的结果进行三角化再重投影的结果
视频分辨率为5K, 网页上有压缩, 可以下载后观看
ROI区域
{"total_aruco": 15.59795636718507, "total_charuco": 3.3424906619587693, "total_wand": 3.731393881937239}


https://ones.ainewera.com/wiki/api/wiki/editor/JNwe8qUX/XBUyqCL3/resources/wls5WNUOlwluckHteZE09S3FYJMsLCDH4C80MhCNtAY.mp4?token=W.ofEhDdPUGjj6qIP_ZMxJ7W-NVd232OfKx0mpjVaJfqsDN6fQBUrxHfVW_2T8FTk

非ROI区域
{"total_aruco": 16.830193564250788, "total_charuco": 4.248197651655297, "total_wand": 4.434552642424041}


https://ones.ainewera.com/wiki/api/wiki/editor/JNwe8qUX/XBUyqCL3/resources/_bE7XvTrsWsZWlMerVDEUQpNz5lbEea63cu7d3dkXoM.mp4?token=W.ofEhDdPUGjj6qIP_ZMxJ7W-NVd232OfKx0mpjVaJfqsDN6fQBUrxHfVW_2T8FTk

室内实验结果：
实验流程：
场地: wtt四相机
采集 aruco cube 标定图案 -> 现有的标定方法 
采集 charuco board 标定图案 -> multical
采集 3 marker wand 标定图案 -> 自研挥杆标定
采集 乒乓球对打图案(也就是ROI区域)通过重投影误差比较三种标定方法的精度
采集 手抛乒乓球的图案(也就是边缘区域)通过重投影误差比较三种标定方法的精度
实验结果：
使用灰色展示乒乓球检测模型的效果(包括检测框 检测分数 以及检测框中心点)
使用蓝色blue展示aruco cube相机标定结果对[1]的结果进行三角化再重投影的结果
使用绿色green展示charuco board相机标定结果对[1]的结果进行三角化再重投影的结果
使用红色red展示3 marker wand相机标定结果对[1]的结果进行三角化再重投影的结果
视频分辨率为5K, 网页上有压缩, 可以下载后观看
ROI区域精度
{"total_aruco": 5.921376439636026, "total_charuco": 5.923809543677511, "total_wand": 5.938364765472477}


https://ones.ainewera.com/wiki/api/wiki/editor/JNwe8qUX/XBUyqCL3/resources/dnXyl4fSaJn_UxWkcUFhHeqmFO6As2JMvSexEGYxuzE.mp4?token=W.ofEhDdPUGjj6qIP_ZMxJ7W-NVd232OfKx0mpjVaJfqsDN6fQBUrxHfVW_2T8FTk
非ROI区域精度
{"total_aruco": 2.1804336483482474, "total_charuco": 0.43761370240340364, "total_wand": 1.4009662054794023}


https://ones.ainewera.com/wiki/api/wiki/editor/JNwe8qUX/XBUyqCL3/resources/rw3YuBSraIMHECmD0MtkP6J3gowDK3j0odWJN9z7CJY.mp4?token=W.ofEhDdPUGjj6qIP_ZMxJ7W-NVd232OfKx0mpjVaJfqsDN6fQBUrxHfVW_2T8FTk
室外实验：
室外实验场地: 上海体育大学
灯泡的外壳为白色，红色的纯度与亮度都不够 -> 改进灯泡

更新室外硬件需求（已采购）：
更新原因：之前的买的红色灯泡存在两个问题
1. 外壳为白色，在阳光下会影响红色的显著性
2. 功率不够大，在低曝光低gain值下不显著
【淘宝】https://m.tb.cn/h.5CNWxK7j5WfmB4A?tk=No7JWoCdo6X CZ0015 「德国进口3-12v5a可调直流电源220v转24V3a36V电机水泵风扇调速调」 点击链接直接打开 或者 淘宝搜索直接打开 1个                                      

【淘宝】https://m.tb.cn/h.5CNh9n94LYaoPnc?tk=bs9FWoC6AiR CZ0015 「LED彩色灯泡220V低压12V24V36V E27螺口红色灯笼户外装饰婚庆彩灯」 点击链接直接打开 或者 淘宝搜索直接打开 8个24V12W                               

【淘宝】https://m.tb.cn/h.5yaxkJYTS7vHhVU?tk=3zXTWoChLQv CZ3460 「彩色陶瓷灯头吸顶 纯铜螺口爬宠乌龟箱UVA UVB灯口E27耐高温灯座」 点击链接直接打开 或者 淘宝搜索直接打开 8个                                      

【淘宝】https://m.tb.cn/h.5CxIGNdiMsKTsnV?tk=SvyeWLU5KHp CZ0012 「加粗铜0.75/1平方DC5525监控电源线 2.1公母头12V10A电池dc接头线」 点击链接直接打开 或者 淘宝搜索直接打开 两条公头 两条母头 3m 0.5平方 5.5*2.5规格


室外硬件需求（已采购）：
【淘宝】https://m.tb.cn/h.5uaTdqeoOOTghdJ?tk=j2alWPdUAIH HU9046 「LED迷你灯泡小球泡1W微型直流5V12V24V化妆镜前广告字灯具装饰灯」 点击链接直接打开 或者 淘宝搜索直接打开 -> 红色 24V 8个            

【淘宝】https://m.tb.cn/h.5D8lSoYF7wVoGb3?tk=CAcBWnEUNzN HU0854 「LED半圆球光源低压12伏广告牌220V化妆灯牙杆M10直径60mm5W球泡」 点击链接直接打开 或者 淘宝搜索直接打开 -> 高亮红色 24V 60mm直径 8个

【淘宝】https://m.tb.cn/h.5tz7CE7Ggn9MXCW?tk=qlNBWPdZ8Il HU0854 「松木圆木棒实木棍挂衣杆圆木条DIY手工模型材料建筑装修长木棍」 点击链接直接打开 或者 淘宝搜索直接打开   -> 8根 2cm直径             

【淘宝】https://m.tb.cn/h.5GlBrg9nS6RYRJf?tk=ZhRrWPdYLnH CZ0000 「摩托车射灯夹具前叉抱箍前杠辅助扩展支架铝合金护杠活动管夹可调」 点击链接直接打开 或者 淘宝搜索直接打开  -> 8个 2cm内径           

【淘宝】https://m.tb.cn/h.5uqzklrrE6jlFZn?tk=acLcWlUHYLH CZ0001 「不锈钢内外丝转换接头M5M6M8M10M12M14M20螺纹加厚直通」 点击链接直接打开 或者 淘宝搜索直接打开 -> 螺丝转接头 8个                   

【淘宝】https://m.tb.cn/h.5GOYja0mglPXgRz?tk=i8yHWPde29c CZ3452 「十字卡扣双U型管卡镀锌钢管连接件U型卡抱箍大棚猪场产床固定管卡」 点击链接直接打开 或者 淘宝搜索直接打开 -> 直径2cm 4个            

【淘宝】https://m.tb.cn/h.5x4l1SbZ4yhyrtl?tk=j9MPWn9PMpI CZ0002 「5V转9V12V24VUSB升压线模块 UDP 数控USB彩屏电源升降压恒压恒流」 点击链接直接打开 或者 淘宝搜索直接打开 -> 1个                     

挥杆标定T型杆与L型杆规格：



室内实验结果：
标定精度 : 1.72 pixel error 
耗时 : 30 min
T型杆
d1=250mm  d2=510mm

T型杆检测结果

室内硬件需求（已采购）：
室内实验补充购买器材
【淘宝】https://m.tb.cn/h.5uaTdqeoOOTghdJ?tk=j2alWPdUAIH HU9046 「LED迷你灯泡小球泡1W微型直流5V12V24V化妆镜前广告字灯具装饰灯」 点击链接直接打开 或者 淘宝搜索直接打开 ->                          

【淘宝】https://m.tb.cn/h.5unwKyiHsQQqZOe?tk=wZWQWlWCaUZ MF1643 「黑色自动手摇自喷漆哑光亮光亚光磨砂黑漆汽车防锈不掉色专用油漆」 点击链接直接打开 或者 淘宝搜索直接打开  -> 黑色喷漆一瓶           

【淘宝】https://m.tb.cn/h.5uqzklrrE6jlFZn?tk=acLcWlUHYLH CZ0001 「不锈钢内外丝转换接头M5M6M8M10M12M14M20螺纹加厚直通」 点击链接直接打开 或者 淘宝搜索直接打开 -> 螺丝转接头 6个                    

【淘宝】https://m.tb.cn/h.5uJfSYUlJDpjj6I?tk=Scw2Wl5bca9 MF6563 「1007双头镀锡线24AWG导线电子线连接线红黑色跳线8cm10cm15cm20cm」 点击链接直接打开 或者 淘宝搜索直接打开 -> 补充线材 1捆(20cm*100条)


购买白色圆形marker
关键词：魔豆灯泡
【淘宝】https://m.tb.cn/h.5t3pycDC4VYMv1D?tk=P3rzWQLftpe MF6563 「户外防水串灯复古室外防水挂灯创意小灯泡露台庭院装饰灯花园树灯」 点击链接直接打开 或者 淘宝搜索直接打开       

【淘宝】https://m.tb.cn/h.5uah1050XvqvE6H?tk=ooDtWPd5T0M CZ8908 「欧普照明led灯泡e27/e14大小螺口超亮家用台灯球泡暖黄光节能灯5W」 点击链接直接打开 或者 淘宝搜索直接打开 太大了

【淘宝】https://m.tb.cn/h.5sC45eK9BwdLL9y?tk=nCTfWQLfvnL CZ8908 「led彩灯闪灯串灯满天星星灯户外露营装饰灯摆摊灯夜市氛围灯卧室」 点击链接直接打开 或者 淘宝搜索直接打开        

【淘宝】https://m.tb.cn/h.5HRlJFuGbXcWBka?tk=FERqWQLf6CT HU9046 「led高亮3w过道灯球泡半圆形灯泡白光暖光柜台灯化妆试衣镜前灯USB」 点击链接直接打开 或者 淘宝搜索直接打开       

【淘宝】https://m.tb.cn/h.5uaTdqeoOOTghdJ?tk=j2alWPdUAIH HU9046 「LED迷你灯泡小球泡1W微型直流5V12V24V化妆镜前广告字灯具装饰灯」 点击链接直接打开 或者 淘宝搜索直接打开        

【淘宝】https://m.tb.cn/h.5sSPUdHzeNnGjPN?tk=ThSHWQLf2eS CZ0001 「包邮F5发光二极管带电池盒开关DIY自制LED灯5V电压常亮led手工灯」 点击链接直接打开 或者 淘宝搜索直接打开        

原厂标定杆太贵 => 适用木工制作 or 3D打印 or 钣金制作 => 最终选择钣金制作




外径：2cm
【淘宝】https://m.tb.cn/h.5tz7CE7Ggn9MXCW?tk=qlNBWPdZ8Il HU0854 「松木圆木棒实木棍挂衣杆圆木条DIY手工模型材料建筑装修长木棍」 点击链接直接打开 或者 淘宝搜索直接打开   

【淘宝】https://m.tb.cn/h.5GOYja0mglPXgRz?tk=i8yHWPde29c CZ3452 「十字卡扣双U型管卡镀锌钢管连接件U型卡抱箍大棚猪场产床固定管卡」 点击链接直接打开 或者 淘宝搜索直接打开

【淘宝】https://m.tb.cn/h.5GOcEWBZE5bbXzB?tk=kRn1WPdecsP CZ0000 「十字型连接件光轴十字夹连接块锁紧固定块铝制垂直固定夹光轴夹座」 点击链接直接打开 或者 淘宝搜索直接打开

【淘宝】https://m.tb.cn/h.5GlCoqpCuQGMzqA?tk=8iMTWPdbqti MF6563 「手捏单钢丝卡扣金属软管夹小卡子快装环抱箍汽油管弹力固定喉箍夹」 点击链接直接打开 或者 淘宝搜索直接打开

【淘宝】https://m.tb.cn/h.5ut1Y3Kouurw5pB?tk=RxTkWPd0sqN MF1643 「镀铜七字钩自攻螺丝直角钩金色螺丝白锌7字钩相框挂钩螺丝L型螺丝」 点击链接直接打开 或者 淘宝搜索直接打开

【淘宝】https://m.tb.cn/h.5GlBrg9nS6RYRJf?tk=ZhRrWPdYLnH CZ0000 「摩托车射灯夹具前叉抱箍前杠辅助扩展支架铝合金护杠活动管夹可调」 点击链接直接打开 或者 淘宝搜索直接打开

【淘宝】https://m.tb.cn/h.5ut1F1qzWfITLoR?tk=KCSHWPdbotp CZ3458 「304不锈钢手柄喉箍手拧卡箍抱箍管箍水管收紧箍圈卡扣固定夹管卡」 点击链接直接打开 或者 淘宝搜索直接打开 

粘合剂
【淘宝】https://m.tb.cn/h.5GOCQd3dhPdGFL3?tk=ttEyWPdAmus CA6496 「双面胶高粘度透明固定墙面车用无痕防水强力纳米3m亚克力胶两面胶布耐高温不留痕万能魔力防滑贴强力粘胶胶带(@ ~)」 点击链接直接打开 或者 淘宝搜索直接打开


实验场地：
复用mhub即可