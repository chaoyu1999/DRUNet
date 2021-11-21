# DRUNet
- 论文：基于全局和局部残差图像预测的红外小型无人机目标检测（Infrared Small UAV Target Detection Based on Residual Image Prediction via Global and Local）
- 论文地址：[https://ieeexplore.ieee.org/document/9452107]
# ![avatar](https://img-blog.csdnimg.cn/79e6cd279d96409b98463c4023b7f3cf.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzY2MzQ2Mw==,size_16,color_FFFFFF,t_70#pic_center)
# 项目使用方法
1.自行准备数据，放到```data```目录中，已给出示例  
2.执行```train.py```开始训练  
3.训练的模型保存在```model_data```目录中  
4.执行```predict.py```进行测试，测试的结果在```data/test/predict```中  

# 环境
- Anaconda + windows10 + pytorch 1.9 + cuda 11 以及其它必须的库
# 其它声明
- 由于论文作者没有开源，本项目为本人参考作者论文进行的复现，可能存在未知问题或者BUG，一切以作者论文为准。
- 本项目提供一个预训练模型用于测试，配置好环境后可直接执行 ```predict.py```，在```data/test/predict```中查看效果。
