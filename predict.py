import glob
import time

import numpy as np
import cv2
import os
from model.DRUNet_model import *


def get_file_list(file_path):
    """
    获取当前文件夹中最近创建的模型路径
    """
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        # 注意，这里使用lambda表达式，将文件按照最后修改时间顺序升序排列
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        return dir_list[-1]


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = DRUNet()
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    model_path = 'model_data/' + get_file_list('model_data/')
    net.load_state_dict(torch.load(model_path, map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob('data/test/image/*jpg')
    # 遍历素有图片
    st = time.time()
    for index, test_path in enumerate(tests_path):
        # 保存结果地址
        save_res_path = test_path.replace('image', 'predict')
        # 读取图片
        img_ = cv2.imread(test_path, 0).astype(np.float32)
        # 转为batch为1，通道为1
        img = img_.reshape(1, 1, img_.shape[0], img_.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        print('正在预测' + str(index + 1), '/', len(tests_path))
        sub = np.abs(img_ - pred).astype('uint8')  # 目标 = 原图 -预测的背景图
        # 二值化
        _, sub = cv2.threshold(sub, 100, 255, cv2.THRESH_BINARY)
        # 后处理 -- 闭运算
        kernel = np.ones((3, 3), np.uint8)
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, kernel)
        # 找目标框轮廓
        contours, hierarchy = cv2.findContours(sub, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        # 读原图
        img_ = cv2.imread(test_path)
        # 绘制目标框
        for index, contour in enumerate(contours):
            bbox = cv2.boundingRect(contour)
            cv2.rectangle(img_, (bbox[0] - 1, bbox[1] - 1), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          (255, 0, 255), 1)
        # 保存绘制结果
        cv2.imwrite(save_res_path, img_)
    print('Frame/s:', len(tests_path) / (time.time() - st))

