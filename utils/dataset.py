
import cv2
import os
import glob
from torch.utils.data import Dataset


class Data_Loader(Dataset):
    def __init__(self, data_path):
        # TODO
        # 1. Initialize file paths or a list of file names.
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.jpg'))

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).

        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)

        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        return image, label

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        # 返回训练集大小
        return len(self.imgs_path)

