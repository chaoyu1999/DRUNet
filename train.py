from utils.dataset import Data_Loader
from torch import optim
from model.DRUNet_model import *
import torch.utils.data as tud
import os
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

# Tensorboard可视化，存储loss
resWriter = SummaryWriter(comment='--res')


def train_net(net, device, data_path, pass_epoch=0, epochs=300, batch_size=1, lr=0.0001):
    # 加载训练集
    data_dataset = Data_Loader(data_path)
    train_loader = tud.DataLoader(dataset=data_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # 定义Loss
    criterion = nn.MSELoss()
    # best_loss，初始化为正无穷
    best_loss = float('inf')
    for epoch in range(pass_epoch, epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        rImg, rLabel, rPre = None, None, None
        for index, (image, label) in enumerate(train_loader):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            if index + 1 == len(train_loader):
                rImg, rLabel, rPre = image, label, pred
            # 计算loss
            loss = criterion(pred, label)
            print('--epoch:', epoch, '/', epochs, '--number:', index, '/', len(train_loader),
                  '--Loss/train:',
                  loss.item())

            # 更新参数
            loss.backward()
            optimizer.step()

            # value
            if (index + 1) % len(train_loader) == len(train_loader) // 2 or (index + 1) % len(train_loader) == 0:
                loss_val = val_net(net, 'data/val/')
                # 保存模型
                if loss_val < best_loss:
                    best_loss = loss_val
                    torch.save(net.state_dict(),
                               'model_data/best_model-' + str(epoch) + '-' + str(index + 1) + '.pth')

        # 可视化存储中间训练结果
        resWriter.add_image('res', make_grid(torch.cat([rImg, rLabel, rPre]), nrow=3, pad_value=1, padding=2,
                                             normalize=True), global_step=epoch)


@torch.no_grad()
def val_net(net, val_data_path, batch_size=1):
    criterion = nn.MSELoss()
    # 测试模式
    net.eval()
    # average mse
    avg_mse = 0
    # 加载训练集
    data_dataset = Data_Loader(val_data_path)
    val_loader = tud.DataLoader(dataset=data_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    for index, (image, label) in enumerate(val_loader):
        torch.cuda.empty_cache()
        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        # 使用网络参数，输出预测结果
        pred = net(image)
        mse = criterion(pred, label).item()
        avg_mse += (mse / len(val_loader))
        print('--num:', index, '/', len(val_loader), '--val/MSE:', mse)
    print('----average mse:', avg_mse)
    return avg_mse


def get_file_list(file_path):
    """
    获取当前文件夹中最近创建的模型路径
    """
    dir_list = os.listdir(file_path)
    if not dir_list:
        return None
    else:
        # 注意，这里使用lambda表达式，将文件按照最后修改时间顺序升序排列
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        return 'model_data/' + dir_list[-1]


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络
    net = DRUNet()
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型
    model_path = get_file_list('model_data/')
    pass_epoch = int(model_path.split('-')[1]) + 1 if model_path else 0  # 从模型读取训练过的epoch
    if model_path:
        net.load_state_dict(torch.load(model_path, map_location=device))
    # 指定训练集地址，开始训练
    data_path = "data/train/"
    train_net(net, device, data_path, pass_epoch)
