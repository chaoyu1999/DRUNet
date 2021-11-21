import torch
import torch.nn as nn


class InputCov(nn.Module):
    """
    对输入的原始图像进行卷积
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.input_cov = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, stride=1, dilation=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, stride=1, dilation=2)
        )

    def forward(self, initial_data):
        op = self.input_cov(initial_data)  # (batch,C,W,H)
        op_ = torch.add(op, initial_data)
        return op_


class StdCovLocalResBlock(nn.Module):
    """
    使用标准卷积的局部残差模块
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.std_lrb = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        )

    def forward(self, input_map):
        op = self.std_lrb(input_map)  # (abtch,C,W,H)
        add_map = torch.sum(input_map, dim=1)  # (batch_size,W,H)
        add_map__ = torch.div(add_map, input_map.shape[1])
        add_map_ = add_map__.unsqueeze(1)  # （batch,1,W,H）
        op_ = torch.add(op, add_map_)
        return op_


class DilCovLocalResBlock(nn.Module):
    """
    使用空洞卷积的局部残差模块
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dil_lrb = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, stride=1, dilation=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, stride=1, dilation=2)
        )

    def forward(self, input_map):
        op = self.dil_lrb(input_map)  # (abtch,C,W,H)
        add_map = torch.sum(input_map, dim=1)  # (batch_size,W,H)
        add_map__ = torch.div(add_map, input_map.shape[1])
        add_map_ = add_map__.unsqueeze(1)  # （batch,1,W,H）
        op_ = torch.add(op, add_map_)
        return op_


class LeftGlobalResBlock(nn.Module):
    """
    网络左半部分的全局残差模块
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.std_cov = StdCovLocalResBlock(in_channels, out_channels)
        self.dil_cov = DilCovLocalResBlock(out_channels, out_channels)

    def forward(self, down_map):
        std_cov = self.std_cov(down_map)
        dil_cov = self.dil_cov(std_cov)
        add_map = torch.sum(down_map, dim=1)
        add_map__ = torch.div(add_map, down_map.shape[1])
        add_map_ = add_map__.unsqueeze(1)  # （batch,1,W,H）
        dil_ = torch.add(dil_cov, add_map_)
        return dil_


class RightGlobalResBlock(nn.Module):
    """
    网络右半部分的全局残差模块
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.std_cov_1 = StdCovLocalResBlock(in_channels, out_channels)
        self.std_cov_2 = StdCovLocalResBlock(out_channels, out_channels)

    def forward(self, up_map):
        std_cov_1 = self.std_cov_1(up_map)
        std_cov_2 = self.std_cov_2(std_cov_1)
        add_map = torch.sum(up_map, dim=1)
        add_map__ = torch.div(add_map, up_map.shape[1])
        add_map_ = add_map__.unsqueeze(1)  # （batch,1,W,H）
        std_ = torch.add(std_cov_2, add_map_)
        return std_


class Down(nn.Module):
    """
    下采样模块
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=0, stride=2)

    def forward(self, input_map):
        return self.down(input_map)


class Up(nn.Module):
    """
    上采样模块
    """

    def __init__(self):
        super().__init__()
        self.up = nn.PixelShuffle(2)

    def forward(self, input_map, skip_map):
        up = self.up(input_map)
        up_map = torch.sum(up, dim=1)
        up_map__ = torch.div(up_map, input_map.shape[1])
        up_map_ = up_map__.unsqueeze(1)  # （batch,1,W,H）
        skip_ = torch.add(skip_map, up_map_)

        return skip_


class OutputCov(nn.Module):
    """
    输出
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cov = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, input_map):
        return self.cov(input_map)
