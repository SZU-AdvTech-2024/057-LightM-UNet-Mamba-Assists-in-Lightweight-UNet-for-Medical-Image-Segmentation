from __future__ import annotations

import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import netron

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode

from mamba_ssm import Mamba


"""
编码和解码块的残差卷积均只有一个卷积
激活函数均在参差连接之后
遵守下采样和上采样均在卷积之后
"""

def get_dwconv_layer(
        spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
        bias: bool = False
):
    """
    定义了一个深度可分离卷积层，包括深度卷积和逐点卷积。
    @param spatial_dims: 空间维度（1, 2 或 3）
    @param in_channels: 输入通道数
    @param out_channels: 输出通道数
    @param kernel_size: 卷积核大小，默认为 3
    @param stride: 步长，默认为 1
    @param bias: 是否使用偏置，默认为 False
    @return: 一个顺序执行的复合层，首先深度卷积其次逐点卷积
    """
    depth_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels,
                             strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels)
    point_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels,
                             strides=stride, kernel_size=1, bias=bias, conv_only=True, groups=1)
    return torch.nn.Sequential(depth_conv, point_conv)


"""
    Mamba层定义，这里实际上是文章中提到的RVM层
"""


class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)

        self.mamba = Mamba(
            d_model=input_dim,  # 输入特征的通道数
            d_state=d_state,  # 状态向量大小（越大则状态缓存的信息越多，越能捕获长距离依赖，但计算越复杂）
            d_conv=d_conv,  # 局部卷积的卷积核大小
            expand=expand,  # 块的输出特征图的通道数会是输入特征图的expand倍
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)  # 将输入数据转换为 float32 类型
        B, C = x.shape[:2]  # 获取特征的batch以及通道数
        assert C == self.input_dim  # 确保输入通道数与定义的通道数一致
        n_tokens = x.shape[2:].numel()  # 计算特征图的总像素数，作为序列的长度
        img_dims = x.shape[2:]  # 获取图像的尺寸
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)  # 将特征图展平为二维张量也就是从B，C，n_tokens变为B，n_tokens，C
        x_norm = self.norm(x_flat)  # 对特征图进行归一化
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat  # 使用Mamba模块进行特征提取
        x_mamba = self.norm(x_mamba)  # 对输出进行归一化
        x_mamba = self.proj(x_mamba)  # 将输出映射为所需的输出通道，这里是一个线性层
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)  # 将输出张量重新shaped为B，C，*img_dims
        return out


"""
SPM层
Bright J
"""


class SPMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)

        self.mamba = Mamba(
            d_model=input_dim // 4,  # 输入特征的通道数
            d_state=d_state,  # 状态向量大小（越大则状态缓存的信息越多，越能捕获长距离依赖，但计算越复杂）
            d_conv=d_conv,  # 局部卷积的卷积核大小
            expand=expand,  # 块的输出特征图的通道数会是输入特征图的expand倍
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):  # x属于B C D H W
        # print('using SPMlayer')
        if x.dtype == torch.float16:
            x = x.type(torch.float32)  # 将输入数据转换为 float32 类型
        B, C = x.shape[:2]  # 获取特征的batch以及通道数
        assert C == self.input_dim  # 确保输入通道数与定义的通道数一致
        n_tokens = x.shape[2:].numel()  # 计算每一个特征图的总像素数，作为每一个序列的长度
        img_dims = x.shape[2:]  # 获取图像的尺寸(H W)
        # print('x:', x.shape)
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)  # 将特征图展平为二维张量也就是从B，C，n_tokens变为B，n_tokens，C
        # print('x_flat:', x_flat.shape)
        x_norm = self.norm(x_flat)  # 对特征图进行归一化
        # print('x_norm:', x_norm.shape)
        x_chunks = torch.chunk(x_norm, 4, dim=2)  # 将特征图切分为4个子块(按照dim=2也就是C这个通道维度)
        # print('x_chunks:', len(x_chunks))
        x_mamba = []  # 存储mamba处理过后的变量
        for x_chunk in x_chunks:  # 对四个子块分别进行mamba处理
            # print('x_chunk:', len(x_chunk[0]), len(x_chunk[0][0]), self.skip_scale.shape)
            x_chunk_mamba = self.mamba(x_chunk) + self.skip_scale * x_chunk  # 使用Mamba模块进行特征提取，同时加上残差
            x_mamba = torch.cat([torch.Tensor(x_mamba).to(x_chunk_mamba.device), x_chunk_mamba], dim=2)
        # print('x_mamba:', x_mamba.shape)
        x_mamba = self.norm(x_mamba)  # 对输出进行归一化
        x_mamba = self.proj(x_mamba)  # 将输出映射为所需的输出通道，这里是一个线性层
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)  # 将输出张量重新shaped为B，C，*img_dims
        # print('out:', out.shape)
        return out

"""
池化层，用以最大池化和平均池化的和吗，以及降采样一倍
"""
class PoolingAdd(nn.Module):
    def __init__(self, spatial_dims: int, stride: int = 1):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride) if spatial_dims == 2 else nn.MaxPool3d(kernel_size=stride, stride=stride)
        self.avg_pool = nn.AvgPool2d(kernel_size=stride, stride=stride) if spatial_dims == 2 else nn.AvgPool3d(kernel_size=stride, stride=stride)

    def forward(self, x):
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)
        return max_out + avg_out


"""
根据步长来确定是否需要进行池化
"""


def get_mamba_layer(spatial_dims: int,
                    in_channels: int,
                    out_channels: int,
                    stride: int = 1):
    """
    获取一个Mamba（SPM）层
    @param spatial_dims: 数据维数
    @param in_channels: 输入通道
    @param out_channels: 输出通道
    @param stride: 池化步长，也就是图像的Size缩小的倍数
    @return: 特征图（输入通道数可以是翻倍，图像size可能缩小一倍）
    """
    mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels)
    if stride > 1:  # 步长不为1则添加池化层
        return nn.Sequential(mamba_layer, PoolingAdd(spatial_dims=spatial_dims, stride=stride))
    return mamba_layer


"""
包含残差连接的SPM层
"""


class ResMambaBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,  # 空间维度（2 或 3）
            in_channels: int,  # 输入通道数
            norm: tuple | str,  # 归一化类型和参数
            kernel_size: int = 3,  # 卷积核大小
            act: tuple | str = ("RELU", {"inplace": True}),  # 激活函数类型和参数
    ) -> None:
        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")  # 判断卷积核大小是否为奇数

        self.norm = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv = get_mamba_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels)

    def forward(self, x):
        identity = x  # 临时存储输入数据
        x = self.norm(x)  # 对输入数据进行归一化
        x = self.conv(x)  # 使用第一个Mamba层
        x += identity  # 将输入与输出相加，完成残差连接，完成残差块
        x = self.act(x)  # 激活

        return x


"""
一个上采样残差块，包含一个深度可分离卷积层，没有经过激活函数以及插值阶段
"""
class ResUpBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,  # 空间维度（1, 2 或 3）
            in_channels: int,   # 输入通道数
            norm: tuple | str,  # 归一化类型和参数
            kernel_size: int = 3,   # 卷积核大小
            act: tuple | str = ("RELU", {"inplace": True}), # 激活函数类型和参数
    ) -> None:
        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")
        # 定义两个归一化层和激活函数
        self.norm = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv = get_dwconv_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.in_channels = in_channels

    def forward(self, x):
        identity = x
        # print(x.shape[1], self.in_channels)
        # assert x.shape[1] == self.in_channels
        x = self.norm(x)
        x = self.conv(x) + self.skip_scale * identity
        x = self.act(x)
        return x


class LightMUNet(nn.Module):

    def __init__(
            self,
            spatial_dims: int = 3,  # 空间维度（1, 2 或 3）
            init_filters: int = 8,  # 初始特征数，初始的卷积核的数目
            in_channels: int = 1,  # 输入通道数
            out_channels: int = 2,  # 输出通道数
            dropout_prob: float | None = None,  # dropout概率
            act: tuple | str = ("RELU", {"inplace": True}),  # 激活函数类型和参数
            norm: tuple | str = ("GROUP", {"num_groups": 8}),  # 归一化类型和参数
            norm_name: str = "",  # 归一化名称
            num_groups: int = 8,  # 分组数
            use_conv_final: bool = True,  # 是否使用最后一层卷积
            blocks_down: tuple = (2, 4, 4, 8),  # 下采样残差块数，原本是只有3个层次，但是最后一个4表示有4个RVM层作为瓶颈层编码器
            blocks_up: tuple = (2, 2, 2),  # 上采样残差块数
            upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,  # 上采样方式，默认为NONTRAINABLE（不可训练和更新的）
    ):
        super().__init__()

        if spatial_dims not in (2, 3):  # 判断空间维度是否为2或3
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)  # 激活函数
        if norm_name:  # 如果norm_name不为空，则使用norm_name作为norm
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})  # 使用分组归一化
        self.norm = norm  # 归一化
        self.upsample_mode = UpsampleMode(upsample_mode)  # 上采样方式
        self.use_conv_final = use_conv_final  # 是否使用最后一层卷积
        self.convInit = get_dwconv_layer(spatial_dims, in_channels, init_filters)  # 初始卷积层
        self.down_layers = self._make_down_layers()  # 构建下采样层
        self.up_layers, self.up_samples = self._make_up_layers()  # 构建上采样层和上采样采样层
        self.conv_final = self._make_final_conv(out_channels)  # 最后一层卷积
        # dropout
        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        """
        构建下采样层（downsampling layers）。
        该方法根据配置参数构建一系列下采样层，每个下采样层包含一个下采样模块
        （downsample_mamba）和多个残差块（ResMambaBlock）。
        """
        down_layers = nn.ModuleList()
        # 循环遍历blocks_down列表，构建下采样层
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):  # 遍历blocks_down列表：blocks_down: tuple = (1, 2, 2, 4)
            layer_in_channels = filters * 2 ** i  # 计算经过当前层的输出通道数，通道数每次经过一次下采样层就翻倍
            # print(i, self.init_filters, layer_in_channels)
            # 下采样层，通道数翻倍，特征图size减少一半。如果是第一层不进行下采样，此处Identity()做法是一个恒等变换，输入就是输出，相当于占位
            # downsample_mamba = get_mamba_layer(spatial_dims, layer_in_channels, layer_in_channels * 2, stride=2)

            # 总体的一个编码器构建，这里设置为先卷积再下采样
            down_layer = nn.Sequential(
                # 每一个Encode Block都包含item个ResMambaBlock，每个ResMambaBlock包含两个RVM块，这里不进行下采样
                *[ResMambaBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)],
                # 除了第一层之外每一层的深度卷积之外，都进行一步下采样
                get_mamba_layer(spatial_dims, layer_in_channels, layer_in_channels * 2, stride=2) if i < len(blocks_down) - 1 else nn.Identity(),
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        # 构建上采样层
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,  # blocks_up: tuple = (1, 1, 1)
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)  # 3层
        for i in range(n_up):  # 解码器除了DWConv外有3层
            sample_in_channels = filters * 2 ** (n_up - i)  # 计算经过当前层的输出通道数，通道数每次经过一次上采样层就减少一半
            # 创建一个包含blocks_up[i]个ResUpBlock的Sequential对象
            up_layers.append(
                nn.Sequential(
                    *[
                        ResUpBlock(spatial_dims, sample_in_channels, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])  # 每个ResUpBlock包含blocks_up[i]个ResUpBlock
                    ]
                )
            )
            # 通道数减半，size翻倍（get_upsample_layer默认是2倍）
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        # 构建最后一层卷积层
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_dwconv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:

        x = self.convInit(x)  # 初始卷积,上面的深度可分离卷积

        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []  # 中间特征图

        for down in self.down_layers:
            x = down(x)
            # print("每一层下采样之后x的形状:", x.shape)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        # 解码部分，处理编码结果和中间特征图并返回最终输出
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = upl(x) + down_x[i + 1]  # 跳跃同层级连接
            x = up(x)

        if self.use_conv_final:
            x = self.conv_final(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()  # 将编码层每一层的输出反转，方便后续残差连接

        x = self.decode(x, down_x)
        return x


if __name__ == '__main__':
    '''
    输出网络结构
    1.修改了编码层中，卷积和池化顺序。
    2.修改了解码层中，卷积和上采样顺序。
    3.修改了池化层为两个池化层的和。
    4.修改了编码器和解码器残差块的卷积数目（2-》1）
    '''
    # spmlayer = SPMLayer(
    #     input_dim=32,
    #     output_dim=64,
    # ).to(device='cuda:0')
    # print('spm:', spmlayer(torch.randn(1, 32, 16, 16, 16).to(device='cuda:0')).shape)

    '展示LightM-Unet结构'
    net = LightMUNet(init_filters=32).to(device='cuda:0')
    # torch.onnx.export(net, torch.randn(1, 1, 64, 64, 64).type(torch.float32).to(device='cuda:1'), 'LightMUnet_raw.onnx')
    # netron.save("LightMUnet_raw.onnx")
    print(net)
    print('done')

# !PYTHONWARNINGS="ignore" python LightMUNet_No_ResMamba.py
