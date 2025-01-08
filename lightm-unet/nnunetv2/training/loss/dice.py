from typing import Callable

import torch
import time
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn

'''
特点：
通用性：适用于大多数情况，计算方式较为直观。
内存消耗：在某些情况下可能会消耗较多内存，尤其是在处理大批次和高分辨率数据时。
主要步骤：
计算交集、假正类、假负类：
'''


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        """
        计算Dice
        @param x: 模型的预测输出，通常是一个形状为 (batch_size, num_classes, *spatial_dimensions) 的张量
        @param y: 真实标签，通常是一个形状为 (batch_size, *spatial_dimensions) 或 (batch_size, num_classes, *spatial_dimensions) 的张量
        @param loss_mask: 可选参数，用于指定哪些区域需要计算损失，通常是一个形状为 (batch_size, *spatial_dimensions) 的布尔张量。
        @return:
        """
        shp_x = x.shape
        # 如果为 True，则在批量维度（batch dimension）上计算 Dice 损失。
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)   # 计算交集、假正类、假负类和真负类
        # 用于指示是否使用分布式数据并行（Distributed Data Parallel，简称 DDP）进行训练
        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)  # 于在 DDP 训练中收集各个进程的数据，确保全局的 Dice 损失计算和梯度同步。
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)
        # 剪裁 tp 值
        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp, max=None)
        # 计算 Dice
        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        # return -dc    # raw
        return 1 - dc


'''
特点：
内存效率：通过减少中间变量的存储，节省内存。
适用场景：特别适用于处理大批次和高分辨率数据，能够显著减少内存消耗。
主要步骤：
计算交集、假正类、假负类
'''


class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        #return -dc  # 因为Dice越接近于1，因为loss越小越好，将其转化为最小化目标，所以返回-dc
        return 1 - dc


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    计算交集、假正类、假负类和真负类
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here
        # benchmark whether tiling the mask would be faster (torch.tile). It probably is for large batch sizes
        # OK it barely makes a difference but the implementation above is a tiny bit faster + uses less vram
        # (using nnUNetv2_train 998 3d_fullres 0)
        # tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        # fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        # fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        # tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn


"""
自适应区域特定损失
author Bright J
"""


class Adaptive_Region_Specific_TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = True, do_bg: bool = True, smooth: float = 1.,
                 num_region_per_axis=(16, 16, 16), A=0.3, B=0.4):
        """
        num_region_per_axis: 分块的形状 (z, x, y)
        3D num_region_per_axis 形状(z, x, y)
        2D num_region_per_axis 形状(x, y)
        """
        super(Adaptive_Region_Specific_TverskyLoss, self).__init__()
        self.smooth = smooth
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.dim = len(num_region_per_axis)
        assert self.dim in [2, 3], "The num of dim must be 2 or 3."
        if self.dim == 3:
            self.pool = nn.AdaptiveAvgPool3d(
                num_region_per_axis)  # 3D: [batchsize, c, z, x, y]，让输出的shape为(batchsize, c, z, x, y)
        elif self.dim == 2:
            self.pool = nn.AdaptiveAvgPool2d(num_region_per_axis)

        self.A = A
        self.B = B
        self.apply_nonlin = apply_nonlin

    def forward(self, x, y, loss_mask=None):
        # 2D/3D: [batchsize, c, (z,) x, y]
        if self.apply_nonlin is not None:  # 如果要求有非线性激活函数
            x = self.apply_nonlin(x)

        shp_x, shp_y = x.shape, y.shape  # 获得输入和标签的shape
        assert self.dim == (len(shp_x) - 2), "The region size must match the data's size."  # 判断输入的标签是否是指定的3D/2D数据

        if not self.do_bg:  # 如果要求不包含背景，则不考虑背景
            x = x[:, 1:]

        with torch.no_grad():  # 不计算梯度
            if len(shp_x) != len(shp_y):  # 如果输入的标签的shape与输入的不一致，则将标签的维度扩展到与输入一致
                y = y.view((shp_y[0], 1, *shp_y[1:]))  # 将标签的维度扩展到与输入一致例如y的形状从 (2, 4, 4) 转换为 (2, 1, 4, 4)

            # 将y转换为独热编码
            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # 如果只是单类别的标注，y只会有0（背景），1（类别）的数值，此时也已经是独热编码，所以直接赋值
                y_onehot = y
            else:
                gt = y.long()  # 将y转换为long类型
                y_onehot = torch.zeros(shp_x, device=x.device)  # 创建一个与输入x相同的全零张量
                """
                Tensor.scatter_(dim, index, src) 根据索引将特定值填充到张量
                    dim: 指定沿着哪个维度进行填充。
                    index: 包含索引值的张量，这些索引值指定了在 src 中取值的位置。
                    src: 包含要填充的值的张量或标量。
                """
                y_onehot.scatter_(1, gt, 1)  # 将y转换为独热编码

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]  # 移除背景的编码

            # 如果存在loss_mask，则将loss_mask与y_onehot进行逐元素相乘，
            # 为不同的像素或区域分配不同的权重，使它们在损失计算中占据合理的的比重，
            # 允许在计算损失时忽略某些特定的像素或区域
            if loss_mask is not None:
                y_onehot = y_onehot * loss_mask

        # the three in [batchsize, class_num, (z,) x, y]计算交并补
        tp = x * y_onehot
        fp = x * (1 - y_onehot)
        fn = (1 - x) * y_onehot

        # the three in [batchsize, class_num, (num_region_per_axis_z,) num_region_per_axis_x, num_region_per_axis_y]
        region_tp = self.pool(tp)
        region_fp = self.pool(fp)
        region_fn = self.pool(fn)

        if self.batch_dice:  # 如果要求计算batch的dice，则计算batch的dice，也就是将批量的数据都加起来了
            region_tp = region_tp.sum(0)
            region_fp = region_fp.sum(0)
            region_fn = region_fn.sum(0)

        # [(batchsize,) class_num, (num_region_per_axis_z,) num_region_per_axis_x, num_region_per_axis_y]
        alpha = self.A + self.B * (region_fp + self.smooth) / (region_fp + region_fn + self.smooth)
        beta = self.A + self.B * (region_fn + self.smooth) / (region_fp + region_fn + self.smooth)

        # [(batchsize,) class_num, (num_region_per_axis_z,) num_region_per_axis_x, num_region_per_axis_y]
        region_tversky = (region_tp + self.smooth) / (region_tp + alpha * region_fp + beta * region_fn + self.smooth)
        region_tversky = 1 - region_tversky

        # [(batchsize,) class_num]
        # if self.batch_dice:
        #     region_tversky = region_tversky.sum(list(range(1, len(shp_x)-1)))
        # else:
        #     region_tversky = region_tversky.sum(list(range(2, len(shp_x))))

        # print(region_tversky.shape)

        region_tversky = region_tversky.mean()

        return region_tversky


if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1

    pred = torch.rand((2, 3, 320, 320, 320))  # (b, c, x, y, z)的数字，均匀分布在0-1之间
    ref = torch.randint(0, 3, (2, 320, 320, 320))  # (b, x, y, z)的数字，为0, 1, 2三种数字

    dl_old = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    dl_new = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0,
                                         ddp=False)
    adpl = Adaptive_Region_Specific_TverskyLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0)
    start = time.time()
    res_old = dl_old(pred, ref)
    time_length = time.time() - start
    print("old time cost: {}s, and loss: {}".format(time_length, res_old))

    start = time.time()
    res_new = dl_new(pred, ref)
    time_length = time.time() - start
    print("new time cost: {}s, and loss: {}".format(time_length, res_new))

    start = time.time()
    res_adpl = adpl(pred, ref)
    time_length = time.time() - start
    print("adpl time cost: {}s, and loss: {}".format(time_length, res_adpl))
