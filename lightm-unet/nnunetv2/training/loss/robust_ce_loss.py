import torch
from torch import nn, Tensor
import numpy as np


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    这段代码的主要目的是确保在计算交叉熵损失时，目标标签 target 的形状与网络输出 input 的形状匹配。具体来说：
    维度一致性：如果 target 和 input 的维度数相同，说明 target 可能是一个多通道的张量。
    单通道验证：通过 assert target.shape[1] == 1 确保 target 的第二个维度大小为 1，即 target 只有一个通道。
    去除多余维度：通过 target = target[:, 0] 去掉 target 的第二个维度，使其变成一个一维的张量，这样可以与 input 的形状匹配，便于计算交叉熵损失。
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # 如果target是float
        if target.ndim == input.ndim:   # 判断维度是否相等
            assert target.shape[1] == 1
            target = target[:, 0]   # 如果标签是多个通道的标注则只保留第一个维度，用于计算二类交叉熵损失
        return super().forward(input, target.long())


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()
