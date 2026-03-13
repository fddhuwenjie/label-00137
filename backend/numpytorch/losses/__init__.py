"""
损失函数模块

包含:
- MSELoss: 均方误差损失
- L1Loss: 平均绝对误差损失
- SmoothL1Loss: 平滑 L1 损失 (Huber Loss)
- CrossEntropyLoss: 交叉熵损失 (带 Softmax)
- BinaryCrossEntropyLoss: 二元交叉熵损失
- BCEWithLogitsLoss: 带 Logits 的二元交叉熵损失
- NLLLoss: 负对数似然损失
"""

from .mse import MSELoss, L1Loss, SmoothL1Loss
from .cross_entropy import CrossEntropyLoss, BinaryCrossEntropyLoss, BCEWithLogitsLoss, NLLLoss

__all__ = [
    'MSELoss',
    'L1Loss',
    'SmoothL1Loss',
    'CrossEntropyLoss',
    'BinaryCrossEntropyLoss',
    'BCEWithLogitsLoss',
    'NLLLoss',
]
