"""
NumpyTorch - 基于 NumPy 实现的深度学习框架

用于学习 PyTorch 原理，包含：
- Tensor 类和自动微分引擎
- 神经网络层：Linear, ReLU, Sigmoid, Conv2d, MaxPool2d, AvgPool2d, RNNCell, RNN
- 损失函数：MSELoss, CrossEntropyLoss
- 优化器：SGD
"""

from .tensor import (
    Tensor,
    zeros,
    ones,
    randn,
    rand,
    from_numpy,
    stack,
    concatenate,
)

from . import layers
from . import losses
from . import optim

__version__ = "0.1.0"
__all__ = [
    "Tensor",
    "zeros",
    "ones",
    "randn",
    "rand",
    "from_numpy",
    "stack",
    "concatenate",
    "layers",
    "losses",
    "optim",
]
