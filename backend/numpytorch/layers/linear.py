"""
线性层 (全连接层) 实现

y = x @ W.T + b

梯度计算:
- dW = grad_output.T @ x
- db = sum(grad_output, axis=0)
- dx = grad_output @ W
"""

import numpy as np
from typing import Optional, List
from ..tensor import Tensor


class Linear:
    """
    线性层 (全连接层)
    
    Args:
        in_features: 输入特征数
        out_features: 输出特征数
        bias: 是否使用偏置
    
    Attributes:
        weight: 权重矩阵 (out_features, in_features)
        bias: 偏置向量 (out_features,)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # 使用 Kaiming 初始化
        # std = sqrt(2 / in_features)
        std = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(out_features, in_features) * std,
            requires_grad=True
        )
        
        if bias:
            self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        else:
            self.bias = None
    
    def __call__(self, x: Tensor) -> Tensor:
        """前向传播"""
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入 Tensor (batch_size, in_features) 或 (in_features,)
        
        Returns:
            输出 Tensor (batch_size, out_features) 或 (out_features,)
        """
        # y = x @ W.T + b
        # x: (N, in_features)
        # W: (out_features, in_features)
        # W.T: (in_features, out_features)
        # y: (N, out_features)
        
        batch_input = x.ndim == 2
        if not batch_input:
            x = x.reshape(1, -1)
        
        # 矩阵乘法
        output = x @ self.weight.T
        
        # 添加偏置
        if self.use_bias and self.bias is not None:
            output = output + self.bias
        
        if not batch_input:
            output = output.reshape(-1)
        
        return output
    
    def parameters(self) -> List[Tensor]:
        """返回所有可训练参数"""
        if self.use_bias and self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]
    
    def zero_grad(self):
        """清零梯度"""
        self.weight.zero_grad()
        if self.use_bias and self.bias is not None:
            self.bias.zero_grad()
    
    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"
