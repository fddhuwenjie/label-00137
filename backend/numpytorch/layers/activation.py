"""
激活函数层实现

包含:
- ReLU: max(0, x)
- Sigmoid: 1 / (1 + exp(-x))
- Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
- LeakyReLU: max(alpha * x, x)
"""

import numpy as np
from typing import List
from ..tensor import Tensor


class ReLU:
    """
    ReLU 激活函数
    
    前向: y = max(0, x)
    反向: dy/dx = 1 if x > 0 else 0
    """
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        result = Tensor(
            np.maximum(0, x.data),
            requires_grad=x.requires_grad
        )
        
        if result.requires_grad:
            result._parents = [x]
            x_data = x.data.copy()
            
            def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                return [grad * (x_data > 0).astype(np.float64)]
            
            result.grad_fn = grad_fn
        
        return result
    
    def parameters(self) -> List[Tensor]:
        """ReLU 没有可训练参数"""
        return []
    
    def zero_grad(self):
        """无操作"""
        pass
    
    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid:
    """
    Sigmoid 激活函数
    
    前向: y = 1 / (1 + exp(-x))
    反向: dy/dx = y * (1 - y)
    """
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """前向传播 - 数值稳定版本"""
        # 数值稳定的实现
        result_data = np.where(
            x.data >= 0,
            1 / (1 + np.exp(-x.data)),
            np.exp(x.data) / (1 + np.exp(x.data))
        )
        
        result = Tensor(result_data, requires_grad=x.requires_grad)
        
        if result.requires_grad:
            result._parents = [x]
            
            def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                return [grad * result_data * (1 - result_data)]
            
            result.grad_fn = grad_fn
        
        return result
    
    def parameters(self) -> List[Tensor]:
        """Sigmoid 没有可训练参数"""
        return []
    
    def zero_grad(self):
        """无操作"""
        pass
    
    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh:
    """
    Tanh 激活函数
    
    前向: y = tanh(x)
    反向: dy/dx = 1 - y^2
    """
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        result_data = np.tanh(x.data)
        result = Tensor(result_data, requires_grad=x.requires_grad)
        
        if result.requires_grad:
            result._parents = [x]
            
            def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                return [grad * (1 - result_data ** 2)]
            
            result.grad_fn = grad_fn
        
        return result
    
    def parameters(self) -> List[Tensor]:
        """Tanh 没有可训练参数"""
        return []
    
    def zero_grad(self):
        """无操作"""
        pass
    
    def __repr__(self) -> str:
        return "Tanh()"


class LeakyReLU:
    """
    Leaky ReLU 激活函数
    
    前向: y = max(alpha * x, x)
    反向: dy/dx = 1 if x > 0 else alpha
    
    Args:
        alpha: 负半轴的斜率，默认 0.01
    """
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        result_data = np.where(x.data > 0, x.data, self.alpha * x.data)
        result = Tensor(result_data, requires_grad=x.requires_grad)
        
        if result.requires_grad:
            result._parents = [x]
            x_data = x.data.copy()
            alpha = self.alpha
            
            def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                mask = np.where(x_data > 0, 1.0, alpha)
                return [grad * mask]
            
            result.grad_fn = grad_fn
        
        return result
    
    def parameters(self) -> List[Tensor]:
        """LeakyReLU 没有可训练参数"""
        return []
    
    def zero_grad(self):
        """无操作"""
        pass
    
    def __repr__(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"


class Softmax:
    """
    Softmax 激活函数
    
    前向: y = exp(x) / sum(exp(x))
    
    Args:
        axis: 计算 softmax 的轴，默认 -1 (最后一个轴)
    """
    
    def __init__(self, axis: int = -1):
        self.axis = axis
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """前向传播 - 数值稳定版本"""
        # 减去最大值以提高数值稳定性
        x_max = np.max(x.data, axis=self.axis, keepdims=True)
        exp_x = np.exp(x.data - x_max)
        result_data = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
        
        result = Tensor(result_data, requires_grad=x.requires_grad)
        
        if result.requires_grad:
            result._parents = [x]
            axis = self.axis
            
            def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                # Softmax 的雅可比矩阵计算
                s = result_data
                grad_x = s * (grad - np.sum(grad * s, axis=axis, keepdims=True))
                return [grad_x]
            
            result.grad_fn = grad_fn
        
        return result
    
    def parameters(self) -> List[Tensor]:
        """Softmax 没有可训练参数"""
        return []
    
    def zero_grad(self):
        """无操作"""
        pass
    
    def __repr__(self) -> str:
        return f"Softmax(axis={self.axis})"
