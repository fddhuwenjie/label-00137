"""
优化器实现

包含:
- SGD: 随机梯度下降
- SGD with Momentum: 带动量的 SGD
- Adam: 自适应矩估计优化器
"""

import numpy as np
from typing import List, Optional
from ..tensor import Tensor


class SGD:
    """
    随机梯度下降优化器
    
    更新公式:
        不带动量: param = param - lr * grad
        带动量: 
            v = momentum * v + grad
            param = param - lr * v
        
        带 Nesterov 动量:
            v = momentum * v + grad
            param = param - lr * (momentum * v + grad)
    
    Args:
        params: 待优化的参数列表
        lr: 学习率
        momentum: 动量系数，默认 0
        weight_decay: 权重衰减 (L2 正则化)，默认 0
        nesterov: 是否使用 Nesterov 动量
    """
    
    def __init__(self, params: List[Tensor], lr: float = 0.01, 
                 momentum: float = 0, weight_decay: float = 0,
                 nesterov: bool = False):
        
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # 初始化动量缓存
        self.velocities = [np.zeros_like(p.data) for p in params]
    
    def step(self):
        """
        执行一步参数更新
        """
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # 添加权重衰减
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            if self.momentum != 0:
                # 带动量的更新
                v = self.velocities[i]
                v = self.momentum * v + grad
                self.velocities[i] = v
                
                if self.nesterov:
                    # Nesterov 动量
                    param.data = param.data - self.lr * (self.momentum * v + grad)
                else:
                    param.data = param.data - self.lr * v
            else:
                # 普通 SGD
                param.data = param.data - self.lr * grad
    
    def zero_grad(self):
        """
        清零所有参数的梯度
        """
        for param in self.params:
            param.grad = None
    
    def __repr__(self) -> str:
        return (f"SGD(lr={self.lr}, momentum={self.momentum}, "
                f"weight_decay={self.weight_decay}, nesterov={self.nesterov})")


class Adam:
    """
    Adam 优化器
    
    结合了动量和自适应学习率的优化器。
    
    更新公式:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)
    
    Args:
        params: 待优化的参数列表
        lr: 学习率
        betas: (beta1, beta2) 动量系数
        eps: 数值稳定性常数
        weight_decay: 权重衰减
    """
    
    def __init__(self, params: List[Tensor], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0):
        
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError(f"Invalid betas: {betas}")
        
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # 初始化一阶和二阶矩估计
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        
        # 时间步
        self.t = 0
    
    def step(self):
        """
        执行一步参数更新
        """
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # 添加权重衰减 (AdamW 风格)
            if self.weight_decay != 0:
                param.data = param.data - self.lr * self.weight_decay * param.data
            
            # 更新一阶矩估计
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # 更新二阶矩估计
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # 偏差校正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # 更新参数
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        """
        清零所有参数的梯度
        """
        for param in self.params:
            param.grad = None
    
    def __repr__(self) -> str:
        return (f"Adam(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
                f"eps={self.eps}, weight_decay={self.weight_decay})")


class RMSprop:
    """
    RMSprop 优化器
    
    使用梯度平方的指数移动平均来缩放学习率。
    
    更新公式:
        v = alpha * v + (1 - alpha) * grad^2
        param = param - lr * grad / (sqrt(v) + eps)
    
    Args:
        params: 待优化的参数列表
        lr: 学习率
        alpha: 平滑系数
        eps: 数值稳定性常数
        weight_decay: 权重衰减
        momentum: 动量系数
    """
    
    def __init__(self, params: List[Tensor], lr: float = 0.01,
                 alpha: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0, momentum: float = 0):
        
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        # 初始化梯度平方的移动平均
        self.v = [np.zeros_like(p.data) for p in params]
        
        # 如果使用动量，初始化动量缓存
        if momentum > 0:
            self.buffer = [np.zeros_like(p.data) for p in params]
    
    def step(self):
        """
        执行一步参数更新
        """
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # 添加权重衰减
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # 更新梯度平方的移动平均
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (grad ** 2)
            
            # 计算更新量
            avg = np.sqrt(self.v[i]) + self.eps
            
            if self.momentum > 0:
                self.buffer[i] = self.momentum * self.buffer[i] + grad / avg
                param.data = param.data - self.lr * self.buffer[i]
            else:
                param.data = param.data - self.lr * grad / avg
    
    def zero_grad(self):
        """
        清零所有参数的梯度
        """
        for param in self.params:
            param.grad = None
    
    def __repr__(self) -> str:
        return (f"RMSprop(lr={self.lr}, alpha={self.alpha}, "
                f"eps={self.eps}, weight_decay={self.weight_decay})")
