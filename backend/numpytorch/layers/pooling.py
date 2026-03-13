"""
池化层实现

包含:
- MaxPool2d: 最大池化
- AvgPool2d: 平均池化

池化操作降低特征图的空间维度，减少计算量并提供一定的平移不变性。
"""

import numpy as np
from typing import List, Tuple, Union
from ..tensor import Tensor


class MaxPool2d:
    """
    2D 最大池化层
    
    Args:
        kernel_size: 池化窗口大小
        stride: 步长，默认等于 kernel_size
        padding: 填充
    
    反向传播:
        梯度只传递到前向传播时的最大值位置
    """
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], 
                 stride: int = None, padding: int = 0):
        
        if isinstance(kernel_size, int):
            self.kernel_h = self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        
        # 默认步长等于 kernel_size
        if stride is None:
            self.stride = self.kernel_h
        else:
            self.stride = stride
        
        self.padding = padding
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入 Tensor (N, C, H, W)
        
        Returns:
            输出 Tensor (N, C, out_h, out_w)
        """
        N, C, H, W = x.shape
        
        # 添加 padding
        if self.padding > 0:
            x_padded = np.pad(x.data, 
                            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                            mode='constant', constant_values=float('-inf'))
        else:
            x_padded = x.data
        
        _, _, H_padded, W_padded = x_padded.shape
        
        # 计算输出尺寸
        out_h = (H_padded - self.kernel_h) // self.stride + 1
        out_w = (W_padded - self.kernel_w) // self.stride + 1
        
        # 重塑为池化窗口
        # 将输入分割成不重叠的块
        output = np.zeros((N, C, out_h, out_w))
        max_indices = np.zeros((N, C, out_h, out_w, 2), dtype=np.int32)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.kernel_h
                w_start = j * self.stride
                w_end = w_start + self.kernel_w
                
                # 提取池化窗口
                window = x_padded[:, :, h_start:h_end, w_start:w_end]
                
                # 找到最大值
                output[:, :, i, j] = window.max(axis=(2, 3))
                
                # 记录最大值位置（在窗口内的相对位置）
                window_flat = window.reshape(N, C, -1)
                max_idx = window_flat.argmax(axis=2)
                max_indices[:, :, i, j, 0] = max_idx // self.kernel_w
                max_indices[:, :, i, j, 1] = max_idx % self.kernel_w
        
        result = Tensor(output, requires_grad=x.requires_grad)
        
        if result.requires_grad:
            result._parents = [x]
            
            # 保存用于反向传播
            input_shape = x.shape
            stride = self.stride
            kernel_h = self.kernel_h
            kernel_w = self.kernel_w
            padding = self.padding
            max_idx_saved = max_indices.copy()
            
            def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                N, C, out_h, out_w = grad.shape
                _, _, H, W = input_shape
                
                # 创建梯度数组
                if padding > 0:
                    grad_x = np.zeros((N, C, H + 2 * padding, W + 2 * padding))
                else:
                    grad_x = np.zeros(input_shape)
                
                # 将梯度传递到最大值位置
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride
                        w_start = j * stride
                        
                        # 获取最大值在窗口内的位置
                        for n in range(N):
                            for c in range(C):
                                hi = max_idx_saved[n, c, i, j, 0]
                                wi = max_idx_saved[n, c, i, j, 1]
                                grad_x[n, c, h_start + hi, w_start + wi] += grad[n, c, i, j]
                
                # 移除 padding
                if padding > 0:
                    grad_x = grad_x[:, :, padding:-padding, padding:-padding]
                
                return [grad_x]
            
            result.grad_fn = grad_fn
        
        return result
    
    def parameters(self) -> List[Tensor]:
        """池化层没有可训练参数"""
        return []
    
    def zero_grad(self):
        """无操作"""
        pass
    
    def __repr__(self) -> str:
        return f"MaxPool2d(kernel_size=({self.kernel_h}, {self.kernel_w}), stride={self.stride})"


class AvgPool2d:
    """
    2D 平均池化层
    
    Args:
        kernel_size: 池化窗口大小
        stride: 步长，默认等于 kernel_size
        padding: 填充
    
    反向传播:
        梯度平均分配到池化窗口内的所有位置
    """
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], 
                 stride: int = None, padding: int = 0):
        
        if isinstance(kernel_size, int):
            self.kernel_h = self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        
        if stride is None:
            self.stride = self.kernel_h
        else:
            self.stride = stride
        
        self.padding = padding
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入 Tensor (N, C, H, W)
        
        Returns:
            输出 Tensor (N, C, out_h, out_w)
        """
        N, C, H, W = x.shape
        
        # 添加 padding
        if self.padding > 0:
            x_padded = np.pad(x.data, 
                            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                            mode='constant', constant_values=0)
        else:
            x_padded = x.data
        
        _, _, H_padded, W_padded = x_padded.shape
        
        # 计算输出尺寸
        out_h = (H_padded - self.kernel_h) // self.stride + 1
        out_w = (W_padded - self.kernel_w) // self.stride + 1
        
        # 计算平均值
        output = np.zeros((N, C, out_h, out_w))
        pool_size = self.kernel_h * self.kernel_w
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.kernel_h
                w_start = j * self.stride
                w_end = w_start + self.kernel_w
                
                window = x_padded[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = window.mean(axis=(2, 3))
        
        result = Tensor(output, requires_grad=x.requires_grad)
        
        if result.requires_grad:
            result._parents = [x]
            
            input_shape = x.shape
            stride = self.stride
            kernel_h = self.kernel_h
            kernel_w = self.kernel_w
            padding = self.padding
            
            def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                N, C, out_h, out_w = grad.shape
                _, _, H, W = input_shape
                pool_size = kernel_h * kernel_w
                
                # 创建梯度数组
                if padding > 0:
                    grad_x = np.zeros((N, C, H + 2 * padding, W + 2 * padding))
                else:
                    grad_x = np.zeros(input_shape)
                
                # 梯度平均分配到池化窗口
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride
                        h_end = h_start + kernel_h
                        w_start = j * stride
                        w_end = w_start + kernel_w
                        
                        grad_x[:, :, h_start:h_end, w_start:w_end] += \
                            grad[:, :, i:i+1, j:j+1] / pool_size
                
                # 移除 padding
                if padding > 0:
                    grad_x = grad_x[:, :, padding:-padding, padding:-padding]
                
                return [grad_x]
            
            result.grad_fn = grad_fn
        
        return result
    
    def parameters(self) -> List[Tensor]:
        """池化层没有可训练参数"""
        return []
    
    def zero_grad(self):
        """无操作"""
        pass
    
    def __repr__(self) -> str:
        return f"AvgPool2d(kernel_size=({self.kernel_h}, {self.kernel_w}), stride={self.stride})"


class GlobalAvgPool2d:
    """
    全局平均池化
    
    将整个特征图池化为单个值
    """
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入 Tensor (N, C, H, W)
        
        Returns:
            输出 Tensor (N, C, 1, 1)
        """
        output = x.data.mean(axis=(2, 3), keepdims=True)
        result = Tensor(output, requires_grad=x.requires_grad)
        
        if result.requires_grad:
            result._parents = [x]
            input_shape = x.shape
            
            def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                N, C, H, W = input_shape
                pool_size = H * W
                grad_x = np.broadcast_to(grad / pool_size, input_shape).copy()
                return [grad_x]
            
            result.grad_fn = grad_fn
        
        return result
    
    def parameters(self) -> List[Tensor]:
        return []
    
    def zero_grad(self):
        pass
    
    def __repr__(self) -> str:
        return "GlobalAvgPool2d()"
