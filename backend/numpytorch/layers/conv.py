"""
卷积层实现

使用 im2col 技术将卷积运算转换为矩阵乘法，提高计算效率。

im2col 原理:
1. 将输入图像的每个卷积窗口展开为一列
2. 将卷积核展开为一行
3. 进行矩阵乘法得到输出
"""

import numpy as np
from typing import List, Tuple, Union
from ..tensor import Tensor


def im2col(x: np.ndarray, kernel_h: int, kernel_w: int, 
           stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    将图像转换为列矩阵
    
    Args:
        x: 输入数组 (N, C, H, W)
        kernel_h: 卷积核高度
        kernel_w: 卷积核宽度
        stride: 步长
        padding: 填充
    
    Returns:
        col: 列矩阵 (N * out_h * out_w, C * kernel_h * kernel_w)
    """
    N, C, H, W = x.shape
    
    # 添加 padding
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                   mode='constant', constant_values=0)
    
    _, _, H_padded, W_padded = x.shape
    
    out_h = (H_padded - kernel_h) // stride + 1
    out_w = (W_padded - kernel_w) // stride + 1
    
    # 使用循环实现 (更直观，便于理解)
    col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w))
    
    for i in range(kernel_h):
        i_max = i + stride * out_h
        for j in range(kernel_w):
            j_max = j + stride * out_w
            col[:, :, i, j, :, :] = x[:, :, i:i_max:stride, j:j_max:stride]
    
    # 转换形状: (N, C, kh, kw, oh, ow) -> (N * oh * ow, C * kh * kw)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    
    return col


def col2im(col: np.ndarray, input_shape: Tuple, kernel_h: int, kernel_w: int,
           stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    将列矩阵转换回图像格式
    
    Args:
        col: 列矩阵 (N * out_h * out_w, C * kernel_h * kernel_w)
        input_shape: 原始输入形状 (N, C, H, W)
        kernel_h: 卷积核高度
        kernel_w: 卷积核宽度
        stride: 步长
        padding: 填充
    
    Returns:
        x: 图像数组 (N, C, H, W)
    """
    N, C, H, W = input_shape
    
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    
    out_h = (H_padded - kernel_h) // stride + 1
    out_w = (W_padded - kernel_w) // stride + 1
    
    # 转换形状: (N * oh * ow, C * kh * kw) -> (N, oh, ow, C, kh, kw)
    col = col.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
    
    # 创建输出数组
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=col.dtype)
    
    # 累加梯度
    for i in range(kernel_h):
        i_max = i + stride * out_h
        for j in range(kernel_w):
            j_max = j + stride * out_w
            x_padded[:, :, i:i_max:stride, j:j_max:stride] += col[:, :, i, j, :, :]
    
    # 移除 padding
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded


class Conv2d:
    """
    2D 卷积层
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小 (int 或 tuple)
        stride: 步长
        padding: 填充
        bias: 是否使用偏置
    
    Attributes:
        weight: 卷积核 (out_channels, in_channels, kernel_h, kernel_w)
        bias: 偏置 (out_channels,)
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int]], 
                 stride: int = 1, padding: int = 0, bias: bool = True):
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            self.kernel_h = self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        
        # Kaiming 初始化
        fan_in = in_channels * self.kernel_h * self.kernel_w
        std = np.sqrt(2.0 / fan_in)
        
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, self.kernel_h, self.kernel_w) * std,
            requires_grad=True
        )
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None
        
        # 缓存，用于反向传播
        self._col = None
        self._input_shape = None
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入 Tensor (N, C, H, W)
        
        Returns:
            输出 Tensor (N, out_channels, out_h, out_w)
        """
        N, C, H, W = x.shape
        
        # 计算输出尺寸
        out_h = (H + 2 * self.padding - self.kernel_h) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_w) // self.stride + 1
        
        # im2col 转换
        col = im2col(x.data, self.kernel_h, self.kernel_w, self.stride, self.padding)
        # col: (N * out_h * out_w, C * kh * kw)
        
        # 将权重展平: (out_channels, C * kh * kw)
        weight_col = self.weight.data.reshape(self.out_channels, -1)
        
        # 矩阵乘法: (N * out_h * out_w, C * kh * kw) @ (C * kh * kw, out_channels)
        # = (N * out_h * out_w, out_channels)
        output = col @ weight_col.T
        
        # 添加偏置
        if self.use_bias and self.bias is not None:
            output = output + self.bias.data
        
        # 重塑输出: (N, out_h, out_w, out_channels) -> (N, out_channels, out_h, out_w)
        output = output.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        
        result = Tensor(output, requires_grad=x.requires_grad or self.weight.requires_grad)
        
        if result.requires_grad:
            # 保存用于反向传播
            self._col = col
            self._input_shape = x.shape
            
            result._parents = [x, self.weight]
            if self.use_bias and self.bias is not None:
                result._parents.append(self.bias)
            
            # 捕获必要的变量
            col_saved = col.copy()
            input_shape = x.shape
            weight_data = self.weight.data.copy()
            stride = self.stride
            padding = self.padding
            kernel_h = self.kernel_h
            kernel_w = self.kernel_w
            out_channels = self.out_channels
            use_bias = self.use_bias
            
            def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                # grad: (N, out_channels, out_h, out_w)
                N, _, out_h, out_w = grad.shape
                
                # 转换 grad 形状: (N, out_channels, out_h, out_w) -> (N * out_h * out_w, out_channels)
                grad_col = grad.transpose(0, 2, 3, 1).reshape(-1, out_channels)
                
                # 权重梯度: dW = col.T @ grad_col
                # (C * kh * kw, N * out_h * out_w) @ (N * out_h * out_w, out_channels)
                # = (C * kh * kw, out_channels)
                grad_weight = (col_saved.T @ grad_col).T
                grad_weight = grad_weight.reshape(weight_data.shape)
                
                # 输入梯度: dx_col = grad_col @ weight_col
                weight_col = weight_data.reshape(out_channels, -1)
                grad_col_x = grad_col @ weight_col
                # (N * out_h * out_w, C * kh * kw)
                
                # col2im 还原
                grad_x = col2im(grad_col_x, input_shape, kernel_h, kernel_w, stride, padding)
                
                if use_bias:
                    # 偏置梯度
                    grad_bias = grad_col.sum(axis=0)
                    return [grad_x, grad_weight, grad_bias]
                else:
                    return [grad_x, grad_weight]
            
            result.grad_fn = grad_fn
        
        return result
    
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
        return (f"Conv2d({self.in_channels}, {self.out_channels}, "
                f"kernel_size=({self.kernel_h}, {self.kernel_w}), "
                f"stride={self.stride}, padding={self.padding})")
