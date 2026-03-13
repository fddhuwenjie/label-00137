"""
基础运算操作模块

提供独立的运算函数，主要用于内部实现和高级操作。
大部分基础运算已集成在 Tensor 类中，这里提供额外的功能性函数。
"""

import numpy as np
from typing import Tuple, Optional, List
from .tensor import Tensor, _unbroadcast_grad


def relu(x: Tensor) -> Tensor:
    """
    ReLU 激活函数: max(0, x)
    
    前向: y = max(0, x)
    反向: dy/dx = 1 if x > 0 else 0
    """
    result = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
    
    if result.requires_grad:
        result._parents = [x]
        x_data = x.data.copy()
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            return [grad * (x_data > 0).astype(np.float64)]
        
        result.grad_fn = grad_fn
    
    return result


def sigmoid(x: Tensor) -> Tensor:
    """
    Sigmoid 激活函数: 1 / (1 + exp(-x))
    
    前向: y = 1 / (1 + exp(-x))
    反向: dy/dx = y * (1 - y)
    """
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


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Softmax 函数: exp(x) / sum(exp(x))
    
    数值稳定版本：先减去最大值
    """
    # 数值稳定
    x_max = np.max(x.data, axis=axis, keepdims=True)
    exp_x = np.exp(x.data - x_max)
    result_data = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    result = Tensor(result_data, requires_grad=x.requires_grad)
    
    if result.requires_grad:
        result._parents = [x]
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            # Softmax 的雅可比矩阵计算
            # d(softmax_i)/d(x_j) = softmax_i * (delta_ij - softmax_j)
            # 简化计算: grad_x = softmax * (grad - sum(grad * softmax, axis))
            s = result_data
            grad_x = s * (grad - np.sum(grad * s, axis=axis, keepdims=True))
            return [grad_x]
        
        result.grad_fn = grad_fn
    
    return result


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Log Softmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
    
    数值稳定版本
    """
    x_max = np.max(x.data, axis=axis, keepdims=True)
    shifted = x.data - x_max
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    result_data = shifted - log_sum_exp
    
    result = Tensor(result_data, requires_grad=x.requires_grad)
    
    if result.requires_grad:
        result._parents = [x]
        softmax_data = np.exp(result_data)
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            # d(log_softmax)/dx = I - softmax
            grad_x = grad - softmax_data * np.sum(grad, axis=axis, keepdims=True)
            return [grad_x]
        
        result.grad_fn = grad_fn
    
    return result


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """
    Dropout 正则化
    
    Args:
        x: 输入 Tensor
        p: dropout 概率
        training: 是否在训练模式
    """
    if not training or p == 0:
        return x
    
    mask = (np.random.rand(*x.shape) > p).astype(np.float64)
    scale = 1 / (1 - p)  # 缩放以保持期望不变
    result_data = x.data * mask * scale
    
    result = Tensor(result_data, requires_grad=x.requires_grad)
    
    if result.requires_grad:
        result._parents = [x]
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            return [grad * mask * scale]
        
        result.grad_fn = grad_fn
    
    return result


def batch_norm(x: Tensor, gamma: Tensor, beta: Tensor, 
               running_mean: Optional[np.ndarray] = None,
               running_var: Optional[np.ndarray] = None,
               training: bool = True, momentum: float = 0.1,
               eps: float = 1e-5) -> Tuple[Tensor, np.ndarray, np.ndarray]:
    """
    批量归一化
    
    Args:
        x: 输入 (N, C) 或 (N, C, H, W)
        gamma: 缩放参数
        beta: 偏移参数
        running_mean: 运行均值
        running_var: 运行方差
        training: 是否训练模式
        momentum: 动量
        eps: 数值稳定性常数
    """
    if training:
        if x.ndim == 2:
            mean = x.data.mean(axis=0)
            var = x.data.var(axis=0)
        else:  # 4D: (N, C, H, W)
            mean = x.data.mean(axis=(0, 2, 3))
            var = x.data.var(axis=(0, 2, 3))
        
        # 更新运行统计
        if running_mean is not None:
            running_mean = (1 - momentum) * running_mean + momentum * mean
        else:
            running_mean = mean.copy()
        
        if running_var is not None:
            running_var = (1 - momentum) * running_var + momentum * var
        else:
            running_var = var.copy()
    else:
        mean = running_mean
        var = running_var
    
    # 归一化
    if x.ndim == 2:
        x_norm = (x.data - mean) / np.sqrt(var + eps)
        result_data = gamma.data * x_norm + beta.data
    else:
        # 扩展维度以便广播
        mean = mean.reshape(1, -1, 1, 1)
        var = var.reshape(1, -1, 1, 1)
        gamma_data = gamma.data.reshape(1, -1, 1, 1)
        beta_data = beta.data.reshape(1, -1, 1, 1)
        
        x_norm = (x.data - mean) / np.sqrt(var + eps)
        result_data = gamma_data * x_norm + beta_data
    
    requires_grad = x.requires_grad or gamma.requires_grad or beta.requires_grad
    result = Tensor(result_data, requires_grad=requires_grad)
    
    if result.requires_grad:
        result._parents = [x, gamma, beta]
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            if x.ndim == 2:
                N = x.data.shape[0]
                
                # gamma 和 beta 的梯度
                grad_gamma = np.sum(grad * x_norm, axis=0)
                grad_beta = np.sum(grad, axis=0)
                
                # x 的梯度
                std_inv = 1 / np.sqrt(var + eps)
                dx_norm = grad * gamma.data
                dvar = np.sum(dx_norm * (x.data - mean) * (-0.5) * (var + eps) ** (-1.5), axis=0)
                dmean = np.sum(dx_norm * (-std_inv), axis=0) + dvar * np.mean(-2 * (x.data - mean), axis=0)
                grad_x = dx_norm * std_inv + dvar * 2 * (x.data - mean) / N + dmean / N
            else:
                N, C, H, W = x.data.shape
                
                mean_flat = mean.reshape(-1)
                var_flat = var.reshape(-1)
                
                grad_gamma = np.sum(grad * x_norm, axis=(0, 2, 3))
                grad_beta = np.sum(grad, axis=(0, 2, 3))
                
                std_inv = 1 / np.sqrt(var + eps)
                dx_norm = grad * gamma_data
                dvar = np.sum(dx_norm * (x.data - mean) * (-0.5) * (var + eps) ** (-1.5), axis=(0, 2, 3), keepdims=True)
                dmean = np.sum(dx_norm * (-std_inv), axis=(0, 2, 3), keepdims=True) + dvar * np.mean(-2 * (x.data - mean), axis=(0, 2, 3), keepdims=True)
                grad_x = dx_norm * std_inv + dvar * 2 * (x.data - mean) / (N * H * W) + dmean / (N * H * W)
            
            return [grad_x, grad_gamma, grad_beta]
        
        result.grad_fn = grad_fn
    
    return result, running_mean, running_var


def im2col(x: np.ndarray, kernel_h: int, kernel_w: int, 
           stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    将图像转换为列矩阵，用于高效卷积计算
    
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
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    _, _, H_padded, W_padded = x.shape
    
    out_h = (H_padded - kernel_h) // stride + 1
    out_w = (W_padded - kernel_w) // stride + 1
    
    # 使用 stride_tricks 高效实现
    shape = (N, C, kernel_h, kernel_w, out_h, out_w)
    strides = (x.strides[0], x.strides[1], x.strides[2], x.strides[3],
               x.strides[2] * stride, x.strides[3] * stride)
    
    col = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
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
    
    col = col.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
    
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=col.dtype)
    
    for i in range(kernel_h):
        i_max = i + stride * out_h
        for j in range(kernel_w):
            j_max = j + stride * out_w
            x_padded[:, :, i:i_max:stride, j:j_max:stride] += col[:, :, i, j, :, :]
    
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded


def one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """
    将索引转换为 one-hot 编码
    
    Args:
        indices: 索引数组
        num_classes: 类别数量
    
    Returns:
        one-hot 编码数组
    """
    return np.eye(num_classes)[indices]


def clip(x: Tensor, min_val: float, max_val: float) -> Tensor:
    """
    裁剪 Tensor 的值到指定范围
    """
    result = Tensor(np.clip(x.data, min_val, max_val), requires_grad=x.requires_grad)
    
    if result.requires_grad:
        result._parents = [x]
        x_data = x.data.copy()
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            mask = ((x_data >= min_val) & (x_data <= max_val)).astype(np.float64)
            return [grad * mask]
        
        result.grad_fn = grad_fn
    
    return result
