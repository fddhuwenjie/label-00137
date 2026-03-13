"""
均方误差损失 (MSE Loss)

公式: MSE = mean((pred - target)^2)

用于回归任务。
"""

import numpy as np
from typing import List
from ..tensor import Tensor


class MSELoss:
    """
    均方误差损失
    
    前向: loss = mean((pred - target)^2)
    反向: grad = 2 * (pred - target) / n
    
    Args:
        reduction: 损失的聚合方式
            - 'mean': 返回平均损失 (默认)
            - 'sum': 返回总损失
            - 'none': 返回每个样本的损失
    """
    
    def __init__(self, reduction: str = 'mean'):
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
    
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.forward(pred, target)
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        计算 MSE 损失
        
        Args:
            pred: 预测值 Tensor
            target: 目标值 Tensor (形状与 pred 相同)
        
        Returns:
            损失 Tensor
        """
        # 确保 target 是 Tensor
        if not isinstance(target, Tensor):
            target = Tensor(target)
        
        # 计算差值
        diff = pred.data - target.data
        
        # 计算平方误差
        squared_error = diff ** 2
        
        # 根据 reduction 方式聚合
        if self.reduction == 'mean':
            loss_data = np.mean(squared_error)
        elif self.reduction == 'sum':
            loss_data = np.sum(squared_error)
        else:  # 'none'
            loss_data = squared_error
        
        result = Tensor(loss_data, requires_grad=pred.requires_grad)
        
        if result.requires_grad:
            result._parents = [pred]
            pred_data = pred.data.copy()
            target_data = target.data.copy()
            n = pred.data.size
            
            def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                # d(MSE)/d(pred) = 2 * (pred - target) / n (for mean reduction)
                diff = pred_data - target_data
                
                if self.reduction == 'mean':
                    grad_pred = grad * 2 * diff / n
                elif self.reduction == 'sum':
                    grad_pred = grad * 2 * diff
                else:  # 'none'
                    grad_pred = grad * 2 * diff
                
                return [grad_pred]
            
            result.grad_fn = grad_fn
        
        return result
    
    def __repr__(self) -> str:
        return f"MSELoss(reduction='{self.reduction}')"


class L1Loss:
    """
    L1 损失 (平均绝对误差)
    
    前向: loss = mean(|pred - target|)
    反向: grad = sign(pred - target) / n
    
    Args:
        reduction: 损失的聚合方式
    """
    
    def __init__(self, reduction: str = 'mean'):
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
    
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.forward(pred, target)
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """计算 L1 损失"""
        if not isinstance(target, Tensor):
            target = Tensor(target)
        
        diff = pred.data - target.data
        abs_error = np.abs(diff)
        
        if self.reduction == 'mean':
            loss_data = np.mean(abs_error)
        elif self.reduction == 'sum':
            loss_data = np.sum(abs_error)
        else:
            loss_data = abs_error
        
        result = Tensor(loss_data, requires_grad=pred.requires_grad)
        
        if result.requires_grad:
            result._parents = [pred]
            pred_data = pred.data.copy()
            target_data = target.data.copy()
            n = pred.data.size
            
            def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                diff = pred_data - target_data
                sign = np.sign(diff)
                
                if self.reduction == 'mean':
                    grad_pred = grad * sign / n
                elif self.reduction == 'sum':
                    grad_pred = grad * sign
                else:
                    grad_pred = grad * sign
                
                return [grad_pred]
            
            result.grad_fn = grad_fn
        
        return result
    
    def __repr__(self) -> str:
        return f"L1Loss(reduction='{self.reduction}')"


class SmoothL1Loss:
    """
    平滑 L1 损失 (Huber Loss)
    
    当 |x| < beta 时使用 L2 损失，否则使用 L1 损失。
    
    前向:
        if |x| < beta:
            loss = 0.5 * x^2 / beta
        else:
            loss = |x| - 0.5 * beta
    
    Args:
        beta: 平滑参数
        reduction: 损失的聚合方式
    """
    
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        self.beta = beta
        self.reduction = reduction
    
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.forward(pred, target)
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """计算 Smooth L1 损失"""
        if not isinstance(target, Tensor):
            target = Tensor(target)
        
        diff = pred.data - target.data
        abs_diff = np.abs(diff)
        
        # 分段计算
        loss_data = np.where(
            abs_diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            abs_diff - 0.5 * self.beta
        )
        
        if self.reduction == 'mean':
            loss_data = np.mean(loss_data)
        elif self.reduction == 'sum':
            loss_data = np.sum(loss_data)
        
        result = Tensor(loss_data, requires_grad=pred.requires_grad)
        
        if result.requires_grad:
            result._parents = [pred]
            pred_data = pred.data.copy()
            target_data = target.data.copy()
            beta = self.beta
            n = pred.data.size
            
            def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                diff = pred_data - target_data
                abs_diff = np.abs(diff)
                
                grad_pred = np.where(
                    abs_diff < beta,
                    diff / beta,
                    np.sign(diff)
                )
                
                if self.reduction == 'mean':
                    grad_pred = grad * grad_pred / n
                elif self.reduction == 'sum':
                    grad_pred = grad * grad_pred
                else:
                    grad_pred = grad * grad_pred
                
                return [grad_pred]
            
            result.grad_fn = grad_fn
        
        return result
    
    def __repr__(self) -> str:
        return f"SmoothL1Loss(beta={self.beta}, reduction='{self.reduction}')"
