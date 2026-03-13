"""
交叉熵损失 (Cross Entropy Loss)

结合 Softmax 和交叉熵损失，用于多分类任务。

关键公式:
- Softmax: p_i = exp(x_i) / sum(exp(x_j))
- CrossEntropy: L = -sum(y_i * log(p_i))
- 合并后的梯度 (数值稳定): grad = softmax - one_hot(target)
"""

import numpy as np
from typing import List, Optional
from ..tensor import Tensor


class CrossEntropyLoss:
    """
    交叉熵损失 (带 Softmax)
    
    该实现将 Softmax 和交叉熵合并计算，具有更好的数值稳定性。
    
    公式:
        softmax: p = exp(x - max(x)) / sum(exp(x - max(x)))
        loss = -sum(target * log(p))
        
        对于类别索引形式的 target:
        loss = -log(p[target])
    
    梯度:
        grad = softmax - target (one-hot)
    
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
    
    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        return self.forward(logits, target)
    
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        计算交叉熵损失
        
        Args:
            logits: 未经 softmax 的原始分数 (batch_size, num_classes)
            target: 目标类别索引 (batch_size,) 或 one-hot 编码 (batch_size, num_classes)
        
        Returns:
            损失 Tensor
        """
        if not isinstance(target, Tensor):
            target = Tensor(target)
        
        # 数值稳定的 softmax
        logits_shifted = logits.data - np.max(logits.data, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        softmax = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # 判断 target 是类别索引还是 one-hot
        if target.data.ndim == 1 or (target.data.ndim == 2 and target.data.shape[1] == 1):
            # 类别索引形式
            target_indices = target.data.astype(np.int64).flatten()
            batch_size = logits.data.shape[0]
            
            # 获取正确类别的概率
            correct_probs = softmax[np.arange(batch_size), target_indices]
            
            # 计算损失: -log(p_correct)
            # 添加小值避免 log(0)
            eps = 1e-12
            loss_per_sample = -np.log(correct_probs + eps)
            
            # one-hot 编码用于梯度计算
            one_hot = np.zeros_like(softmax)
            one_hot[np.arange(batch_size), target_indices] = 1
        else:
            # one-hot 形式
            one_hot = target.data
            eps = 1e-12
            loss_per_sample = -np.sum(one_hot * np.log(softmax + eps), axis=-1)
        
        # 根据 reduction 聚合
        if self.reduction == 'mean':
            loss_data = np.mean(loss_per_sample)
        elif self.reduction == 'sum':
            loss_data = np.sum(loss_per_sample)
        else:
            loss_data = loss_per_sample
        
        result = Tensor(loss_data, requires_grad=logits.requires_grad)
        
        if result.requires_grad:
            result._parents = [logits]
            batch_size = logits.data.shape[0]
            
            def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                # 交叉熵 + softmax 的梯度是 softmax - one_hot
                grad_logits = softmax - one_hot
                
                if self.reduction == 'mean':
                    grad_logits = grad * grad_logits / batch_size
                elif self.reduction == 'sum':
                    grad_logits = grad * grad_logits
                else:
                    grad_logits = grad.reshape(-1, 1) * grad_logits
                
                return [grad_logits]
            
            result.grad_fn = grad_fn
        
        return result
    
    def __repr__(self) -> str:
        return f"CrossEntropyLoss(reduction='{self.reduction}')"


class BinaryCrossEntropyLoss:
    """
    二元交叉熵损失
    
    用于二分类或多标签分类任务。
    输入应该经过 sigmoid 激活。
    
    公式:
        loss = -[y * log(p) + (1-y) * log(1-p)]
    
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
        """
        计算二元交叉熵损失
        
        Args:
            pred: 预测概率 (已经经过 sigmoid)
            target: 目标标签 (0 或 1)
        
        Returns:
            损失 Tensor
        """
        if not isinstance(target, Tensor):
            target = Tensor(target)
        
        eps = 1e-12
        pred_clipped = np.clip(pred.data, eps, 1 - eps)
        
        # BCE = -[y * log(p) + (1-y) * log(1-p)]
        loss_data = -(target.data * np.log(pred_clipped) + 
                     (1 - target.data) * np.log(1 - pred_clipped))
        
        if self.reduction == 'mean':
            loss_data = np.mean(loss_data)
        elif self.reduction == 'sum':
            loss_data = np.sum(loss_data)
        
        result = Tensor(loss_data, requires_grad=pred.requires_grad)
        
        if result.requires_grad:
            result._parents = [pred]
            pred_data = pred.data.copy()
            target_data = target.data.copy()
            n = pred.data.size
            
            def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                pred_clipped = np.clip(pred_data, eps, 1 - eps)
                # d(BCE)/d(pred) = (pred - target) / (pred * (1 - pred))
                grad_pred = (pred_clipped - target_data) / (pred_clipped * (1 - pred_clipped) + eps)
                
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
        return f"BinaryCrossEntropyLoss(reduction='{self.reduction}')"


class BCEWithLogitsLoss:
    """
    带 Logits 的二元交叉熵损失
    
    结合 Sigmoid 和 BCE，数值更稳定。
    
    公式:
        loss = max(x, 0) - x*y + log(1 + exp(-|x|))
    
    Args:
        reduction: 损失的聚合方式
    """
    
    def __init__(self, reduction: str = 'mean'):
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
    
    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        return self.forward(logits, target)
    
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        计算带 logits 的二元交叉熵损失
        
        Args:
            logits: 原始分数 (未经 sigmoid)
            target: 目标标签 (0 或 1)
        
        Returns:
            损失 Tensor
        """
        if not isinstance(target, Tensor):
            target = Tensor(target)
        
        # 数值稳定的实现
        # loss = max(x, 0) - x*y + log(1 + exp(-|x|))
        x = logits.data
        y = target.data
        
        loss_data = np.maximum(x, 0) - x * y + np.log(1 + np.exp(-np.abs(x)))
        
        if self.reduction == 'mean':
            loss_data = np.mean(loss_data)
        elif self.reduction == 'sum':
            loss_data = np.sum(loss_data)
        
        result = Tensor(loss_data, requires_grad=logits.requires_grad)
        
        if result.requires_grad:
            result._parents = [logits]
            logits_data = logits.data.copy()
            target_data = target.data.copy()
            n = logits.data.size
            
            def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                # sigmoid(x) - y
                sigmoid = 1 / (1 + np.exp(-logits_data))
                grad_logits = sigmoid - target_data
                
                if self.reduction == 'mean':
                    grad_logits = grad * grad_logits / n
                elif self.reduction == 'sum':
                    grad_logits = grad * grad_logits
                else:
                    grad_logits = grad * grad_logits
                
                return [grad_logits]
            
            result.grad_fn = grad_fn
        
        return result
    
    def __repr__(self) -> str:
        return f"BCEWithLogitsLoss(reduction='{self.reduction}')"


class NLLLoss:
    """
    负对数似然损失
    
    用于已经经过 log_softmax 的输入。
    
    公式:
        loss = -log_softmax[target]
    
    Args:
        reduction: 损失的聚合方式
    """
    
    def __init__(self, reduction: str = 'mean'):
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
    
    def __call__(self, log_probs: Tensor, target: Tensor) -> Tensor:
        return self.forward(log_probs, target)
    
    def forward(self, log_probs: Tensor, target: Tensor) -> Tensor:
        """
        计算 NLL 损失
        
        Args:
            log_probs: 对数概率 (经过 log_softmax)
            target: 目标类别索引
        
        Returns:
            损失 Tensor
        """
        if not isinstance(target, Tensor):
            target = Tensor(target)
        
        batch_size = log_probs.data.shape[0]
        target_indices = target.data.astype(np.int64).flatten()
        
        # 获取目标类别的对数概率
        loss_per_sample = -log_probs.data[np.arange(batch_size), target_indices]
        
        if self.reduction == 'mean':
            loss_data = np.mean(loss_per_sample)
        elif self.reduction == 'sum':
            loss_data = np.sum(loss_per_sample)
        else:
            loss_data = loss_per_sample
        
        result = Tensor(loss_data, requires_grad=log_probs.requires_grad)
        
        if result.requires_grad:
            result._parents = [log_probs]
            num_classes = log_probs.data.shape[1]
            
            def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                grad_log_probs = np.zeros_like(log_probs.data)
                grad_log_probs[np.arange(batch_size), target_indices] = -1
                
                if self.reduction == 'mean':
                    grad_log_probs = grad * grad_log_probs / batch_size
                elif self.reduction == 'sum':
                    grad_log_probs = grad * grad_log_probs
                else:
                    grad_log_probs = grad.reshape(-1, 1) * grad_log_probs
                
                return [grad_log_probs]
            
            result.grad_fn = grad_fn
        
        return result
    
    def __repr__(self) -> str:
        return f"NLLLoss(reduction='{self.reduction}')"
