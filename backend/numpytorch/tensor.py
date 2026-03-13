"""
Tensor 类和自动微分引擎

实现了类似 PyTorch 的 Tensor 类，支持：
- 自动梯度计算 (autograd)
- 计算图构建
- 反向传播 (backward)
"""

import numpy as np
from typing import Optional, List, Tuple, Callable, Union


class Tensor:
    """
    Tensor 类 - 包装 numpy.ndarray，支持自动微分
    
    属性:
        data: numpy.ndarray, 存储实际数据
        requires_grad: bool, 是否需要计算梯度
        grad: numpy.ndarray, 存储梯度
        grad_fn: Callable, 反向传播函数
        _parents: List[Tensor], 计算图中的父节点
    """
    
    def __init__(self, data: Union[np.ndarray, list, float, int], 
                 requires_grad: bool = False,
                 dtype: np.dtype = None):
        """
        初始化 Tensor
        
        Args:
            data: 输入数据，可以是 numpy 数组、列表或标量
            requires_grad: 是否需要计算梯度
            dtype: 数据类型，默认为 float64
        """
        if isinstance(data, Tensor):
            data = data.data
        
        if dtype is None:
            dtype = np.float64
            
        self.data = np.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self.grad_fn: Optional[Callable] = None
        self._parents: List['Tensor'] = []
        self._name: str = ""  # 用于调试
    
    @property
    def shape(self) -> Tuple:
        """返回 Tensor 的形状"""
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        """返回 Tensor 的维度数"""
        return self.data.ndim
    
    @property
    def size(self) -> int:
        """返回 Tensor 的元素总数"""
        return self.data.size
    
    @property
    def dtype(self) -> np.dtype:
        """返回 Tensor 的数据类型"""
        return self.data.dtype
    
    @property
    def T(self) -> 'Tensor':
        """返回转置后的 Tensor"""
        return self.transpose()
    
    def __repr__(self) -> str:
        grad_info = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        return f"Tensor({self.data}{grad_info})"
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):
        """索引操作"""
        result = Tensor(self.data[idx], requires_grad=self.requires_grad)
        
        if self.requires_grad:
            result._parents = [self]
            
            def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                # 创建全零梯度，然后在索引位置填入梯度
                full_grad = np.zeros_like(self.data)
                full_grad[idx] = grad
                return [full_grad]
            
            result.grad_fn = grad_fn
        
        return result
    
    # ==================== 反向传播 ====================
    
    def backward(self, grad: Optional[np.ndarray] = None):
        """
        执行反向传播，计算梯度
        
        使用拓扑排序确保梯度按正确顺序传播
        
        Args:
            grad: 输出的梯度，默认为全1（标量损失的情况）
        """
        if not self.requires_grad:
            return
        
        # 如果没有提供梯度，默认为 1（标量输出的情况）
        if grad is None:
            if self.data.size == 1:
                grad = np.ones_like(self.data)
            else:
                raise RuntimeError("grad must be specified for non-scalar outputs")
        
        # 拓扑排序
        topo_order = []
        visited = set()
        
        def build_topo(tensor: Tensor):
            if id(tensor) not in visited:
                visited.add(id(tensor))
                for parent in tensor._parents:
                    build_topo(parent)
                topo_order.append(tensor)
        
        build_topo(self)
        
        # 初始化当前节点的梯度
        self.grad = grad
        
        # 反向遍历，传播梯度
        for tensor in reversed(topo_order):
            if tensor.grad_fn is not None and tensor.grad is not None:
                # 计算父节点的梯度
                parent_grads = tensor.grad_fn(tensor.grad)
                
                # 累加梯度到父节点（BUG：应该是 +=，错误使用了 = 导致覆盖）
                for parent, parent_grad in zip(tensor._parents, parent_grads):
                    if parent.requires_grad:
                        if parent.grad is None:
                            parent.grad = np.zeros_like(parent.data)
                        parent.grad = parent_grad  # 错误：应该是 parent.grad + parent_grad
    
    def zero_grad(self):
        """清零梯度"""
        self.grad = None
    
    # ==================== 基础运算 ====================
    
    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """加法运算"""
        return add(self, other)
    
    def __radd__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """右加法"""
        return add(self, other)
    
    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """减法运算"""
        return sub(self, other)
    
    def __rsub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """右减法"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return sub(other, self)
    
    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """乘法运算（逐元素）"""
        return mul(self, other)
    
    def __rmul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """右乘法"""
        return mul(self, other)
    
    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """除法运算"""
        return div(self, other)
    
    def __rtruediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """右除法"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return div(other, self)
    
    def __neg__(self) -> 'Tensor':
        """取负"""
        return neg(self)
    
    def __pow__(self, power: Union[int, float]) -> 'Tensor':
        """幂运算"""
        return pow_op(self, power)
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """矩阵乘法"""
        return matmul(self, other)
    
    # ==================== 数学运算 ====================
    
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """求和"""
        return tensor_sum(self, axis=axis, keepdims=keepdims)
    
    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """求均值"""
        return tensor_mean(self, axis=axis, keepdims=keepdims)
    
    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """求最大值"""
        return tensor_max(self, axis=axis, keepdims=keepdims)
    
    def exp(self) -> 'Tensor':
        """指数函数"""
        return exp(self)
    
    def log(self) -> 'Tensor':
        """对数函数"""
        return log(self)
    
    def sqrt(self) -> 'Tensor':
        """平方根"""
        return sqrt(self)
    
    def tanh(self) -> 'Tensor':
        """双曲正切"""
        return tanh(self)
    
    def transpose(self, *axes) -> 'Tensor':
        """转置"""
        return transpose(self, axes if axes else None)
    
    def reshape(self, *shape) -> 'Tensor':
        """重塑形状"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape(self, shape)
    
    def flatten(self) -> 'Tensor':
        """展平为一维"""
        return reshape(self, (-1,))
    
    # ==================== 工具方法 ====================
    
    def detach(self) -> 'Tensor':
        """返回一个不需要梯度的新 Tensor"""
        return Tensor(self.data.copy(), requires_grad=False)
    
    def numpy(self) -> np.ndarray:
        """返回 numpy 数组"""
        return self.data.copy()
    
    def item(self) -> float:
        """返回标量值"""
        return self.data.item()
    
    def copy(self) -> 'Tensor':
        """复制 Tensor"""
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)


# ==================== 运算函数实现 ====================

def _ensure_tensor(x: Union[Tensor, np.ndarray, float, int]) -> Tensor:
    """确保输入是 Tensor"""
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


def _unbroadcast_grad(grad: np.ndarray, target_shape: Tuple) -> np.ndarray:
    """
    处理广播导致的梯度形状不匹配问题
    将梯度还原到目标形状
    """
    # 如果形状相同，直接返回
    if grad.shape == target_shape:
        return grad
    
    # 处理维度不同的情况
    ndim_diff = len(grad.shape) - len(target_shape)
    
    # 对多出的维度求和
    for _ in range(ndim_diff):
        grad = grad.sum(axis=0)
    
    # 对广播的维度求和
    for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
        if target_dim == 1 and grad_dim != 1:
            grad = grad.sum(axis=i, keepdims=True)
    
    return grad


def add(a: Tensor, b: Union[Tensor, float, int]) -> Tensor:
    """加法: c = a + b"""
    b = _ensure_tensor(b)
    result = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    if result.requires_grad:
        result._parents = [a, b]
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            grad_a = _unbroadcast_grad(grad, a.shape) if a.requires_grad else np.zeros_like(a.data)
            grad_b = _unbroadcast_grad(grad, b.shape) if b.requires_grad else np.zeros_like(b.data)
            return [grad_a, grad_b]
        
        result.grad_fn = grad_fn
    
    return result


def sub(a: Tensor, b: Union[Tensor, float, int]) -> Tensor:
    """减法: c = a - b"""
    b = _ensure_tensor(b)
    result = Tensor(a.data - b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    if result.requires_grad:
        result._parents = [a, b]
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            grad_a = _unbroadcast_grad(grad, a.shape) if a.requires_grad else np.zeros_like(a.data)
            grad_b = _unbroadcast_grad(-grad, b.shape) if b.requires_grad else np.zeros_like(b.data)
            return [grad_a, grad_b]
        
        result.grad_fn = grad_fn
    
    return result


def mul(a: Tensor, b: Union[Tensor, float, int]) -> Tensor:
    """逐元素乘法: c = a * b"""
    b = _ensure_tensor(b)
    result = Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    if result.requires_grad:
        result._parents = [a, b]
        # 保存前向传播时的值用于反向传播
        a_data = a.data.copy()
        b_data = b.data.copy()
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            grad_a = _unbroadcast_grad(grad * b_data, a.shape) if a.requires_grad else np.zeros_like(a.data)
            grad_b = _unbroadcast_grad(grad * a_data, b.shape) if b.requires_grad else np.zeros_like(b.data)
            return [grad_a, grad_b]
        
        result.grad_fn = grad_fn
    
    return result


def div(a: Tensor, b: Union[Tensor, float, int]) -> Tensor:
    """除法: c = a / b"""
    b = _ensure_tensor(b)
    result = Tensor(a.data / b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    if result.requires_grad:
        result._parents = [a, b]
        a_data = a.data.copy()
        b_data = b.data.copy()
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            grad_a = _unbroadcast_grad(grad / b_data, a.shape) if a.requires_grad else np.zeros_like(a.data)
            grad_b = _unbroadcast_grad(-grad * a_data / (b_data ** 2), b.shape) if b.requires_grad else np.zeros_like(b.data)
            return [grad_a, grad_b]
        
        result.grad_fn = grad_fn
    
    return result


def neg(a: Tensor) -> Tensor:
    """取负: c = -a"""
    result = Tensor(-a.data, requires_grad=a.requires_grad)
    
    if result.requires_grad:
        result._parents = [a]
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            return [-grad]
        
        result.grad_fn = grad_fn
    
    return result


def pow_op(a: Tensor, power: Union[int, float]) -> Tensor:
    """幂运算: c = a^power"""
    result = Tensor(a.data ** power, requires_grad=a.requires_grad)
    
    if result.requires_grad:
        result._parents = [a]
        a_data = a.data.copy()
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            return [grad * power * (a_data ** (power - 1))]
        
        result.grad_fn = grad_fn
    
    return result


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """矩阵乘法: c = a @ b"""
    result = Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    if result.requires_grad:
        result._parents = [a, b]
        a_data = a.data.copy()
        b_data = b.data.copy()
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            # grad_a = grad @ b.T
            # grad_b = a.T @ grad
            if a.ndim == 1 and b.ndim == 1:
                # 向量点积
                grad_a = grad * b_data if a.requires_grad else np.zeros_like(a.data)
                grad_b = grad * a_data if b.requires_grad else np.zeros_like(b.data)
            elif a.ndim == 1:
                # (n,) @ (n, m) -> (m,)
                grad_a = grad @ b_data.T if a.requires_grad else np.zeros_like(a.data)
                grad_b = np.outer(a_data, grad) if b.requires_grad else np.zeros_like(b.data)
            elif b.ndim == 1:
                # (n, m) @ (m,) -> (n,)
                grad_a = np.outer(grad, b_data) if a.requires_grad else np.zeros_like(a.data)
                grad_b = a_data.T @ grad if b.requires_grad else np.zeros_like(b.data)
            else:
                # 矩阵乘法
                grad_a = grad @ b_data.swapaxes(-1, -2) if a.requires_grad else np.zeros_like(a.data)
                grad_b = a_data.swapaxes(-1, -2) @ grad if b.requires_grad else np.zeros_like(b.data)
            
            return [grad_a, grad_b]
        
        result.grad_fn = grad_fn
    
    return result


def tensor_sum(a: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """求和"""
    result = Tensor(a.data.sum(axis=axis, keepdims=keepdims), requires_grad=a.requires_grad)
    
    if result.requires_grad:
        result._parents = [a]
        original_shape = a.shape
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            # 梯度需要广播回原始形状
            if axis is None:
                return [np.full(original_shape, grad)]
            else:
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                return [np.broadcast_to(grad, original_shape).copy()]
        
        result.grad_fn = grad_fn
    
    return result


def tensor_mean(a: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """求均值"""
    result = Tensor(a.data.mean(axis=axis, keepdims=keepdims), requires_grad=a.requires_grad)
    
    if result.requires_grad:
        result._parents = [a]
        original_shape = a.shape
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            if axis is None:
                n = a.data.size
            else:
                n = a.data.shape[axis]
            
            if axis is None:
                return [np.full(original_shape, grad / n)]
            else:
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                return [np.broadcast_to(grad / n, original_shape).copy()]
        
        result.grad_fn = grad_fn
    
    return result


def tensor_max(a: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """求最大值"""
    result_data = a.data.max(axis=axis, keepdims=keepdims)
    result = Tensor(result_data, requires_grad=a.requires_grad)
    
    if result.requires_grad:
        result._parents = [a]
        a_data = a.data.copy()
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            if axis is None:
                # 全局最大值
                mask = (a_data == a_data.max()).astype(np.float64)
                mask = mask / mask.sum()  # 如果有多个最大值，平均分配梯度
                return [mask * grad]
            else:
                # 沿某个轴的最大值
                max_vals = np.max(a_data, axis=axis, keepdims=True)
                mask = (a_data == max_vals).astype(np.float64)
                mask = mask / mask.sum(axis=axis, keepdims=True)
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                return [mask * grad]
        
        result.grad_fn = grad_fn
    
    return result


def exp(a: Tensor) -> Tensor:
    """指数函数: c = exp(a)"""
    result_data = np.exp(a.data)
    result = Tensor(result_data, requires_grad=a.requires_grad)
    
    if result.requires_grad:
        result._parents = [a]
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            return [grad * result_data]
        
        result.grad_fn = grad_fn
    
    return result


def log(a: Tensor) -> Tensor:
    """对数函数: c = log(a)"""
    result = Tensor(np.log(a.data), requires_grad=a.requires_grad)
    
    if result.requires_grad:
        result._parents = [a]
        a_data = a.data.copy()
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            return [grad / a_data]
        
        result.grad_fn = grad_fn
    
    return result


def sqrt(a: Tensor) -> Tensor:
    """平方根: c = sqrt(a)"""
    result_data = np.sqrt(a.data)
    result = Tensor(result_data, requires_grad=a.requires_grad)
    
    if result.requires_grad:
        result._parents = [a]
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            return [grad / (2 * result_data)]
        
        result.grad_fn = grad_fn
    
    return result


def tanh(a: Tensor) -> Tensor:
    """双曲正切: c = tanh(a)"""
    result_data = np.tanh(a.data)
    result = Tensor(result_data, requires_grad=a.requires_grad)
    
    if result.requires_grad:
        result._parents = [a]
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            return [grad * (1 - result_data ** 2)]
        
        result.grad_fn = grad_fn
    
    return result


def transpose(a: Tensor, axes: Optional[Tuple] = None) -> Tensor:
    """转置"""
    if axes is None or len(axes) == 0:
        result = Tensor(a.data.T, requires_grad=a.requires_grad)
    else:
        result = Tensor(np.transpose(a.data, axes), requires_grad=a.requires_grad)
    
    if result.requires_grad:
        result._parents = [a]
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            if axes is None or len(axes) == 0:
                return [grad.T]
            else:
                # 逆转置
                inverse_axes = np.argsort(axes)
                return [np.transpose(grad, inverse_axes)]
        
        result.grad_fn = grad_fn
    
    return result


def reshape(a: Tensor, shape: Tuple) -> Tensor:
    """重塑形状"""
    result = Tensor(a.data.reshape(shape), requires_grad=a.requires_grad)
    
    if result.requires_grad:
        result._parents = [a]
        original_shape = a.shape
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            return [grad.reshape(original_shape)]
        
        result.grad_fn = grad_fn
    
    return result


# ==================== 工具函数 ====================

def zeros(shape: Tuple, requires_grad: bool = False) -> Tensor:
    """创建全零 Tensor"""
    return Tensor(np.zeros(shape), requires_grad=requires_grad)


def ones(shape: Tuple, requires_grad: bool = False) -> Tensor:
    """创建全一 Tensor"""
    return Tensor(np.ones(shape), requires_grad=requires_grad)


def randn(*shape, requires_grad: bool = False) -> Tensor:
    """创建随机 Tensor（标准正态分布）"""
    return Tensor(np.random.randn(*shape), requires_grad=requires_grad)


def rand(*shape, requires_grad: bool = False) -> Tensor:
    """创建随机 Tensor（均匀分布 [0, 1)）"""
    return Tensor(np.random.rand(*shape), requires_grad=requires_grad)


def from_numpy(array: np.ndarray, requires_grad: bool = False) -> Tensor:
    """从 numpy 数组创建 Tensor"""
    return Tensor(array, requires_grad=requires_grad)


def stack(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """堆叠多个 Tensor"""
    data = np.stack([t.data for t in tensors], axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)
    result = Tensor(data, requires_grad=requires_grad)
    
    if result.requires_grad:
        result._parents = tensors
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            return [np.take(grad, i, axis=axis) for i in range(len(tensors))]
        
        result.grad_fn = grad_fn
    
    return result


def concatenate(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """连接多个 Tensor"""
    data = np.concatenate([t.data for t in tensors], axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)
    result = Tensor(data, requires_grad=requires_grad)
    
    if result.requires_grad:
        result._parents = tensors
        sizes = [t.shape[axis] for t in tensors]
        
        def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
            grads = np.split(grad, np.cumsum(sizes)[:-1], axis=axis)
            return grads
        
        result.grad_fn = grad_fn
    
    return result
