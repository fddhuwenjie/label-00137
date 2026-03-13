"""
循环神经网络层实现

包含:
- RNNCell: 单个 RNN 单元
- RNN: 处理整个序列的 RNN 层

RNN 公式:
h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_ih + b_hh)
"""

import numpy as np
from typing import List, Tuple, Optional
from ..tensor import Tensor


class RNNCell:
    """
    RNN 单元
    
    处理单个时间步的输入，更新隐藏状态。
    
    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
        bias: 是否使用偏置
        nonlinearity: 激活函数，'tanh' 或 'relu'
    
    公式:
        h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_ih + b_hh)
    """
    
    def __init__(self, input_size: int, hidden_size: int, 
                 bias: bool = True, nonlinearity: str = 'tanh'):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias
        self.nonlinearity = nonlinearity
        
        # 初始化权重 (使用 Xavier 初始化)
        std = np.sqrt(1.0 / hidden_size)
        
        # 输入到隐藏的权重 (hidden_size, input_size)
        self.weight_ih = Tensor(
            np.random.randn(hidden_size, input_size) * std,
            requires_grad=True
        )
        
        # 隐藏到隐藏的权重 (hidden_size, hidden_size)
        self.weight_hh = Tensor(
            np.random.randn(hidden_size, hidden_size) * std,
            requires_grad=True
        )
        
        if bias:
            self.bias_ih = Tensor(np.zeros(hidden_size), requires_grad=True)
            self.bias_hh = Tensor(np.zeros(hidden_size), requires_grad=True)
        else:
            self.bias_ih = None
            self.bias_hh = None
    
    def __call__(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        return self.forward(x, h)
    
    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入 Tensor (batch_size, input_size)
            h: 上一时间步的隐藏状态 (batch_size, hidden_size)
               如果为 None，则初始化为零
        
        Returns:
            新的隐藏状态 (batch_size, hidden_size)
        """
        batch_size = x.shape[0] if x.ndim > 1 else 1
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # 初始化隐藏状态
        if h is None:
            h = Tensor(np.zeros((batch_size, self.hidden_size)), requires_grad=x.requires_grad)
        elif h.ndim == 1:
            h = h.reshape(1, -1)
        
        # 计算 W_ih @ x + W_hh @ h + b
        # x: (batch, input_size)
        # W_ih: (hidden_size, input_size)
        # W_ih.T: (input_size, hidden_size)
        # x @ W_ih.T: (batch, hidden_size)
        
        ih = x @ self.weight_ih.T
        hh = h @ self.weight_hh.T
        
        combined = ih + hh
        
        if self.use_bias:
            combined = combined + self.bias_ih + self.bias_hh
        
        # 应用激活函数
        if self.nonlinearity == 'tanh':
            h_new = combined.tanh()
        elif self.nonlinearity == 'relu':
            # ReLU
            result_data = np.maximum(0, combined.data)
            h_new = Tensor(result_data, requires_grad=combined.requires_grad)
            
            if h_new.requires_grad:
                h_new._parents = [combined]
                combined_data = combined.data.copy()
                
                def grad_fn(grad: np.ndarray) -> List[np.ndarray]:
                    return [grad * (combined_data > 0).astype(np.float64)]
                
                h_new.grad_fn = grad_fn
        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")
        
        return h_new
    
    def parameters(self) -> List[Tensor]:
        """返回所有可训练参数"""
        params = [self.weight_ih, self.weight_hh]
        if self.use_bias:
            params.extend([self.bias_ih, self.bias_hh])
        return params
    
    def zero_grad(self):
        """清零梯度"""
        for p in self.parameters():
            p.zero_grad()
    
    def __repr__(self) -> str:
        return f"RNNCell({self.input_size}, {self.hidden_size}, nonlinearity='{self.nonlinearity}')"


class RNN:
    """
    多层 RNN
    
    处理整个序列，支持多层堆叠。
    
    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
        num_layers: RNN 层数
        bias: 是否使用偏置
        batch_first: 如果 True，输入形状为 (batch, seq, feature)
                     否则为 (seq, batch, feature)
        nonlinearity: 激活函数
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = True, 
                 nonlinearity: str = 'tanh'):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.nonlinearity = nonlinearity
        
        # 创建多层 RNNCell
        self.cells = []
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            self.cells.append(RNNCell(cell_input_size, hidden_size, bias, nonlinearity))
    
    def __call__(self, x: Tensor, h_0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        return self.forward(x, h_0)
    
    def forward(self, x: Tensor, h_0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列
               如果 batch_first: (batch, seq_len, input_size)
               否则: (seq_len, batch, input_size)
            h_0: 初始隐藏状态 (num_layers, batch, hidden_size)
        
        Returns:
            output: 所有时间步的输出 (batch, seq_len, hidden_size) 或 (seq_len, batch, hidden_size)
            h_n: 最后一个时间步的隐藏状态 (num_layers, batch, hidden_size)
        """
        # 转换为 (seq_len, batch, input_size)
        if self.batch_first:
            # (batch, seq, feature) -> (seq, batch, feature)
            x_data = x.data.transpose(1, 0, 2)
        else:
            x_data = x.data
        
        seq_len, batch_size, _ = x_data.shape
        
        # 初始化隐藏状态
        if h_0 is None:
            h = [Tensor(np.zeros((batch_size, self.hidden_size)), requires_grad=x.requires_grad) 
                 for _ in range(self.num_layers)]
        else:
            h = [Tensor(h_0.data[i], requires_grad=h_0.requires_grad) 
                 for i in range(self.num_layers)]
        
        # 存储所有时间步的输出
        outputs = []
        
        # 遍历序列
        for t in range(seq_len):
            # 获取当前时间步的输入
            x_t = Tensor(x_data[t], requires_grad=x.requires_grad)
            
            # 通过每一层
            layer_input = x_t
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx] = cell(layer_input, h[layer_idx])
                layer_input = h[layer_idx]
            
            # 最后一层的输出
            outputs.append(h[-1].data)
        
        # 组合输出
        output_data = np.stack(outputs, axis=0)  # (seq_len, batch, hidden)
        
        if self.batch_first:
            output_data = output_data.transpose(1, 0, 2)  # (batch, seq_len, hidden)
        
        # 组合最终隐藏状态
        h_n_data = np.stack([h_i.data for h_i in h], axis=0)  # (num_layers, batch, hidden)
        
        output = Tensor(output_data, requires_grad=x.requires_grad)
        h_n = Tensor(h_n_data, requires_grad=x.requires_grad)
        
        # 设置梯度函数（简化实现）
        if output.requires_grad:
            output._parents = [x]
            # 注意：这是简化实现，完整的 BPTT 需要更复杂的梯度计算
            # 在实际使用中，推荐使用逐步计算的方式
        
        return output, h_n
    
    def parameters(self) -> List[Tensor]:
        """返回所有可训练参数"""
        params = []
        for cell in self.cells:
            params.extend(cell.parameters())
        return params
    
    def zero_grad(self):
        """清零梯度"""
        for cell in self.cells:
            cell.zero_grad()
    
    def __repr__(self) -> str:
        return (f"RNN({self.input_size}, {self.hidden_size}, "
                f"num_layers={self.num_layers}, batch_first={self.batch_first})")


class LSTMCell:
    """
    LSTM 单元 (可选扩展)
    
    包含遗忘门、输入门、输出门和细胞状态。
    
    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
        bias: 是否使用偏置
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias
        
        # 初始化权重
        std = np.sqrt(1.0 / hidden_size)
        
        # 输入到隐藏的权重 (4 * hidden_size, input_size)
        # 四个门的权重合并在一起
        self.weight_ih = Tensor(
            np.random.randn(4 * hidden_size, input_size) * std,
            requires_grad=True
        )
        
        # 隐藏到隐藏的权重 (4 * hidden_size, hidden_size)
        self.weight_hh = Tensor(
            np.random.randn(4 * hidden_size, hidden_size) * std,
            requires_grad=True
        )
        
        if bias:
            self.bias_ih = Tensor(np.zeros(4 * hidden_size), requires_grad=True)
            self.bias_hh = Tensor(np.zeros(4 * hidden_size), requires_grad=True)
        else:
            self.bias_ih = None
            self.bias_hh = None
    
    def __call__(self, x: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        return self.forward(x, state)
    
    def forward(self, x: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        """
        前向传播
        
        Args:
            x: 输入 (batch_size, input_size)
            state: (h, c) 上一时间步的隐藏状态和细胞状态
        
        Returns:
            (h_new, c_new): 新的隐藏状态和细胞状态
        """
        batch_size = x.shape[0] if x.ndim > 1 else 1
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        if state is None:
            h = Tensor(np.zeros((batch_size, self.hidden_size)), requires_grad=x.requires_grad)
            c = Tensor(np.zeros((batch_size, self.hidden_size)), requires_grad=x.requires_grad)
        else:
            h, c = state
            if h.ndim == 1:
                h = h.reshape(1, -1)
            if c.ndim == 1:
                c = c.reshape(1, -1)
        
        # 计算门
        gates = x @ self.weight_ih.T + h @ self.weight_hh.T
        
        if self.use_bias:
            gates = gates + self.bias_ih + self.bias_hh
        
        # 分割四个门
        gates_data = gates.data
        i_gate = self._sigmoid(gates_data[:, :self.hidden_size])
        f_gate = self._sigmoid(gates_data[:, self.hidden_size:2*self.hidden_size])
        g_gate = np.tanh(gates_data[:, 2*self.hidden_size:3*self.hidden_size])
        o_gate = self._sigmoid(gates_data[:, 3*self.hidden_size:])
        
        # 更新细胞状态
        c_new_data = f_gate * c.data + i_gate * g_gate
        
        # 更新隐藏状态
        h_new_data = o_gate * np.tanh(c_new_data)
        
        h_new = Tensor(h_new_data, requires_grad=x.requires_grad)
        c_new = Tensor(c_new_data, requires_grad=x.requires_grad)
        
        # 简化的梯度实现（完整实现需要保存所有中间变量）
        if h_new.requires_grad:
            h_new._parents = [x, gates]
            c_new._parents = [x, gates]
        
        return h_new, c_new
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """数值稳定的 sigmoid"""
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    
    def parameters(self) -> List[Tensor]:
        params = [self.weight_ih, self.weight_hh]
        if self.use_bias:
            params.extend([self.bias_ih, self.bias_hh])
        return params
    
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
    
    def __repr__(self) -> str:
        return f"LSTMCell({self.input_size}, {self.hidden_size})"
