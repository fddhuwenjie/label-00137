# NumpyTorch - 基于 NumPy 的深度学习框架

## How to Run

```bash
# 使用 Docker Compose 运行
docker-compose up --build -d

# 查看运行日志
docker-compose logs -f

# 停止服务
docker-compose down
```

**本地运行（无 Docker）：**

```bash
cd backend
pip install -r requirements.txt

# 运行测试
python -m pytest tests/ -v

# 运行示例
python examples/mnist_mlp.py
python examples/simple_rnn.py
python examples/conv_demo.py
```

## Services

| 服务 | 端口 | 说明 |
|------|------|------|
| backend | 8084 | NumpyTorch 深度学习框架（运行测试和示例） |

**验证测试步骤：**

```bash
# 1. 启动并运行测试和演示（前台模式，直接查看输出）
docker-compose up --build

# 或者后台运行后查看日志
docker-compose up --build -d
docker-compose logs -f

# 2. 预期输出包含：
#    - "12 passed" (单元测试全部通过)
#    - "NumpyTorch 交互式演示" (演示各模块功能)
#    - "所有演示运行完成"
```

**演示内容：**
- Tensor 基本操作 (加减乘除)
- 自动微分 (backward 计算梯度)
- 神经网络层 (Linear, ReLU, Sigmoid)
- 卷积神经网络 (Conv2d, MaxPool2d)
- 循环神经网络 (RNNCell)
- 训练示例 (学习 y=2x+1)
- 分类任务 (CrossEntropyLoss)

> **注意**：本项目是纯 Python 学习框架，不提供 HTTP Web 服务。运行 `docker-compose up` 会执行单元测试和演示脚本，通过查看终端输出来验证功能。

## 测试账号

本项目为学习型框架，无需账号认证。

## 题目内容

我现在想深入学习一下torch的原理，自己基于numpy实现各个计算函数，各个层包括线性层、relu、sigmoid、卷积、池化、rnn单元、rnn、损失函数层实现两个mse和交叉熵withsoftmax，模拟实现自动梯度引擎，可以简化实现到各个层的内部，帮我规划一下实现方案

**实现内容：**
1. **自动微分引擎** - Tensor 类，支持计算图构建和反向传播
2. **基础运算** - 加减乘除、矩阵乘法、求和、均值等
3. **神经网络层**：
   - Linear（线性层/全连接层）
   - ReLU、Sigmoid、Tanh、Softmax（激活函数）
   - Conv2d（2D卷积层，使用 im2col 技术）
   - MaxPool2d、AvgPool2d（池化层）
   - RNNCell、RNN（循环神经网络）
4. **损失函数**：
   - MSELoss（均方误差）
   - CrossEntropyLoss（交叉熵，带 Softmax）
5. **优化器**：
   - SGD（随机梯度下降，支持动量）
   - Adam

---

## 项目介绍

NumpyTorch 是一个教学目的的深度学习框架，完全基于 NumPy 实现，帮助理解 PyTorch 等框架的底层原理。

### 项目结构

```
137/
├── backend/                    # 主项目目录
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── numpytorch/            # 核心框架
│   │   ├── __init__.py
│   │   ├── tensor.py          # Tensor 类和自动微分
│   │   ├── ops.py             # 基础运算操作
│   │   ├── layers/            # 神经网络层
│   │   │   ├── linear.py      # 线性层
│   │   │   ├── activation.py  # 激活函数
│   │   │   ├── conv.py        # 卷积层
│   │   │   ├── pooling.py     # 池化层
│   │   │   └── rnn.py         # RNN 层
│   │   ├── losses/            # 损失函数
│   │   │   ├── mse.py
│   │   │   └── cross_entropy.py
│   │   └── optim/             # 优化器
│   │       └── sgd.py
│   ├── examples/              # 示例代码
│   │   ├── mnist_mlp.py       # MLP 分类示例
│   │   ├── simple_rnn.py      # RNN 序列示例
│   │   └── conv_demo.py       # CNN 图像示例
│   └── tests/                 # 单元测试
│       └── test_all.py
├── docker-compose.yml
├── .gitignore
└── README.md
```

### 核心特性

#### 1. 自动微分引擎

```python
from numpytorch import Tensor

# 创建需要梯度的 Tensor
x = Tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 2 * x

# 反向传播
loss = y.sum()
loss.backward()

print(x.grad)  # [6.0, 8.0] (dy/dx = 2x + 2)
```

#### 2. 构建神经网络

```python
from numpytorch import Tensor
from numpytorch.layers import Linear, ReLU
from numpytorch.losses import CrossEntropyLoss
from numpytorch.optim import SGD

# 定义网络
linear1 = Linear(784, 128)
relu = ReLU()
linear2 = Linear(128, 10)

# 前向传播
x = Tensor(data, requires_grad=True)
h = relu(linear1(x))
logits = linear2(h)

# 计算损失
criterion = CrossEntropyLoss()
loss = criterion(logits, targets)

# 反向传播
loss.backward()

# 更新参数
optimizer = SGD([linear1.weight, linear1.bias, linear2.weight, linear2.bias], lr=0.01)
optimizer.step()
```

#### 3. 卷积神经网络

```python
from numpytorch.layers import Conv2d, MaxPool2d, Linear, ReLU

# 构建 CNN
conv1 = Conv2d(1, 16, kernel_size=3, padding=1)
pool = MaxPool2d(kernel_size=2)
conv2 = Conv2d(16, 32, kernel_size=3, padding=1)
fc = Linear(32 * 7 * 7, 10)

# 前向传播
x = conv1(input)      # (N, 1, 28, 28) -> (N, 16, 28, 28)
x = ReLU()(x)
x = pool(x)           # (N, 16, 28, 28) -> (N, 16, 14, 14)
x = conv2(x)          # (N, 16, 14, 14) -> (N, 32, 14, 14)
x = ReLU()(x)
x = pool(x)           # (N, 32, 14, 14) -> (N, 32, 7, 7)
x = x.reshape(N, -1)  # Flatten
logits = fc(x)
```

#### 4. 循环神经网络

```python
from numpytorch.layers import RNN, Linear

# 构建 RNN
rnn = RNN(input_size=10, hidden_size=64, num_layers=2, batch_first=True)
fc = Linear(64, num_classes)

# 处理序列
output, h_n = rnn(sequence)  # sequence: (batch, seq_len, features)
logits = fc(output[:, -1, :])  # 使用最后时间步
```

### 实现原理

#### 自动微分

采用动态计算图，每个 Tensor 记录：
- `data`: NumPy 数组存储实际数据
- `grad`: 存储梯度
- `grad_fn`: 反向传播函数
- `_parents`: 计算图中的父节点

反向传播使用拓扑排序确保梯度按正确顺序计算。

#### im2col 卷积

将卷积运算转换为矩阵乘法：
1. 将输入的每个卷积窗口展开为一列
2. 将卷积核展开为一行
3. 矩阵乘法得到输出

这种方法虽然增加内存使用，但可以利用高度优化的矩阵乘法库。

### 运行示例

**MLP 手写数字分类：**
```bash
python examples/mnist_mlp.py
```

**RNN 序列预测：**
```bash
python examples/simple_rnn.py
```

**CNN 图像分类：**
```bash
python examples/conv_demo.py
```

### 单元测试

```bash
python -m pytest tests/ -v
```

测试覆盖：
- Tensor 基本操作
- 自动微分正确性
- 各层的前向/反向传播
- 损失函数
- 优化器
- 数值梯度检验

## 技术栈

- Python 3.11
- NumPy >= 1.24.0
- pytest >= 7.0.0

## License

MIT License
