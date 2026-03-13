#!/usr/bin/env python3
"""
NumpyTorch 交互式演示

在控制台中演示框架的核心功能
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from numpytorch import Tensor, randn
from numpytorch.layers import Linear, ReLU, Sigmoid, Conv2d, MaxPool2d, RNNCell
from numpytorch.losses import MSELoss, CrossEntropyLoss
from numpytorch.optim import SGD, Adam


def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_tensor_basics():
    """演示 Tensor 基本操作"""
    print_header("1. Tensor 基本操作")
    
    print("\n>>> a = Tensor([1, 2, 3], requires_grad=True)")
    a = Tensor([1, 2, 3], requires_grad=True)
    print(f"a = {a}")
    
    print("\n>>> b = Tensor([4, 5, 6], requires_grad=True)")
    b = Tensor([4, 5, 6], requires_grad=True)
    print(f"b = {b}")
    
    print("\n>>> c = a + b")
    c = a + b
    print(f"c = {c}")
    
    print("\n>>> d = a * b")
    d = a * b
    print(f"d = {d}")
    
    print("\n>>> e = d.sum()")
    e = d.sum()
    print(f"e = {e}")


def demo_autograd():
    """演示自动微分"""
    print_header("2. 自动微分 (Autograd)")
    
    print("\n# 计算 y = x^2 + 2x 在 x=3 处的导数")
    print(">>> x = Tensor([3.0], requires_grad=True)")
    x = Tensor([3.0], requires_grad=True)
    
    print(">>> y = x ** 2 + 2 * x")
    y = x ** 2 + 2 * x
    print(f"y = {y}")
    
    print("\n>>> y.backward()")
    y.backward()
    
    print(f"\n导数 dy/dx = 2x + 2 = 2*3 + 2 = 8")
    print(f"x.grad = {x.grad}")
    
    print("\n# 多变量梯度")
    print(">>> a = Tensor([2.0], requires_grad=True)")
    print(">>> b = Tensor([3.0], requires_grad=True)")
    print(">>> z = a * b + a")
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    z = a * b + a
    
    print(">>> z.backward()")
    z.backward()
    
    print(f"\nz = ab + a, dz/da = b + 1 = 4, dz/db = a = 2")
    print(f"a.grad = {a.grad}, b.grad = {b.grad}")


def demo_neural_network():
    """演示神经网络层"""
    print_header("3. 神经网络层")
    
    print("\n# Linear 层")
    print(">>> linear = Linear(4, 2)")
    linear = Linear(4, 2)
    
    print(">>> x = Tensor([[1, 2, 3, 4]], requires_grad=True)")
    x = Tensor([[1, 2, 3, 4]], requires_grad=True)
    
    print(">>> y = linear(x)")
    y = linear(x)
    print(f"输入形状: {x.shape} -> 输出形状: {y.shape}")
    print(f"y = {y.data}")
    
    print("\n# 激活函数")
    print(">>> relu = ReLU()")
    print(">>> x = Tensor([-2, -1, 0, 1, 2])")
    relu = ReLU()
    x = Tensor([-2, -1, 0, 1, 2])
    y = relu(x)
    print(f"ReLU([-2, -1, 0, 1, 2]) = {y.data}")
    
    print("\n>>> sigmoid = Sigmoid()")
    sigmoid = Sigmoid()
    y = sigmoid(x)
    print(f"Sigmoid([-2, -1, 0, 1, 2]) = {np.round(y.data, 4)}")


def demo_conv():
    """演示卷积层"""
    print_header("4. 卷积神经网络")
    
    print("\n# 创建 Conv2d 层")
    print(">>> conv = Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)")
    conv = Conv2d(1, 4, kernel_size=3, padding=1)
    
    print("\n# 输入: batch=1, channels=1, height=8, width=8")
    print(">>> x = randn(1, 1, 8, 8, requires_grad=True)")
    x = randn(1, 1, 8, 8, requires_grad=True)
    
    print(">>> y = conv(x)")
    y = conv(x)
    print(f"输入形状: {x.shape} -> 输出形状: {y.shape}")
    
    print("\n# 池化层")
    print(">>> pool = MaxPool2d(kernel_size=2)")
    pool = MaxPool2d(kernel_size=2)
    z = pool(y)
    print(f"MaxPool 后: {y.shape} -> {z.shape}")


def demo_rnn():
    """演示 RNN"""
    print_header("5. 循环神经网络 (RNN)")
    
    print("\n# 创建 RNNCell")
    print(">>> rnn_cell = RNNCell(input_size=4, hidden_size=8)")
    rnn_cell = RNNCell(input_size=4, hidden_size=8)
    
    print("\n# 处理单个时间步")
    print(">>> x_t = Tensor([[1, 2, 3, 4]], requires_grad=True)  # batch=1, features=4")
    x_t = Tensor([[1, 2, 3, 4]], requires_grad=True)
    
    print(">>> h = rnn_cell(x_t)")
    h = rnn_cell(x_t)
    print(f"隐藏状态形状: {h.shape}")
    print(f"h = {np.round(h.data[0, :4], 4)}... (显示前4个)")
    
    print("\n# 处理序列 (3个时间步)")
    h = None
    for t in range(3):
        x_t = Tensor(np.random.randn(1, 4), requires_grad=True)
        h = rnn_cell(x_t, h)
        print(f"  时间步 {t+1}: h 范围 [{h.data.min():.3f}, {h.data.max():.3f}]")


def demo_training():
    """演示训练过程"""
    print_header("6. 简单训练示例")
    
    print("\n# 目标: 学习 y = 2x + 1")
    print(">>> 生成数据")
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 2 * X + 1 + np.random.randn(100, 1) * 0.1
    
    print(">>> 创建模型: Linear(1, 1)")
    model = Linear(1, 1)
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    
    print(">>> 训练 100 轮...")
    for epoch in range(100):
        # 前向传播
        X_tensor = Tensor(X, requires_grad=True)
        y_tensor = Tensor(y)
        
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss = {loss.data:.6f}")
    
    print(f"\n学习到的参数:")
    print(f"  权重 W = {model.weight.data[0, 0]:.4f} (期望: 2.0)")
    print(f"  偏置 b = {model.bias.data[0]:.4f} (期望: 1.0)")


def demo_classification():
    """演示分类任务"""
    print_header("7. 分类任务示例")
    
    print("\n# 3分类任务")
    print(">>> logits = Tensor([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0]], requires_grad=True)")
    logits = Tensor([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0]], requires_grad=True)
    
    print(">>> targets = Tensor([0, 1])  # 正确类别")
    targets = Tensor([0, 1])
    
    print(">>> criterion = CrossEntropyLoss()")
    criterion = CrossEntropyLoss()
    
    print(">>> loss = criterion(logits, targets)")
    loss = criterion(logits, targets)
    print(f"Loss = {loss.data:.4f}")
    
    print("\n>>> loss.backward()")
    loss.backward()
    print(f"logits.grad =\n{np.round(logits.grad, 4)}")
    
    print("\n预测结果:")
    predictions = np.argmax(logits.data, axis=1)
    print(f"  预测: {predictions}, 实际: [0, 1]")


def main():
    print("\n" + "=" * 60)
    print("       NumpyTorch 交互式演示")
    print("       基于 NumPy 实现的深度学习框架")
    print("=" * 60)
    
    demos = [
        ("Tensor 基本操作", demo_tensor_basics),
        ("自动微分", demo_autograd),
        ("神经网络层", demo_neural_network),
        ("卷积神经网络", demo_conv),
        ("循环神经网络", demo_rnn),
        ("训练示例", demo_training),
        ("分类任务", demo_classification),
    ]
    
    for name, demo_func in demos:
        demo_func()
        print()
    
    print("=" * 60)
    print("  演示完成！")
    print("=" * 60)
    print("\n更多示例请运行:")
    print("  python examples/mnist_mlp.py   # MLP 分类")
    print("  python examples/simple_rnn.py  # RNN 序列")
    print("  python examples/conv_demo.py   # CNN 图像")
    print()


if __name__ == "__main__":
    main()
