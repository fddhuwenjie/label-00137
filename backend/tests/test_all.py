"""
NumpyTorch 单元测试

测试所有核心组件:
- Tensor 和自动微分
- 各种层
- 损失函数
- 优化器
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpytorch import Tensor, randn, zeros, ones
from numpytorch.layers import Linear, ReLU, Sigmoid, Conv2d, MaxPool2d, AvgPool2d, RNNCell, RNN
from numpytorch.losses import MSELoss, CrossEntropyLoss
from numpytorch.optim import SGD, Adam


def test_tensor_basic():
    """测试 Tensor 基本操作"""
    print("测试 Tensor 基本操作...", end=" ")
    
    # 创建 Tensor
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)
    
    # 加法
    c = a + b
    assert np.allclose(c.data, [5, 7, 9])
    
    # 乘法
    d = a * b
    assert np.allclose(d.data, [4, 10, 18])
    
    # 矩阵乘法
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = Tensor([[5, 6], [7, 8]], requires_grad=True)
    z = x @ y
    assert np.allclose(z.data, [[19, 22], [43, 50]])
    
    print("通过!")


def test_autograd():
    """测试自动微分"""
    print("测试自动微分...", end=" ")
    
    # 简单的梯度测试
    x = Tensor([2.0], requires_grad=True)
    y = x ** 2  # y = x^2, dy/dx = 2x = 4
    y.backward()
    assert np.allclose(x.grad, [4.0])
    
    # 链式法则
    x = Tensor([3.0], requires_grad=True)
    y = x * 2  # y = 2x
    z = y ** 2  # z = 4x^2, dz/dx = 8x = 24
    z.backward()
    assert np.allclose(x.grad, [24.0])
    
    # 多变量
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    c = a * b  # c = ab
    d = c + a  # d = ab + a
    d.backward()
    # dd/da = b + 1 = 4
    # dd/db = a = 2
    assert np.allclose(a.grad, [4.0])
    assert np.allclose(b.grad, [2.0])
    
    print("通过!")


def test_linear_layer():
    """测试线性层"""
    print("测试线性层...", end=" ")
    
    linear = Linear(10, 5)
    x = Tensor(np.random.randn(3, 10), requires_grad=True)
    
    y = linear(x)
    
    assert y.shape == (3, 5)
    
    # 测试反向传播
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert linear.weight.grad is not None
    assert linear.bias.grad is not None
    
    print("通过!")


def test_activations():
    """测试激活函数"""
    print("测试激活函数...", end=" ")
    
    x = Tensor(np.array([-1, 0, 1, 2]), requires_grad=True)
    
    # ReLU
    relu = ReLU()
    y_relu = relu(x)
    assert np.allclose(y_relu.data, [0, 0, 1, 2])
    
    # Sigmoid
    sigmoid = Sigmoid()
    y_sigmoid = sigmoid(x)
    expected = 1 / (1 + np.exp(-x.data))
    assert np.allclose(y_sigmoid.data, expected)
    
    # 测试梯度
    y_relu.sum().backward()
    assert x.grad is not None
    
    print("通过!")


def test_conv2d():
    """测试卷积层"""
    print("测试卷积层...", end=" ")
    
    conv = Conv2d(3, 8, kernel_size=3, padding=1)
    x = Tensor(np.random.randn(2, 3, 16, 16), requires_grad=True)
    
    y = conv(x)
    
    # 检查输出形状 (with padding=1, kernel=3, 输出大小不变)
    assert y.shape == (2, 8, 16, 16)
    
    # 测试反向传播
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
    assert conv.weight.grad is not None
    
    print("通过!")


def test_pooling():
    """测试池化层"""
    print("测试池化层...", end=" ")
    
    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    
    # MaxPool
    maxpool = MaxPool2d(kernel_size=2)
    y_max = maxpool(x)
    assert y_max.shape == (2, 3, 4, 4)
    
    # AvgPool
    x2 = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    avgpool = AvgPool2d(kernel_size=2)
    y_avg = avgpool(x2)
    assert y_avg.shape == (2, 3, 4, 4)
    
    # 测试反向传播
    y_max.sum().backward()
    assert x.grad is not None
    
    print("通过!")


def test_rnn():
    """测试 RNN"""
    print("测试 RNN...", end=" ")
    
    # RNNCell
    cell = RNNCell(input_size=10, hidden_size=20)
    x = Tensor(np.random.randn(5, 10), requires_grad=True)
    h = cell(x)
    assert h.shape == (5, 20)
    
    # RNN
    rnn = RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
    x_seq = Tensor(np.random.randn(5, 8, 10), requires_grad=True)  # (batch, seq, features)
    output, h_n = rnn(x_seq)
    
    assert output.shape == (5, 8, 20)  # (batch, seq, hidden)
    assert h_n.shape == (2, 5, 20)  # (num_layers, batch, hidden)
    
    print("通过!")


def test_mse_loss():
    """测试 MSE 损失"""
    print("测试 MSE 损失...", end=" ")
    
    criterion = MSELoss()
    
    pred = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    target = Tensor(np.array([1.5, 2.5, 3.5]))
    
    loss = criterion(pred, target)
    
    # MSE = mean((0.5)^2 + (0.5)^2 + (0.5)^2) = 0.25
    assert np.allclose(loss.data, 0.25)
    
    # 测试反向传播
    loss.backward()
    assert pred.grad is not None
    # grad = 2 * (pred - target) / n = 2 * (-0.5, -0.5, -0.5) / 3 = (-1/3, -1/3, -1/3)
    expected_grad = 2 * (pred.data - target.data) / 3
    assert np.allclose(pred.grad, expected_grad)
    
    print("通过!")


def test_cross_entropy():
    """测试交叉熵损失"""
    print("测试交叉熵损失...", end=" ")
    
    criterion = CrossEntropyLoss()
    
    # logits: (batch_size=3, num_classes=4)
    logits = Tensor(np.array([
        [2.0, 1.0, 0.1, 0.05],
        [0.1, 0.5, 2.0, 0.3],
        [0.2, 0.3, 0.4, 2.5]
    ]), requires_grad=True)
    
    target = Tensor(np.array([0, 2, 3]))  # 正确类别
    
    loss = criterion(logits, target)
    
    # 损失应该是正数
    assert loss.data > 0
    
    # 测试反向传播
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    
    print("通过!")


def test_sgd():
    """测试 SGD 优化器"""
    print("测试 SGD 优化器...", end=" ")
    
    # 简单的优化问题: 最小化 (x - 3)^2
    x = Tensor([0.0], requires_grad=True)
    
    optimizer = SGD([x], lr=0.1)
    
    for _ in range(100):
        optimizer.zero_grad()
        loss = (x - 3) ** 2
        loss.backward()
        optimizer.step()
    
    # x 应该接近 3
    assert np.allclose(x.data, [3.0], atol=0.01)
    
    print("通过!")


def test_adam():
    """测试 Adam 优化器"""
    print("测试 Adam 优化器...", end=" ")
    
    # 简单的优化问题
    x = Tensor([0.0], requires_grad=True)
    
    optimizer = Adam([x], lr=0.1)
    
    for _ in range(100):
        optimizer.zero_grad()
        loss = (x - 5) ** 2
        loss.backward()
        optimizer.step()
    
    # x 应该接近 5
    assert np.allclose(x.data, [5.0], atol=0.1)
    
    print("通过!")


def test_gradient_numerical():
    """数值梯度检验"""
    print("数值梯度检验...", end=" ")
    
    eps = 1e-5
    
    # 测试线性层
    linear = Linear(4, 3)
    x = Tensor(np.random.randn(2, 4), requires_grad=True)
    
    # 解析梯度
    y = linear(x)
    loss = y.sum()
    loss.backward()
    
    analytic_grad = x.grad.copy()
    
    # 数值梯度
    numerical_grad = np.zeros_like(x.data)
    for i in range(x.data.shape[0]):
        for j in range(x.data.shape[1]):
            x_plus = x.data.copy()
            x_plus[i, j] += eps
            x_minus = x.data.copy()
            x_minus[i, j] -= eps
            
            y_plus = linear(Tensor(x_plus)).sum().data
            y_minus = linear(Tensor(x_minus)).sum().data
            
            numerical_grad[i, j] = (y_plus - y_minus) / (2 * eps)
    
    # 比较
    assert np.allclose(analytic_grad, numerical_grad, rtol=1e-3, atol=1e-3)
    
    print("通过!")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("NumpyTorch 单元测试")
    print("=" * 60)
    print()
    
    tests = [
        test_tensor_basic,
        test_autograd,
        test_linear_layer,
        test_activations,
        test_conv2d,
        test_pooling,
        test_rnn,
        test_mse_loss,
        test_cross_entropy,
        test_sgd,
        test_adam,
        test_gradient_numerical,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"失败! 错误: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
