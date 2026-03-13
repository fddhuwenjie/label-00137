"""
MNIST MLP 示例

使用简单的多层感知机 (MLP) 在 MNIST 数据集上训练手写数字分类模型。

网络结构:
    输入 (784) -> Linear (128) -> ReLU -> Linear (64) -> ReLU -> Linear (10) -> Softmax

这个示例演示了:
1. 如何使用 NumpyTorch 构建神经网络
2. 前向传播和反向传播
3. 使用 SGD 优化器训练模型
4. 使用 CrossEntropyLoss 损失函数
"""

import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpytorch import Tensor
from numpytorch.layers import Linear, ReLU
from numpytorch.losses import CrossEntropyLoss
from numpytorch.optim import SGD, Adam


class MLP:
    """
    简单的多层感知机
    """
    
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int):
        """
        Args:
            input_size: 输入特征维度
            hidden_sizes: 隐藏层大小列表
            output_size: 输出类别数
        """
        self.layers = []
        self.activations = []
        
        # 构建网络
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(Linear(prev_size, hidden_size))
            self.activations.append(ReLU())
            prev_size = hidden_size
        
        # 输出层
        self.layers.append(Linear(prev_size, output_size))
    
    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activations[i](x)
        
        # 最后一层不加激活（CrossEntropyLoss 包含 softmax）
        x = self.layers[-1](x)
        return x
    
    def parameters(self):
        """返回所有参数"""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def zero_grad(self):
        """清零梯度"""
        for layer in self.layers:
            layer.zero_grad()


def generate_synthetic_mnist(n_samples: int = 1000, n_classes: int = 10):
    """
    生成合成的 MNIST 风格数据
    
    由于实际 MNIST 数据需要下载，这里生成简单的合成数据用于演示。
    每个类别对应一个基础模式。
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    samples_per_class = n_samples // n_classes
    
    for class_idx in range(n_classes):
        # 每个类别生成一个基础模式
        base_pattern = np.zeros(784)
        
        # 在不同位置放置特征
        start_row = (class_idx % 5) * 5
        start_col = (class_idx // 5) * 14
        
        for i in range(5):
            for j in range(5):
                idx = (start_row + i) * 28 + (start_col + j)
                if idx < 784:
                    base_pattern[idx] = 1.0
        
        # 添加噪声生成样本
        for _ in range(samples_per_class):
            sample = base_pattern + np.random.randn(784) * 0.3
            sample = np.clip(sample, 0, 1)
            X.append(sample)
            y.append(class_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    # 打乱数据
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def train_epoch(model, X, y, criterion, optimizer, batch_size: int = 32):
    """训练一个 epoch"""
    n_samples = len(X)
    total_loss = 0
    correct = 0
    
    # 打乱数据
    indices = np.random.permutation(n_samples)
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        X_batch = Tensor(X[batch_indices], requires_grad=True)
        y_batch = Tensor(y[batch_indices])
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        logits = model.forward(X_batch)
        
        # 计算损失
        loss = criterion(logits, y_batch)
        total_loss += loss.data * len(batch_indices)
        
        # 计算准确率
        predictions = np.argmax(logits.data, axis=1)
        correct += np.sum(predictions == y[batch_indices])
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
    
    avg_loss = total_loss / n_samples
    accuracy = correct / n_samples
    
    return avg_loss, accuracy


def evaluate(model, X, y, criterion, batch_size: int = 32):
    """评估模型"""
    n_samples = len(X)
    total_loss = 0
    correct = 0
    
    for i in range(0, n_samples, batch_size):
        X_batch = Tensor(X[i:i + batch_size])
        y_batch = Tensor(y[i:i + batch_size])
        
        logits = model.forward(X_batch)
        loss = criterion(logits, y_batch)
        
        total_loss += loss.data * len(X_batch)
        
        predictions = np.argmax(logits.data, axis=1)
        correct += np.sum(predictions == y[i:i + batch_size])
    
    avg_loss = total_loss / n_samples
    accuracy = correct / n_samples
    
    return avg_loss, accuracy


def main():
    print("=" * 60)
    print("NumpyTorch MLP MNIST 示例")
    print("=" * 60)
    
    # 生成数据
    print("\n生成合成数据...")
    X_train, y_train = generate_synthetic_mnist(n_samples=800)
    X_test, y_test = generate_synthetic_mnist(n_samples=200)
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 创建模型
    print("\n创建 MLP 模型...")
    model = MLP(
        input_size=784,
        hidden_sizes=[128, 64],
        output_size=10
    )
    
    # 损失函数和优化器
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    print(f"模型参数数量: {sum(p.data.size for p in model.parameters())}")
    
    # 训练
    print("\n开始训练...")
    n_epochs = 20
    
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, X_train, y_train, criterion, optimizer)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            test_loss, test_acc = evaluate(model, X_test, y_test, criterion)
            print(f"Epoch {epoch + 1:3d}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # 最终评估
    print("\n" + "=" * 60)
    print("最终评估")
    print("=" * 60)
    test_loss, test_acc = evaluate(model, X_test, y_test, criterion)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    
    print("\n训练完成!")


if __name__ == "__main__":
    main()
