"""
简单 RNN 示例

演示如何使用 RNNCell 和 RNN 处理序列数据。

任务: 序列求和预测
给定一个数字序列，预测序列的和（回归任务）

这个示例演示了:
1. 如何使用 RNNCell 逐步处理序列
2. 如何使用 RNN 处理整个序列
3. 序列到单值的回归任务
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpytorch import Tensor
from numpytorch.layers import RNNCell, RNN, Linear
from numpytorch.losses import MSELoss
from numpytorch.optim import SGD, Adam


class SequenceSumModel:
    """
    序列求和模型
    
    使用 RNN 处理序列，然后用 Linear 层预测和。
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        self.rnn_cell = RNNCell(input_size, hidden_size, nonlinearity='tanh')
        self.fc = Linear(hidden_size, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 (batch_size, seq_len, input_size)
        
        Returns:
            预测值 (batch_size, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # 初始化隐藏状态
        h = Tensor(np.zeros((batch_size, self.rnn_cell.hidden_size)), requires_grad=True)
        
        # 逐步处理序列
        for t in range(seq_len):
            x_t = Tensor(x.data[:, t, :], requires_grad=x.requires_grad)
            h = self.rnn_cell(x_t, h)
        
        # 使用最后的隐藏状态预测
        output = self.fc(h)
        
        return output
    
    def parameters(self):
        """返回所有参数"""
        return self.rnn_cell.parameters() + self.fc.parameters()
    
    def zero_grad(self):
        """清零梯度"""
        self.rnn_cell.zero_grad()
        self.fc.zero_grad()


class SequenceClassifier:
    """
    序列分类模型
    
    使用 RNN 处理序列，分类序列的特征。
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        self.rnn = RNN(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = Linear(hidden_size, num_classes)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 (batch_size, seq_len, input_size)
        
        Returns:
            分类 logits (batch_size, num_classes)
        """
        # RNN 处理序列
        output, h_n = self.rnn(x)
        
        # 使用最后一个时间步的输出
        last_output = Tensor(output.data[:, -1, :], requires_grad=x.requires_grad)
        
        # 分类
        logits = self.fc(last_output)
        
        return logits
    
    def parameters(self):
        return self.rnn.parameters() + self.fc.parameters()
    
    def zero_grad(self):
        self.rnn.zero_grad()
        self.fc.zero_grad()


def generate_sequence_sum_data(n_samples: int, seq_len: int, input_size: int = 1):
    """
    生成序列求和数据
    
    每个序列是随机数，目标是序列所有元素的和。
    """
    np.random.seed(42)
    
    X = np.random.randn(n_samples, seq_len, input_size)
    y = X.sum(axis=(1, 2), keepdims=True)  # 序列求和
    
    return X, y


def generate_sequence_pattern_data(n_samples: int, seq_len: int):
    """
    生成序列模式分类数据
    
    三种模式:
    0: 上升趋势
    1: 下降趋势
    2: 平稳
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    samples_per_class = n_samples // 3
    
    for label in range(3):
        for _ in range(samples_per_class):
            if label == 0:  # 上升
                seq = np.linspace(0, 1, seq_len) + np.random.randn(seq_len) * 0.1
            elif label == 1:  # 下降
                seq = np.linspace(1, 0, seq_len) + np.random.randn(seq_len) * 0.1
            else:  # 平稳
                seq = 0.5 + np.random.randn(seq_len) * 0.1
            
            X.append(seq.reshape(-1, 1))
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # 打乱
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def train_sum_model():
    """训练序列求和模型"""
    print("=" * 60)
    print("序列求和 RNN 示例")
    print("=" * 60)
    
    # 生成数据
    seq_len = 10
    X_train, y_train = generate_sequence_sum_data(500, seq_len)
    X_test, y_test = generate_sequence_sum_data(100, seq_len)
    
    print(f"\n数据形状: X={X_train.shape}, y={y_train.shape}")
    
    # 创建模型
    model = SequenceSumModel(input_size=1, hidden_size=32)
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # 训练
    print("\n开始训练...")
    n_epochs = 50
    batch_size = 32
    
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i + batch_size]
            X_batch = Tensor(X_train[batch_idx], requires_grad=True)
            y_batch = Tensor(y_train[batch_idx])
            
            optimizer.zero_grad()
            
            pred = model.forward(X_batch)
            loss = criterion(pred, y_batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # 测试
            X_test_tensor = Tensor(X_test)
            pred_test = model.forward(X_test_tensor)
            test_loss = criterion(pred_test, Tensor(y_test)).data
            
            print(f"Epoch {epoch + 1:3d}/{n_epochs} | "
                  f"Train Loss: {avg_loss:.4f} | Test Loss: {test_loss:.4f}")
    
    # 验证
    print("\n验证预测结果:")
    for i in range(5):
        x = Tensor(X_test[i:i+1])
        pred = model.forward(x)
        actual = float(y_test[i].flatten()[0])
        predicted = float(pred.data[0, 0])
        print(f"  序列和: 实际={actual:.4f}, 预测={predicted:.4f}, 误差={abs(actual-predicted):.4f}")


def train_pattern_model():
    """训练序列模式分类模型"""
    print("\n" + "=" * 60)
    print("序列模式分类 RNN 示例")
    print("=" * 60)
    
    # 生成数据
    seq_len = 20
    X_train, y_train = generate_sequence_pattern_data(300, seq_len)
    X_test, y_test = generate_sequence_pattern_data(60, seq_len)
    
    print(f"\n数据形状: X={X_train.shape}, y={y_train.shape}")
    print(f"类别: 0=上升, 1=下降, 2=平稳")
    
    # 创建模型
    model = SequenceClassifier(input_size=1, hidden_size=16, num_classes=3)
    
    from numpytorch.losses import CrossEntropyLoss
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.5, momentum=0.9)
    
    # 训练
    print("\n开始训练...")
    n_epochs = 30
    batch_size = 32
    
    for epoch in range(n_epochs):
        total_loss = 0
        correct = 0
        n_samples = 0
        
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i + batch_size]
            X_batch = Tensor(X_train[batch_idx], requires_grad=True)
            y_batch = Tensor(y_train[batch_idx])
            
            optimizer.zero_grad()
            
            logits = model.forward(X_batch)
            loss = criterion(logits, y_batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data * len(batch_idx)
            correct += np.sum(np.argmax(logits.data, axis=1) == y_train[batch_idx])
            n_samples += len(batch_idx)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # 测试
            X_test_tensor = Tensor(X_test)
            logits_test = model.forward(X_test_tensor)
            test_acc = np.mean(np.argmax(logits_test.data, axis=1) == y_test)
            
            print(f"Epoch {epoch + 1:3d}/{n_epochs} | "
                  f"Train Loss: {total_loss/n_samples:.4f}, Train Acc: {correct/n_samples:.4f} | "
                  f"Test Acc: {test_acc:.4f}")
    
    print("\n训练完成!")


def main():
    train_sum_model()
    train_pattern_model()
    print("\n所有 RNN 示例完成!")


if __name__ == "__main__":
    main()
