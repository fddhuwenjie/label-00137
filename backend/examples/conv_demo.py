"""
卷积神经网络示例

演示如何使用 Conv2d、MaxPool2d 等层构建 CNN。

任务: 简单图像分类
使用合成的简单图像模式进行分类。

这个示例演示了:
1. 如何使用 Conv2d 进行特征提取
2. 如何使用 MaxPool2d 进行下采样
3. CNN 的典型结构: Conv -> ReLU -> Pool -> Flatten -> FC
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpytorch import Tensor
from numpytorch.layers import Conv2d, MaxPool2d, Linear, ReLU
from numpytorch.losses import CrossEntropyLoss, MSELoss
from numpytorch.optim import SGD, Adam


class SimpleCNN:
    """
    简单的 CNN 模型
    
    结构:
        Conv2d(1, 8, 3) -> ReLU -> MaxPool2d(2) ->
        Conv2d(8, 16, 3) -> ReLU -> MaxPool2d(2) ->
        Flatten -> Linear -> ReLU -> Linear
    """
    
    def __init__(self, input_channels: int, num_classes: int, input_size: int = 28):
        # 计算经过卷积和池化后的特征图大小
        # 输入: (input_size, input_size)
        # Conv1: (input_size-2, input_size-2)
        # Pool1: ((input_size-2)//2, (input_size-2)//2)
        # Conv2: ((input_size-2)//2-2, (input_size-2)//2-2)
        # Pool2: (((input_size-2)//2-2)//2, ((input_size-2)//2-2)//2)
        
        self.conv1 = Conv2d(input_channels, 8, kernel_size=3, padding=0)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=2)
        
        self.conv2 = Conv2d(8, 16, kernel_size=3, padding=0)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(kernel_size=2)
        
        # 计算 flatten 后的大小
        size_after_conv1 = input_size - 2  # 28 -> 26
        size_after_pool1 = size_after_conv1 // 2  # 26 -> 13
        size_after_conv2 = size_after_pool1 - 2  # 13 -> 11
        size_after_pool2 = size_after_conv2 // 2  # 11 -> 5
        
        flatten_size = 16 * size_after_pool2 * size_after_pool2
        
        self.fc1 = Linear(flatten_size, 32)
        self.relu3 = ReLU()
        self.fc2 = Linear(32, num_classes)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 (batch_size, channels, height, width)
        
        Returns:
            分类 logits (batch_size, num_classes)
        """
        # 第一个卷积块
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x
    
    def parameters(self):
        """返回所有参数"""
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params
    
    def zero_grad(self):
        """清零梯度"""
        self.conv1.zero_grad()
        self.conv2.zero_grad()
        self.fc1.zero_grad()
        self.fc2.zero_grad()


def generate_pattern_images(n_samples: int, image_size: int = 28, n_classes: int = 4):
    """
    生成简单的图像模式数据
    
    四种模式:
    0: 水平条纹
    1: 垂直条纹
    2: 左上到右下对角线
    3: 右上到左下对角线
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    samples_per_class = n_samples // n_classes
    
    for label in range(n_classes):
        for _ in range(samples_per_class):
            img = np.zeros((image_size, image_size))
            
            if label == 0:  # 水平条纹
                for i in range(0, image_size, 4):
                    img[i:i+2, :] = 1
            elif label == 1:  # 垂直条纹
                for j in range(0, image_size, 4):
                    img[:, j:j+2] = 1
            elif label == 2:  # 左上到右下对角线
                for i in range(image_size):
                    for j in range(max(0, i-2), min(image_size, i+3)):
                        img[i, j] = 1
            else:  # 右上到左下对角线
                for i in range(image_size):
                    j_center = image_size - 1 - i
                    for j in range(max(0, j_center-2), min(image_size, j_center+3)):
                        img[i, j] = 1
            
            # 添加噪声
            img = img + np.random.randn(image_size, image_size) * 0.2
            img = np.clip(img, 0, 1)
            
            X.append(img)
            y.append(label)
    
    X = np.array(X)[:, np.newaxis, :, :]  # 添加通道维度: (N, 1, H, W)
    y = np.array(y)
    
    # 打乱
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def demo_conv_operations():
    """演示卷积操作"""
    print("=" * 60)
    print("卷积操作演示")
    print("=" * 60)
    
    # 创建一个简单的输入
    print("\n1. 单个卷积层测试:")
    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)  # (batch=2, channels=3, H=8, W=8)
    conv = Conv2d(3, 6, kernel_size=3, padding=1)
    
    print(f"   输入形状: {x.shape}")
    print(f"   卷积核形状: {conv.weight.shape}")
    
    y = conv(x)
    print(f"   输出形状: {y.shape}")
    
    # 测试反向传播
    loss = y.sum()
    loss.backward()
    print(f"   输入梯度形状: {x.grad.shape if x.grad is not None else 'None'}")
    print(f"   卷积核梯度形状: {conv.weight.grad.shape if conv.weight.grad is not None else 'None'}")
    
    # 池化测试
    print("\n2. 池化层测试:")
    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    pool = MaxPool2d(kernel_size=2)
    
    print(f"   输入形状: {x.shape}")
    y = pool(x)
    print(f"   MaxPool 输出形状: {y.shape}")
    
    # 组合测试
    print("\n3. Conv + ReLU + Pool 组合:")
    x = Tensor(np.random.randn(1, 1, 16, 16), requires_grad=True)
    conv = Conv2d(1, 4, kernel_size=3, padding=1)
    relu = ReLU()
    pool = MaxPool2d(kernel_size=2)
    
    print(f"   输入: {x.shape}")
    x = conv(x)
    print(f"   Conv2d 后: {x.shape}")
    x = relu(x)
    print(f"   ReLU 后: {x.shape}")
    x = pool(x)
    print(f"   MaxPool 后: {x.shape}")


def train_cnn():
    """训练 CNN 分类模型"""
    print("\n" + "=" * 60)
    print("CNN 图像分类示例")
    print("=" * 60)
    
    # 生成数据
    image_size = 28
    n_classes = 4
    X_train, y_train = generate_pattern_images(200, image_size, n_classes)
    X_test, y_test = generate_pattern_images(40, image_size, n_classes)
    
    print(f"\n数据形状: X={X_train.shape}, y={y_train.shape}")
    print(f"类别: 0=水平条纹, 1=垂直条纹, 2=正对角线, 3=反对角线")
    
    # 创建模型
    print("\n创建 CNN 模型...")
    model = SimpleCNN(input_channels=1, num_classes=n_classes, input_size=image_size)
    
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    
    print(f"模型参数数量: {sum(p.data.size for p in model.parameters())}")
    
    # 训练
    print("\n开始训练...")
    n_epochs = 30
    batch_size = 16
    
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
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # 测试
            X_test_tensor = Tensor(X_test)
            logits_test = model.forward(X_test_tensor)
            test_acc = np.mean(np.argmax(logits_test.data, axis=1) == y_test)
            
            print(f"Epoch {epoch + 1:3d}/{n_epochs} | "
                  f"Train Loss: {total_loss/n_samples:.4f}, Train Acc: {correct/n_samples:.4f} | "
                  f"Test Acc: {test_acc:.4f}")
    
    # 最终评估
    print("\n" + "=" * 60)
    print("最终评估")
    print("=" * 60)
    X_test_tensor = Tensor(X_test)
    logits_test = model.forward(X_test_tensor)
    predictions = np.argmax(logits_test.data, axis=1)
    
    # 混淆矩阵
    print("\n混淆矩阵:")
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_test, predictions):
        confusion[true, pred] += 1
    
    labels = ['水平', '垂直', '正斜', '反斜']
    print("真实\\预测", end="\t")
    for label in labels:
        print(label, end="\t")
    print()
    
    for i, label in enumerate(labels):
        print(label, end="\t\t")
        for j in range(n_classes):
            print(confusion[i, j], end="\t")
        print()
    
    accuracy = np.mean(predictions == y_test)
    print(f"\n总体准确率: {accuracy:.4f}")


def main():
    demo_conv_operations()
    train_cnn()
    print("\nCNN 示例完成!")


if __name__ == "__main__":
    main()
