"""
NumpyTorch 演示服务器

提供 HTTP API 演示深度学习框架的各项功能。
"""

import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from numpytorch import Tensor, randn
from numpytorch.layers import Linear, ReLU, Sigmoid, Conv2d, MaxPool2d, RNNCell
from numpytorch.losses import MSELoss, CrossEntropyLoss
from numpytorch.optim import SGD, Adam


class DemoHandler(BaseHTTPRequestHandler):
    """演示请求处理器"""
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_html_page()
        elif self.path == '/api/autograd':
            self.demo_autograd()
        elif self.path == '/api/linear':
            self.demo_linear()
        elif self.path == '/api/conv':
            self.demo_conv()
        elif self.path == '/api/rnn':
            self.demo_rnn()
        elif self.path == '/api/train':
            self.demo_train()
        elif self.path == '/api/health':
            self.send_json({'status': 'ok', 'message': 'NumpyTorch Demo Server is running!'})
        else:
            self.send_error(404, 'Not Found')
    
    def send_json(self, data):
        """发送 JSON 响应"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'))
    
    def send_html_page(self):
        """发送演示页面"""
        html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NumpyTorch 演示</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: white; text-align: center; margin-bottom: 30px; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .subtitle { color: rgba(255,255,255,0.9); text-align: center; margin-bottom: 40px; font-size: 1.2em; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }
        .card { background: white; border-radius: 16px; padding: 24px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); transition: transform 0.3s; }
        .card:hover { transform: translateY(-5px); }
        .card h2 { color: #333; margin-bottom: 15px; font-size: 1.4em; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        .card p { color: #666; margin-bottom: 15px; line-height: 1.6; }
        .btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-size: 1em; width: 100%; transition: opacity 0.3s; }
        .btn:hover { opacity: 0.9; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .result { background: #f8f9fa; border-radius: 8px; padding: 15px; margin-top: 15px; font-family: 'Monaco', 'Consolas', monospace; font-size: 0.85em; white-space: pre-wrap; max-height: 300px; overflow-y: auto; display: none; border: 1px solid #e9ecef; }
        .result.show { display: block; }
        .loading { text-align: center; color: #667eea; }
        .status { background: #d4edda; color: #155724; padding: 15px; border-radius: 8px; margin-bottom: 30px; text-align: center; }
        .features { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px; }
        .feature-tag { background: #e9ecef; padding: 5px 12px; border-radius: 20px; font-size: 0.85em; color: #495057; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 NumpyTorch 演示</h1>
        <p class="subtitle">基于 NumPy 实现的深度学习框架 - 学习 PyTorch 原理</p>
        
        <div class="status" id="status">正在检查服务状态...</div>
        
        <div class="grid">
            <div class="card">
                <h2>📐 自动微分引擎</h2>
                <p>演示 Tensor 类的自动梯度计算功能。计算 y = x² + 2x 的梯度。</p>
                <div class="features">
                    <span class="feature-tag">计算图</span>
                    <span class="feature-tag">反向传播</span>
                    <span class="feature-tag">链式法则</span>
                </div>
                <button class="btn" onclick="runDemo('autograd', this)">运行演示</button>
                <div class="result" id="result-autograd"></div>
            </div>
            
            <div class="card">
                <h2>🔗 线性层 (Linear)</h2>
                <p>演示全连接层的前向传播和反向传播。</p>
                <div class="features">
                    <span class="feature-tag">权重初始化</span>
                    <span class="feature-tag">矩阵乘法</span>
                    <span class="feature-tag">梯度计算</span>
                </div>
                <button class="btn" onclick="runDemo('linear', this)">运行演示</button>
                <div class="result" id="result-linear"></div>
            </div>
            
            <div class="card">
                <h2>🖼️ 卷积层 (Conv2d)</h2>
                <p>演示 2D 卷积操作，使用 im2col 技术实现高效计算。</p>
                <div class="features">
                    <span class="feature-tag">im2col</span>
                    <span class="feature-tag">特征提取</span>
                    <span class="feature-tag">池化</span>
                </div>
                <button class="btn" onclick="runDemo('conv', this)">运行演示</button>
                <div class="result" id="result-conv"></div>
            </div>
            
            <div class="card">
                <h2>🔄 循环神经网络 (RNN)</h2>
                <p>演示 RNN 单元处理序列数据的能力。</p>
                <div class="features">
                    <span class="feature-tag">RNNCell</span>
                    <span class="feature-tag">隐藏状态</span>
                    <span class="feature-tag">序列处理</span>
                </div>
                <button class="btn" onclick="runDemo('rnn', this)">运行演示</button>
                <div class="result" id="result-rnn"></div>
            </div>
            
            <div class="card">
                <h2>🎯 模型训练演示</h2>
                <p>演示完整的训练流程：前向传播 → 损失计算 → 反向传播 → 参数更新。</p>
                <div class="features">
                    <span class="feature-tag">CrossEntropy</span>
                    <span class="feature-tag">SGD 优化器</span>
                    <span class="feature-tag">训练循环</span>
                </div>
                <button class="btn" onclick="runDemo('train', this)">运行演示</button>
                <div class="result" id="result-train"></div>
            </div>
        </div>
    </div>
    
    <script>
        // 检查服务状态
        fetch('/api/health')
            .then(r => r.json())
            .then(data => {
                document.getElementById('status').innerHTML = '✅ ' + data.message;
            })
            .catch(err => {
                document.getElementById('status').innerHTML = '❌ 服务连接失败';
                document.getElementById('status').style.background = '#f8d7da';
                document.getElementById('status').style.color = '#721c24';
            });
        
        async function runDemo(name, btn) {
            const resultDiv = document.getElementById('result-' + name);
            btn.disabled = true;
            btn.textContent = '运行中...';
            resultDiv.className = 'result show';
            resultDiv.textContent = '⏳ 正在执行...';
            
            try {
                const response = await fetch('/api/' + name);
                const data = await response.json();
                resultDiv.textContent = JSON.stringify(data, null, 2);
            } catch (err) {
                resultDiv.textContent = '❌ 错误: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.textContent = '运行演示';
            }
        }
    </script>
</body>
</html>'''
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def demo_autograd(self):
        """自动微分演示"""
        np.random.seed(42)
        
        # 创建 Tensor
        x = Tensor([2.0, 3.0, 4.0], requires_grad=True)
        
        # 计算 y = x^2 + 2x
        y = x ** 2 + x * 2
        
        # 求和得到标量
        loss = y.sum()
        
        # 反向传播
        loss.backward()
        
        result = {
            "演示": "自动微分 (Autograd)",
            "说明": "计算 y = x² + 2x 的梯度",
            "输入x": x.data.tolist(),
            "y = x² + 2x": y.data.tolist(),
            "loss = sum(y)": float(loss.data),
            "梯度 dy/dx = 2x + 2": x.grad.tolist(),
            "验证": {
                "x=2时梯度": "2*2+2=6 ✓",
                "x=3时梯度": "2*3+2=8 ✓", 
                "x=4时梯度": "2*4+2=10 ✓"
            }
        }
        self.send_json(result)
    
    def demo_linear(self):
        """线性层演示"""
        np.random.seed(42)
        
        # 创建线性层
        linear = Linear(4, 3)
        
        # 输入数据
        x = Tensor(np.random.randn(2, 4), requires_grad=True)
        
        # 前向传播
        y = linear(x)
        
        # 反向传播
        loss = y.sum()
        loss.backward()
        
        result = {
            "演示": "线性层 (Linear Layer)",
            "说明": "y = x @ W.T + b",
            "配置": {
                "输入特征": 4,
                "输出特征": 3,
                "权重形状": list(linear.weight.shape),
                "偏置形状": list(linear.bias.shape)
            },
            "输入x形状": list(x.shape),
            "输出y形状": list(y.shape),
            "输出y值": y.data.tolist(),
            "梯度": {
                "dx形状": list(x.grad.shape),
                "dW形状": list(linear.weight.grad.shape),
                "db形状": list(linear.bias.grad.shape)
            }
        }
        self.send_json(result)
    
    def demo_conv(self):
        """卷积层演示"""
        np.random.seed(42)
        
        # 创建卷积层和池化层
        conv = Conv2d(1, 4, kernel_size=3, padding=1)
        pool = MaxPool2d(kernel_size=2)
        relu = ReLU()
        
        # 输入数据 (batch=1, channels=1, height=8, width=8)
        x = Tensor(np.random.randn(1, 1, 8, 8), requires_grad=True)
        
        # 前向传播
        h1 = conv(x)
        h2 = relu(h1)
        y = pool(h2)
        
        result = {
            "演示": "卷积神经网络层",
            "流程": "Conv2d → ReLU → MaxPool2d",
            "输入": {
                "形状": "(1, 1, 8, 8)",
                "说明": "(batch, channels, height, width)"
            },
            "Conv2d": {
                "输入通道": 1,
                "输出通道": 4,
                "卷积核": "3x3",
                "padding": 1,
                "输出形状": list(h1.shape)
            },
            "ReLU": {
                "输出形状": list(h2.shape)
            },
            "MaxPool2d": {
                "池化核": "2x2",
                "输出形状": list(y.shape)
            },
            "特征图统计": {
                "最小值": float(np.min(y.data)),
                "最大值": float(np.max(y.data)),
                "均值": float(np.mean(y.data))
            }
        }
        self.send_json(result)
    
    def demo_rnn(self):
        """RNN 演示"""
        np.random.seed(42)
        
        # 创建 RNN 单元
        rnn_cell = RNNCell(input_size=4, hidden_size=8)
        
        # 序列数据 (3个时间步)
        seq_len = 3
        batch_size = 2
        
        # 初始隐藏状态
        h = Tensor(np.zeros((batch_size, 8)), requires_grad=True)
        
        hidden_states = []
        for t in range(seq_len):
            x_t = Tensor(np.random.randn(batch_size, 4), requires_grad=True)
            h = rnn_cell(x_t, h)
            hidden_states.append(h.data.tolist())
        
        result = {
            "演示": "循环神经网络 (RNN)",
            "说明": "h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)",
            "配置": {
                "输入大小": 4,
                "隐藏大小": 8,
                "序列长度": seq_len,
                "批次大小": batch_size
            },
            "隐藏状态演变": {
                f"t={t}": {
                    "形状": f"({batch_size}, 8)",
                    "均值": round(np.mean(hidden_states[t]), 4),
                    "范围": f"[{round(np.min(hidden_states[t]), 2)}, {round(np.max(hidden_states[t]), 2)}]"
                }
                for t in range(seq_len)
            },
            "说明": "tanh 激活使隐藏状态保持在 [-1, 1] 范围内"
        }
        self.send_json(result)
    
    def demo_train(self):
        """训练演示"""
        np.random.seed(42)
        
        # 创建简单模型
        linear1 = Linear(4, 8)
        relu = ReLU()
        linear2 = Linear(8, 3)
        
        # 损失函数和优化器
        criterion = CrossEntropyLoss()
        params = linear1.parameters() + linear2.parameters()
        optimizer = SGD(params, lr=0.1)
        
        # 生成数据
        X = np.random.randn(10, 4)
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        
        # 训练记录
        history = []
        
        for epoch in range(5):
            x_tensor = Tensor(X, requires_grad=True)
            y_tensor = Tensor(y)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            h = relu(linear1(x_tensor))
            logits = linear2(h)
            
            # 计算损失
            loss = criterion(logits, y_tensor)
            
            # 计算准确率
            pred = np.argmax(logits.data, axis=1)
            acc = np.mean(pred == y)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            history.append({
                "epoch": epoch + 1,
                "loss": round(float(loss.data), 4),
                "accuracy": round(float(acc), 4)
            })
        
        result = {
            "演示": "完整训练流程",
            "模型": "Linear(4→8) → ReLU → Linear(8→3)",
            "数据": {
                "样本数": 10,
                "输入特征": 4,
                "类别数": 3
            },
            "优化器": "SGD (lr=0.1)",
            "损失函数": "CrossEntropyLoss",
            "训练历史": history,
            "结论": f"损失从 {history[0]['loss']} 降到 {history[-1]['loss']}，准确率从 {history[0]['accuracy']} 升到 {history[-1]['accuracy']}"
        }
        self.send_json(result)
    
    def log_message(self, format, *args):
        """简化日志输出"""
        print(f"[{self.log_date_time_string()}] {args[0]}")


def main():
    port = 8080
    server = HTTPServer(('0.0.0.0', port), DemoHandler)
    print("=" * 60)
    print("NumpyTorch 演示服务器")
    print("=" * 60)
    print(f"\n🚀 服务已启动: http://localhost:{port}")
    print(f"\n📖 可用的 API:")
    print(f"   GET /           - 演示页面")
    print(f"   GET /api/health - 健康检查")
    print(f"   GET /api/autograd - 自动微分演示")
    print(f"   GET /api/linear - 线性层演示")
    print(f"   GET /api/conv   - 卷积层演示")
    print(f"   GET /api/rnn    - RNN 演示")
    print(f"   GET /api/train  - 训练流程演示")
    print(f"\n按 Ctrl+C 停止服务\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.shutdown()


if __name__ == "__main__":
    main()
