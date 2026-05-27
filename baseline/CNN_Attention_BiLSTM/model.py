import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention Mechanism:
    计算 LSTM 所有时间步的加权平均 (Context Vector)
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # 这里的 hidden_size 是 BiLSTM 的输出维度 (即 hidden_size * 2)
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output shape: [Batch, Seq_Len, Hidden_Size * 2]
        
        # 1. 计算注意力分数 (Score)
        # attn_weights shape: [Batch, Seq_Len, 1]
        attn_scores = self.attn(lstm_output) 
        
        # 2. 归一化分数得到权重 (Weights)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 3. 加权求和得到上下文向量 (Context Vector)
        # context shape: [Batch, Hidden_Size * 2]
        # 广播乘法: [B, S, H] * [B, S, 1] -> Sum over S
        context = torch.sum(lstm_output * attn_weights, dim=1)
        
        return context, attn_weights

class NLOS_CNN_Attention_LSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=64,
                 num_layers=2, num_filters=64,
                 kernel_size=3, dropout=0.5):
        super(NLOS_CNN_Attention_LSTM, self).__init__()

        # --- 1. CNN Block (特征提取) ---
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=num_filters,
                      kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # MaxPool 会减少序列长度 (例如 10 -> 5)，有助于提取显著特征并减少计算量
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2,
                      kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # --- 2. BiLSTM Block (时序建模) ---
        self.lstm = nn.LSTM(
            input_size=num_filters * 2,  # 输入维度匹配 CNN 输出通道
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # --- 3. Attention Block (关键时刻聚焦) ---
        # BiLSTM 输出维度是 hidden_size * 2
        self.attention = Attention(hidden_size * 2)

        # --- 4. Fully Connected Block (分类) ---
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1) # 二分类输出
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features]
        
        # 1. CNN 处理
        # 变换维度满足 Conv1d 要求: [Batch, Features, Seq_Len]
        x = x.permute(0, 2, 1)
        cnn_out = self.cnn(x)
        
        # 变换维度满足 LSTM 要求: [Batch, Seq_Len_New, Channels]
        # 注意：由于 MaxPool，这里的 Seq_Len_New 会变短
        cnn_out = cnn_out.permute(0, 2, 1)
        
        # 2. LSTM 处理
        # lstm_out shape: [Batch, Seq_Len_New, Hidden_Size * 2]
        lstm_out, _ = self.lstm(cnn_out)
        
        # 3. Attention 聚合
        # context shape: [Batch, Hidden_Size * 2]
        context, attn_weights = self.attention(lstm_out)
        
        # 4. 分类层
        logits = self.fc(context)
        
        return logits

# --- 模型测试与验证 ---
if __name__ == "__main__":
    # 模拟参数
    batch_size = 32
    seq_len = 10
    input_features = 7
    
    # 初始化模型
    model = NLOS_CNN_Attention_LSTM(input_size=input_features)
    print("=== 模型结构 ===")
    print(model)

    # 创建虚拟输入
    dummy_input = torch.randn(batch_size, seq_len, input_features)
    print(f"\n输入数据形状: {dummy_input.shape}")

    # 前向传播
    output = model(dummy_input)
    print(f"输出数据形状: {output.shape}")
    
    # 验证
    assert output.shape == (batch_size, 1), f"输出形状错误！期望 (32, 1)，实际 {output.shape}"
    print("\n✅ CNN-Attention-BiLSTM 测试通过！")