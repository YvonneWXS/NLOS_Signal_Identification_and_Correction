import torch
import torch.nn as nn


class TC_CNN_BiLSTM(nn.Module):
    """
    论文复现: TC CNN-BiLSTM (Three Channel CNN-BiLSTM)
    参考: Table 1 [cite: 86] & Figure 4 [cite: 75]
    """
    def __init__(self, input_features=7, hidden_size=64,
                 num_filters=64, kernel_size=3, dropout=0.5):
        super(TC_CNN_BiLSTM, self).__init__()
        
        # 论文使用的是 3通道输入 (Original, Real, Imag)
        # 在 datasetGenerator 中我们已经将特征拼接为 (Features * 3)
        # 因此这里的 in_channels 是 input_features * 3
        self.in_channels = input_features * 3
        
        # --- 1. Convolution Layer Block ---
        # 论文[cite: 86]: Conv -> BN -> ReLU -> MaxPool
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, 
                      out_channels=num_filters,
                      kernel_size=kernel_size, 
                      padding=kernel_size // 2), # 保持序列长度
            nn.BatchNorm1d(num_filters),         # [cite: 121] 加快学习速率
            nn.ReLU(),                           # [cite: 121] 稀疏激活性
            nn.Dropout(p=dropout),
            nn.MaxPool1d(kernel_size=2)          # [cite: 122] 减小参数并保留关键特征
        )

        # --- 2. BiLSTM Layer ---
        # [cite: 54, 125] 结合 CNN 特征进行时序分析
        self.lstm = nn.LSTM(
            input_size=num_filters,      # CNN 输出的通道数
            hidden_size=hidden_size,
            num_layers=1,                # 论文 Table 2 [cite: 148] 显示 1 层效果最好
            batch_first=True,
            bidirectional=True           # [cite: 54] 双向
        )

        # --- 3. Dropout & FC Layer ---
        self.dropout = nn.Dropout(p=0.2) # Table 1 [cite: 86] 设置为 0.2
        
        # BiLSTM 输出维度是 hidden_size * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 二分类
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features*3]
        
        # 1. 调整维度以适应 Conv1d: [Batch, Channels, Seq_Len]
        x = x.permute(0, 2, 1) 
        
        # 2. CNN 前向传播
        cnn_out = self.cnn(x) 
        # cnn_out shape: [Batch, num_filters, Seq_Len/2] (因为 MaxPool)
        
        # 3. 调整维度以适应 LSTM: [Batch, Seq_Len_New, Input_Size]
        cnn_out = cnn_out.permute(0, 2, 1)
        
        # 4. BiLSTM 前向传播
        lstm_out, _ = self.lstm(cnn_out)
        # lstm_out shape: [Batch, Seq_Len_New, Hidden_Size * 2]
        
        # 5. 取最后一个时间步 (Sequence Classification)
        # 或者使用 Attention (原论文未提及 Attention，使用的是 Flatten 或 Last Step)
        # 根据论文 [cite: 54] "拼接...作为最终特征向量"，通常取最后一步
        final_feat = lstm_out[:, -1, :] 
        
        # 6. Dropout & Classification
        final_feat = self.dropout(final_feat)
        logits = self.fc(final_feat)
        
        return logits


# --- 模型测试 ---
if __name__ == "__main__":
    batch_size = 32
    seq_len = 10
    input_features = 7
    model = TC_CNN_BiLSTM(input_features=input_features)
    print("模型结构：")
    print(model)

    dummy_input = torch.randn(batch_size, seq_len, input_features*3)
    print(f"\n输入数据形状: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"输出数据形状: {output.shape}")
    print(f"输出值示例 (前5个): \n{output[:5].detach().numpy()}")

    assert output.shape == (batch_size, 1), "输出形状错误！应为 (Batch, 1)"
    print("\n✅ 模型测试通过！形状匹配。")