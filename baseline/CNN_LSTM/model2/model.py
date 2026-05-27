import torch
import torch.nn as nn


class NLOS_CNN_LSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=32,
                 num_layers=2, num_filters=32,
                 kernel_size=3, dropout=0.5):  # 增大dropout默认值
        super(NLOS_CNN_LSTM, self).__init__()

        # CNN层：提取局部特征
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=num_filters,
                      kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2,
                      kernel_size=kernel_size, padding=kernel_size // 2),  # 新增卷积层
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # LSTM层：捕捉时序依赖
        self.lstm = nn.LSTM(
            input_size=num_filters * 2,  # 输入维度为CNN输出通道数
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 双向LSTM，增强时序捕捉能力
        )

        # 全连接层：分类输出
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),  # 双向LSTM输出维度翻倍
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # 输入形状：[Batch, Seq, Feat] → 转换为CNN输入：[Batch, Feat, Seq]
        x = x.permute(0, 2, 1)
        x = self.cnn(x)

        # 转换为LSTM输入：[Batch, Seq, Feat]
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)

        # 取双向LSTM最后一个时间步的输出（拼接前后向）
        out = torch.cat([out[:, -1, :self.lstm.hidden_size],
                         out[:, 0, self.lstm.hidden_size:]], dim=1)

        # 分类输出
        logits = self.fc(out)
        return logits


# --- 模型测试 ---
if __name__ == "__main__":
    batch_size = 32
    seq_len = 10
    input_features = 7
    model = NLOS_CNN_LSTM(input_size=input_features)
    print("模型结构：")
    print(model)

    dummy_input = torch.randn(batch_size, seq_len, input_features)
    print(f"\n输入数据形状: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"输出数据形状: {output.shape}")
    print(f"输出值示例 (前5个): \n{output[:5].detach().numpy()}")

    assert output.shape == (batch_size, 1), "输出形状错误！应为 (Batch, 1)"
    print("\n✅ 模型测试通过！形状匹配。")