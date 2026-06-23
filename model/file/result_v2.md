# result_v2.md -- NLOS GAT Five-Dataset Evaluation Results

## Experiment Summary

| Experiment | Dataset | NLOS% | Epochs | Accuracy | F1 | p_los Gap | sigma_gap (km) |
|-----------|---------|:-----:|:------:|:--------:|:---:|:---------:|:--------------:|
| exp_001 | Berlin1 Potsdamer Platz | 46.1% | 1,377 | 0.8474 | 0.8425 | — | 0.534 |
| exp_002 | Berlin2 Gendarmenmarkt | 45.6% | 5,925 | 0.8524 | 0.8489 | — | 0.600 |
| exp_003 | Frankfurt1 Maintower | 52.0% | 5,851 | 0.8296 | 0.8399 | — | 0.502 |
| exp_004 | Frankfurt2 Westendtower | 25.3% | 3,575 | 0.8659 | 0.7473 | — | 0.651 |
| exp_hk | Hong Kong TST | 2.7% | 505 | 0.9535 | 0.0465 | -0.019 | 0.892 |

## Key Findings

### 1. Classification Performance vs NLOS Rate
- **欧洲四城**: F1 0.75-0.85，模型在 >25% NLOS 环境中有效
- **香港**: F1 0.047，在 2.7% NLOS 环境中模型坍缩为"全预测 LOS"
- **泛化性**: 跨地理区域的 NLOS 检测严重依赖 NLOS 基础率

### 2. Uncertainty (sigma) Analysis
- **欧洲**: sigma_gap 0.50-0.65 km，NLOS 卫星的预测不确定性显著高于 LOS
- **香港**: sigma_gap 0.89 km，不确定性估计部分泛化（即使分类失败）
- **含义**: 模型的 heteroscedastic uncertainty 头比分类头泛化能力强

### 3. p_los Distribution
- **欧洲**: 双峰分布明显 (LOS 峰 ~0.8, NLOS 峰 ~0.2)
- **香港**: 单峰集中在 ~0.93，模型无法区分 LOS/NLOS

### 4. Module 2/3 Limitation
- 香港原始伪距数据存在 ~60km 均值钟差 + >100km 跨卫星标准差
- 疑因 NovAtel RINEX C1C 未做钟差/大气层校正
- 低 NLOS 环境 (<3%) 中 NLOS 检测对定位边际效益可忽略

### 5. Cross-Dataset Comparison Summary

```
Dataset       Acc     F1      sigma_gap   NLOS%
Berlin1       0.847   0.843   0.534       46.1%
Berlin2       0.852   0.849   0.600       45.6%
Frankfurt1    0.830   0.840   0.502       52.0%
Frankfurt2    0.866   0.747   0.651       25.3%
HongKong      0.954   0.047   0.892        2.7%
```

## Visualizations

Located in `part4_visualization/output_all/`:
- `01_classification_metrics.png`: Accuracy/F1 bar chart
- `02_plos_gap.png`: p_los gap comparison
- `03_plos_distribution.png`: p_los distribution overlay (5 datasets)
- `04_nlos_vs_f1.png`: NLOS rate vs F1 scatter
- `05_summary_table.png`: Summary table

## Recommendations

1. **低 NLOS 场景**: BCE-only 训练不足以处理 <3% NLOS 率，需考虑 anomaly detection 框架
2. **Uncertainty 头**: 泛化能力优于分类头，可考虑作为跨域迁移的锚点
3. **HK 定位**: 需获取校正后的伪距数据或广播星历重新处理
4. **数据增强**: HK 数据 NLOS 率 7.5% (训练集 9.1%)，但验证集仅 2.7% — 时间分布偏移需注意
