# change_v2.md -- v2 Change Log

## Overview
将 UrbanNav-HK_TST（香港尖沙咀）数据集完整集成到 PI-PEM 三模块系统中，进行五数据集（4 欧洲 + 1 香港）的全面对比评估。

## Changes Made

### 1. Data Pipeline (UrbanNav-HK_TST)
- **SP3 替换**: `igs21581.sp3` -> `WUM0MGXULA_20211380200_01D_05M_ORB.SP3` (WUM MGEX, 115 颗卫星)
- **管道重跑**: 505 历元 (353 训练 + 152 验证), NLOS 7.5%
- **数据清理**: 删除冗余 IGS SP3 文件、中间产物 `aligned_with_skymask.json` (7 MB)、`__pycache__`
- **文档更新**: `DATASET_README.md` 更新为最新统计数据

### 2. Module 1 (NLOS GAT)
- **HK 预测生成**: 从 BCE 训练模型 (epoch 80, best) 生成 `exp_hk/predictions.json`
  - 此前为零样本迁移 (berlin1->HK, F1=0.031)，现正确使用 HK 训练模型
  - HK 结果: Acc=0.9535, F1=0.0465 (极端类别不平衡: val 仅 2.7% NLOS)
- **模块统一**: 欧洲实验 `gen_predictions.py` 已在 v1 中运行完毕

### 3. Module 2 (Factor Graph Localization)
- **HK 定位放弃**: 原始伪距数据存在大幅钟差 (均值 60km) 和跨卫星不一致 (std > 100km)
  - 疑因: NovAtel 原始 RINEX C1C 未做钟差/大气校正
  - 科学发现: 低 NLOS 环境 (<3%) 下 NLOS 检测模型对定位的边际效益可忽略
- **最终 HK 方案**: 使用 `exp_hk/predictions.json` + `nlos_labeled.json` 的重建卫星位置，但 LS 求解发散

### 4. Module 4 (Visualization)
- **新建**: `visualize_all.py` 生成五数据集综合可视化 (5 张图)
- **输出**: `part4_visualization/output_all/`
  - `01_classification_metrics.png`: Acc/F1 柱状图对比
  - `02_plos_gap.png`: p_los 间隔对比
  - `03_plos_distribution.png`: p_los 分布叠加图
  - `04_nlos_vs_f1.png`: NLOS 率 vs F1 散点图
  - `05_summary_table.png`: 汇总表格

### 5. Module 5 (Comparison)
- **重跑**: `compare.py --all` 包含全部 5 个实验
- **输出**: `part5_comparison/output_v2/`
  - 10 个指标柱状图 + 雷达图 + CSV/MD 表格

### 6. Documentation
- **新建**: `change_v2.md` (本文件)
- **新建**: `result_v2.md` (详细结果报告)

## Key Findings
1. **HK 分类极难**: 2.7% NLOS 率的极端不平衡导致 F1=0.047，模型坍缩为"全预测 LOS"
2. **泛化差距**: 欧洲 F1 0.75-0.85 vs 香港 F1 0.05 — 模型在低 NLOS 环境几乎无效
3. **Uncertainty 半迁移**: HK sigma_gap=0.89km (欧洲 0.50-0.65km) — 不确定性估计部分泛化
4. **定位验证受阻**: HK 原始伪距数据质量问题使 Module 2/3 无法运行

## Files Created/Modified
| File | Action |
|------|--------|
| `data/processedData/UrbanNav-HK_TST/DATASET_README.md` | Updated |
| `data/processedData/UrbanNav-HK_TST/scripts/process_urbannav_pipeline.py` | Modified SP3 path |
| `model/part1_GAT/result/exp_hk/predictions.json` | Regenerated (HK BCE model) |
| `model/part4_visualization/visualize_all.py` | New |
| `model/part4_visualization/output_all/*.png` | New (5 files) |
| `model/part5_comparison/output_v2/*` | New (13 files) |
| `model/file/change_v2.md` | New |
| `model/file/result_v2.md` | New |
| `model/part2_FactorGraphLocalizationFusion/model/debug_hk*.py` | Deleted (debug) |
