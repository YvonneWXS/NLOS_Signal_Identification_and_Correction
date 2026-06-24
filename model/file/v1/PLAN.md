# GNSS NLOS 代码重构与评估 — 完整实施计划

**版本**: v2  
**日期**: 2026-06-24  
**优先级**: P0 + P1 + P2  
**源码来源**: `model_2/`（只读，不修改）  
**目标位置**: `model/`（新建模块化结构）

---

## Phase 0: Git 初始化与源码审计

### 步骤 0.1 — Git 备份当前状态
- 确认 `model_2/` 和 `data/` 安全
- `git add model/file/v1/` + `git commit -m "v2: restructuring plan docs"`
- `git push origin master`

### 步骤 0.2 — 源码审计
遍历 `model_2/` 下所有子目录，记录每个文件的当前功能与复用方式：

**Module 1 源 (`model_2/part1_GAT/model/`)**:
| 源文件 | 审计结论 | 目标归属 |
|--------|---------|---------|
| `GAT_V2026.py` | 核心文件，含模型定义+训练+损失+推理，需拆分 | → `module1_nlos/` 多个文件 |
| `GAT_V2025.py` | 旧版，保留不迁移 | 仅参考 |
| `config.py` | 集中配置，需转为 YAML | → `module1_nlos/config.yaml` |
| `analyze_experiment.py` | 分析逻辑 | → `module1_nlos/inference.py` 部分 |
| `run_full_training.py` | 训练入口 | → `module1_nlos/run.py` |
| `run_serial.py` | 串行训练脚本 | → `scripts/` |
| `sp3_reader.py` | SP3 星历解析 | → `common/sp3_reader.py` |
| `Data_read.py` | 数据读取 | → `module1_nlos/data_loader.py` |
| `NodeFeature_Generate.py` | 特征工程 | → `module1_nlos/features.py` |
| `Depth_Adj_Generate.py` / `Radio_Depth_Generate.py` | 图构建辅助 | → `module1_nlos/features.py` |
| `analyze_model.py` / `analyze_mog.py` | 分析工具 | → `module1_nlos/inference.py` 分析函数 |
| `generate_report.py` | 报告生成 | → `module5_visualization/generate_report.py` |
| `train_wrapper.py` | 训练包装 | → `module1_nlos/trainer.py` |
| `gen_predictions.py` | 推理导出 | → `module1_nlos/inference.py` |
| `positioning_test.py` | 定位测试 | → `module4_experiments/tests/` |
| `New_axis40.txt` / `stations_position.txt` | 静态数据 | → `module1_nlos/` 保留 |

**Module 2 源 (`model_2/part2_FactorGraphLocalizationFusion/model/`)**:
| 源文件 | 审计结论 | 目标归属 |
|--------|---------|---------|
| `fusion/utils.py` | 坐标转换+指标 | → `common/coordinate.py` + `common/metrics.py` |
| `fusion/evaluate_fusion.py` | 评估逻辑 | → `module2_localization/` 各方法文件 |
| `fusion/factor_graph_fusion.py` | FG 实现 | → `module2_localization/factor_graph.py` |
| `fusion/los_anchored_ls.py` | LS 变体 | → `module2_localization/standard_ls.py` |
| `fusion/baselines.py` | WLS/Hard-threshold | → `module2_localization/wls.py` + `hard_threshold.py` |
| `fusion/motion_geometry_predictor.py` | PRNC 预测器 | → `module2_localization/prnc.py` |
| `fusion/train_tcn.py` | TCN 训练 | → `module4_experiments/baseline/` |
| `fusion/platt_calibration.py` | (如果没有则需新建) | → `module2_localization/platt_calibration.py` |
| `fusion/debug_geometry.py` / `diagnose_weighting.py` / `verify_clock_contamination.py` / `verify_nlos_sign.py` | 诊断脚本 | → `module4_experiments/` 诊断工具 |
| `run_fusion.py` / `run_positioning.py` / `run_pos_quick.py` / `run_all.py` / `hk_inference.py` | 运行入口 | → `module2_localization/run.py` 合并 |

**Module 3 源 (`model_2/part3_ResidualFeedbackAndOnline_Correction/model/`)**:
| 源文件 | 审计结论 | 目标归属 |
|--------|---------|---------|
| `residual_feedback.py` | 残差跟踪核心 | → `module3_adaptive/tracker.py` |
| `shift_detector.py` | 场景检测 | → `module3_adaptive/detector.py` |
| `run_module3.py` | 自适应选择器+运行入口 | → `module3_adaptive/selector.py` + `run.py` |
| `evaluate_module3.py` | 评估 | → `module3_adaptive/` 保留 |
| `cross_module_validation.py` | 交叉验证 | → `module4_experiments/` |
| `diagnosis_v3.py` | 诊断 | → `module4_experiments/` |
| `posterior_correction.py` | 后验修正 | → `module3_adaptive/` |
| `reproduce_paper_results.py` | 复现脚本 | → `module4_experiments/` |
| `run_ablation.py` | 消融实验 | → `module4_experiments/` |

**Module 4/5 源 (`model_2/part4_visualization/` + `part5_comparison/`)**:
| 源文件 | 审计结论 | 目标归属 |
|--------|---------|---------|
| `visualize.py` / `visualize_all.py` | 可视化核心 | → `module5_visualization/` 拆分为 8 个文件 |
| `compare.py` | 对比分析 | → `module5_visualization/baseline_comparison_viz.py` |

将审计结果写入 `model/file/v1/findings.md`。

### 步骤 0.3 — Git 备份
- `git commit -m "v2: source audit complete — findings.md"`
- `git push origin master`

---

## Phase 1: 目录骨架 + 公共库提取

### 步骤 1.1 — 创建目录结构
```
model/
├── module1_nlos/tests/
├── module2_localization/tests/
├── module3_adaptive/tests/
├── module4_experiments/tests/baseline/
├── module5_visualization/tests/
├── common/tests/
├── scripts/
├── results/
├── data/                    # 软链接或复制 data/ 的结构信息
├── requirements.txt
├── setup.py
├── pytest.ini
├── .gitignore
└── README.md
```

### 步骤 1.2 — 提取公共库 `common/`

| 目标文件 | 功能 | 源 |
|---------|------|-----|
| `common/__init__.py` | 包初始化 | 新建 |
| `common/coordinate.py` | LLA↔ECEF, ECEF↔ENU, 方位角/仰角计算 | 从 `model_2/part2_.../fusion/utils.py` 提取 |
| `common/sp3_reader.py` | SP3 精密星历解析与插值 | 从 `model_2/part1_GAT/model/sp3_reader.py` 复制 |
| `common/time_utils.py` | GPS 周/秒 ↔ datetime 转换 | 从 `model_2/part2_.../fusion/utils.py` 提取 |
| `common/metrics.py` | CEP50/CEP95/RMSE/MAE/STD/median | 从各模块散落的指标函数归并 |
| `common/logger.py` | 统一日志（控制台+文件+TensorBoard） | 新建 |
| `common/config_manager.py` | YAML 配置加载、合并、CLI 覆盖、验证 | 新建 |

每个 `common/*.py` 写完立即配 `common/tests/test_*.py`，至少包含：
- 正常输入 → 输出形状/类型正确
- 边界输入（空数组、单样本）→ 不崩溃
- 异常输入 → 抛出合理异常

### 步骤 1.3 — 创建 `common/README.md`
按 8 章节模板：概述、架构、配置、使用、API、依赖、测试、FAQ。

### 步骤 1.4 — Git 备份
- `git add model/common/ model/requirements.txt model/setup.py model/pytest.ini model/.gitignore`
- `git commit -m "v2: Phase 1 — common library + directory skeleton"`
- `git push origin master`

---

## Phase 2: Module 1 重构 — NLOS 感知与误差建模

### 步骤 2.1 — 拆分源文件

从 `model_2/part1_GAT/model/GAT_V2026.py` 拆出以下文件：

| 目标文件 | 包含内容 | 说明 |
|---------|---------|------|
| `module1_nlos/data_loader.py` | `GNSSDataset` 类、`collate_fn`、`create_dataloaders` | sky_mask 插值、卫星几何、缓存加载。block-diagonal batching 逻辑保留 |
| `module1_nlos/features.py` | `NodeFeatureGenerator`、`GraphBuilder` | 11 维特征（仰角/方位角/CNO/prStdev/伪距误差等）+ 边构造（方位角差 < threshold） |
| `module1_nlos/model.py` | `GATLayer`（向量化版）、`GATMoG` 网络定义 | 纯模型，无训练逻辑。输出：p_los, mu_nlos, sigma_nlos, sigma_los |
| `module1_nlos/loss.py` | `MoGLoss` 类 | 三阶段：pure_bce → blend → full_nll。含 sigma 分离约束、mu 正则、方向约束 |
| `module1_nlos/trainer.py` | `Trainer` 类 | 训练循环、AMP、梯度累积、early stopping、checkpoint 保存、TensorBoard 日志 |
| `module1_nlos/inference.py` | `InferenceEngine` 类 | 加载 checkpoint → 前向推理 → 输出 p_los/mu/sigma → 分析报告生成 |

额外从源文件引入：
- `module1_nlos/New_axis40.txt` + `stations_position.txt` → 静态参考数据
- `module1_nlos/sp3_reader.py` → 软链接到 `common/sp3_reader.py`

### 步骤 2.2 — 创建 `module1_nlos/config.yaml`
```yaml
# 完整参数见 goal_v1.md 附录 7.1，核心字段：
dataset:
  name: berlin1
  path: data/berlin1/
  train_val_split: 0.7

model:
  architecture: gat_mog
  in_features: 11
  hidden_features: 128
  num_layers: 2
  num_heads: 8
  dropout: 0.1

training:
  num_epochs: 100
  batch_size: 32
  use_block_diagonal_batching: true
  learning_rate: 5.0e-6
  weight_decay: 1.0e-4
  gradient_clip: 10.0

loss:
  pure_bce_epochs: 8
  blend_epochs: 25
  lambda_bce: 0.6
  lambda_direction: 1.0
  lambda_sigma_sep: 5.0
  mu_nlos_target: 0.30
  lambda_mu_reg: 0.20

early_stopping:
  patience: 15
  delta: 0.001

logging:
  use_tensorboard: true
  tensorboard_dir: runs/
  log_frequency: 10

device: cuda:0
```

### 步骤 2.3 — 创建 `module1_nlos/run.py`
```bash
# 训练模式
python -m model.module1_nlos.run --config module1_nlos/config.yaml --dataset berlin1 --mode train

# 推理模式（从已有 checkpoint）
python -m model.module1_nlos.run --dataset berlin1 --mode inference --checkpoint path/to/best_model.pth

# 指定输出目录
python -m model.module1_nlos.run --dataset berlin1 --mode train --output results/exp_test/
```

### 步骤 2.4 — 创建 `module1_nlos/tests/`
- `test_data_loader.py` — 从 `model_2/` 复制一份小的测试数据，验证 DataLoader 输出 tensor 维度正确
- `test_model.py` — 构造 (N=10, D=11) 随机输入，验证 GATMoG 前向传播输出形状：p_los (N,), mu_nlos (N,), sigma_nlos (N,), sigma_los (N,)
- `test_trainer.py` — 用 5 个 epoch 的小数据集验证训练循环不崩溃，loss 下降

### 步骤 2.5 — 创建 `module1_nlos/README.md`
按 8 章节模板：概述、架构（文件依赖图）、配置（每参数含义+调参建议）、使用（命令行+Python API）、API（核心类签名）、依赖、测试、FAQ。

### 步骤 2.6 — 验证 checkpoint 可用
- 从 `model_2/part1_GAT/result/` 复制已有 `best_model.pth` 到 `module1_nlos/`
- 运行 `python -m model.module1_nlos.run --dataset berlin1 --mode inference`，验证推理输出非空且维度正确

### 步骤 2.7 — Git 备份
- `git add model/module1_nlos/`
- `git commit -m "v2: Phase 2 — Module 1 NLOS GAT restructured"`
- `git push origin master`

---

## Phase 3: Module 2 重构 — 融合定位

### 步骤 3.1 — 创建 `module2_localization/base.py`
```python
class LocalizationBase(ABC):
    """所有定位方法的统一基类"""
    def __init__(self, config: Dict, name: str): ...
    
    @abstractmethod
    def solve(self,
              observations: np.ndarray,       # (N,) 伪距观测值
              sv_positions: np.ndarray,       # (N, 3) 卫星 ECEF 位置
              sv_systems: np.ndarray = None,  # (N,) 星座 ID
              additional_info: Dict = None    # MoG 输出等辅助信息
              ) -> Tuple[np.ndarray, float, Dict]:
        """
        Returns:
            position: (3,) ECEF
            clock_bias: float (meters)
            details: {'converged': bool, 'iterations': int, 'residuals': ndarray, ...}
        """
    
    def validate_input(self, observations, sv_positions): ...
```

### 步骤 3.2 — 迁移 5 个已有方法

| 目标文件 | 源 | 关键逻辑 |
|---------|-----|---------|
| `standard_ls.py` | `model_2/part2_.../fusion/los_anchored_ls.py` | 全卫星最小二乘，4 状态（x,y,z,dt） |
| `wls.py` | `model_2/part2_.../fusion/baselines.py` | WLS-elevation（仰角加权）+ WLS-MoG（p_los 加权 + sigma 加权） |
| `hard_threshold.py` | `model_2/part2_.../fusion/baselines.py` | p_los > 0.5 的卫星用于 LS |
| `factor_graph.py` | `model_2/part2_.../fusion/factor_graph_fusion.py` | 因子图优化（L-BFGS-B）+ 多起点 + MoG 残差因子 |
| `platt_calibration.py` | 新建/提取 | Platt 缩放校准 p_los 概率 |

### 步骤 3.3 — 实现 8 个新基线

| 基线方法 | 目标文件 | 实现要点 |
|---------|---------|---------|
| C/N0 加权 LS | `cno_weighted.py` | 观测权重 ∝ C/N0，其余同 LS |
| SNR 加权 LS | `snr_weighted.py` | 观测权重 ∝ SNR (dB → linear) |
| RAIM | `raim.py` | 残差检验：χ² 检测 → 排除最大残差卫星 → 重解 LS，迭代至通过或 <4 颗 |
| IRLS | `irls.py` | Huber 损失：残差 < k 时二次，> k 时线性。迭代至收敛 |
| EKF | `kalman.py` | 状态 (x,y,z,dt,vx,vy,vz,dt_dot)。匀速模型预测 + 伪距观测更新 |
| DNN 端到端 | `dnn.py` | 输入 (N, features) → MLP → (3,) ECEF。需训练 |
| GAT 端到端 | `gat_e2e.py` | 输入图 → GAT → 全局池化 → (3,) ECEF。需训练 |
| INS/GNSS | `ins_gnss.py` | 松耦合：GNSS 位置 → EKF 校正 IMU 积分；紧耦合：伪距 → EKF 校正 |

每个新基线实现后立即配单元测试（输入伪数据验证输出维度+不崩溃）。

### 步骤 3.4 — 创建 `module2_localization/factory.py`
```python
class LocalizationFactory:
    _registry: Dict[str, Type[LocalizationBase]] = {}
    
    @classmethod
    def register(cls, name: str, method_cls: Type[LocalizationBase]): ...
    
    @classmethod
    def create(cls, name: str, config: Dict) -> LocalizationBase: ...
    
    @classmethod
    def list_methods(cls) -> List[str]: ...
```

所有 13 个方法在文件末尾通过 `@LocalizationFactory.register('name')` 自动注册。

### 步骤 3.5 — 创建支持文件
- `module2_localization/config.yaml` — 各方法参数（RAIM 阈值、EKF 噪声协方差等）
- `module2_localization/run.py` — CLI 入口
- `module2_localization/README.md` — 8 章节模板
- `module2_localization/tests/` — `test_base.py` / `test_ls.py` / `test_factory.py` 等

### 步骤 3.6 — Git 备份
- `git add model/module2_localization/`
- `git commit -m "v2: Phase 3 — Module 2 localization (5 existing + 8 new baselines)"`
- `git push origin master`

---

## Phase 4: Module 3 重构 — 自适应选择

### 步骤 4.1 — 拆分源文件

| 目标文件 | 源 | 核心逻辑 |
|---------|-----|---------|
| `module3_adaptive/tracker.py` | `model_2/part3_.../model/residual_feedback.py` | `ResidualInnovationTracker`：滑动窗口 (window_size=50) 跟踪残差创新统计量（均值/方差/自相关），输出创新评分 |
| `module3_adaptive/detector.py` | `model_2/part3_.../model/shift_detector.py` | `SceneQualityDetector`：三指标加权评分（p_los_gap 33% + pdop_ratio 33% + nlos_redundancy 34%），输出场景质量 0-1 |
| `module3_adaptive/selector.py` | `model_2/part3_.../model/run_module3.py` | `AdaptivePositionSelector`：综合评分 ≥ 0.65 → FG，≥ 0.50 → WLS，< 0.50 → LS。含安全回退（相对 LS > 5% 则回退 LS） |
| `module3_adaptive/posterior_correction.py` | `model_2/part3_.../model/posterior_correction.py` | 后验修正模块 |

### 步骤 4.2 — 创建支持文件
- `module3_adaptive/config.yaml` — 阈值、权重、窗口大小
- `module3_adaptive/run.py` — CLI
- `module3_adaptive/README.md` — 8 章节
- `module3_adaptive/tests/`

### 步骤 4.3 — Git 备份
- `git add model/module3_adaptive/`
- `git commit -m "v2: Phase 4 — Module 3 adaptive selector restructured"`
- `git push origin master`

---

## Phase 5: Module 4 构建 — 实验框架

### 步骤 5.1 — 新建文件

| 文件 | 功能 |
|------|------|
| `module4_experiments/config.yaml` | 实验全局配置（数据集列表、方法列表、参数扫描范围） |
| `module4_experiments/param_sweep.py` | `ParamSweepEngine`：接受参数名+值列表，遍历运行 M1→M2→M3，收集 CEP50/CEP95，输出 CSV + 热力图数据 |
| `module4_experiments/baseline_runner.py` | `BaselineRunner`：遍历 dataset × method，调用 Module 2 factory + Module 3 selector，输出每个组合的 CEP50 |
| `module4_experiments/batch_scheduler.py` | `BatchScheduler`：生成 PowerShell 并行脚本，将 N 个独立实验分配到 M 个进程 |
| `module4_experiments/results_aggregator.py` | 扫描 `results/` 目录，聚合各实验的 `metrics.json` 到统一 CSV/JSON |
| `module4_experiments/statistical_test.py` | `wilcoxon_test(method_a_errors, method_b_errors) -> p_value`。遍历所有方法对，输出 p-value 矩阵 |
| `module4_experiments/run.py` | CLI 入口：`--mode baseline_comparison|param_sweep|statistical_test|generate_report` |
| `module4_experiments/README.md` | 8 章节 |

### 步骤 5.2 — Git 备份
- `git add model/module4_experiments/`
- `git commit -m "v2: Phase 5 — Module 4 experiment framework"`
- `git push origin master`

---

## Phase 6: Module 5 构建 — 可视化

### 步骤 6.1 — 从 `model_2/part4_visualization/visualize.py` 拆分

| 目标文件 | 功能 | 输出示例 |
|---------|------|---------|
| `module5_visualization/trajectory_viz.py` | 2D/3D 轨迹对比（GT vs LS vs WLS vs FG vs Adaptive） | `trajectory_{city}.png` |
| `module5_visualization/error_analysis_viz.py` | CDF 误差分布、箱线图、热力图 | `error_cdf.png` / `error_boxplot.png` |
| `module5_visualization/module1_viz.py` | 混淆矩阵、p_los 分布（LOS vs NLOS）、sigma 分布、mu 分布 | `confusion_matrix.png` / `plos_dist.png` |
| `module5_visualization/module3_viz.py` | 创新时间序列、场景质量时间序列、方法选择比例饼图 | `innovation_ts.png` / `scene_quality.png` |
| `module5_visualization/baseline_comparison_viz.py` | 基线对比柱状图（CEP50）、雷达图（多指标）、排名表 | `baseline_bars.png` / `radar.png` |
| `module5_visualization/param_sweep_viz.py` | 参数扫描折线图、双参数热力图 | `sweep_lines.png` / `heatmap.png` |
| `module5_visualization/generate_report.py` | 聚合所有图表 → Markdown 报告 | `report.md` |
| `module5_visualization/run.py` | CLI：`--module all --dataset all --output_dir results/viz/` | |

### 步骤 6.2 — 创建支持文件
- `module5_visualization/config.yaml` — 图表风格（颜色、字体、DPI）
- `module5_visualization/README.md` — 8 章节

### 步骤 6.3 — Git 备份
- `git add model/module5_visualization/`
- `git commit -m "v2: Phase 6 — Module 5 visualization"`
- `git push origin master`

---

## Phase 7: 批量脚本 + 项目级文件

### 步骤 7.1 — 创建 `scripts/`

| 脚本 | 功能 |
|------|------|
| `scripts/run_all_experiments.ps1` | 一键运行完整实验周期：`param($Dataset, $OutputDir)` → M1(infer) → M2(13 methods) → M3(select) → M5(viz) → 聚合 |
| `scripts/baseline_comparison.ps1` | 仅运行基线对比：4 数据集 × 13 方法并行 |
| `scripts/param_sweep.ps1` | 运行所有 P0 参数扫描 |
| `scripts/generate_report.ps1` | 生成最终报告 |

### 步骤 7.2 — 创建项目级文件

| 文件 | 内容 |
|------|------|
| `model/README.md` | 项目概述、目录结构、快速开始（3 条命令跑通）、各模块简介、引用格式 |
| `model/requirements.txt` | `torch>=1.9.0`, `torch_geometric>=2.0`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `pyyaml`, `tensorboard`, `pytest` |
| `model/setup.py` | `setuptools.setup()` 最小安装脚本 |
| `model/pytest.ini` | `[pytest] testpaths = module1_nlos/tests module2_localization/tests module3_adaptive/tests module4_experiments/tests module5_visualization/tests common/tests` |
| `model/.gitignore` | `__pycache__/`, `*.pyc`, `results/`, `runs/`, `.idea/`, `*.pth` (除 checkpoint 目录外), `data/` |

### 步骤 7.3 — Git 备份
- `git add model/scripts/ model/README.md model/requirements.txt model/setup.py model/pytest.ini model/.gitignore`
- `git commit -m "v2: Phase 7 — scripts + project-level files"`
- `git push origin master`

---

## Phase 8: 集成测试

### 步骤 8.1 — 端到端流程验证

以 berlin1 为测试数据集，跑通完整链路：

```powershell
# Step 1: Module 1 推理（使用已有 checkpoint）
python -m model.module1_nlos.run --dataset berlin1 --mode inference --checkpoint model_2/part1_GAT/result/exp_001/best_model.pth --output results/integration_test/

# Step 2: Module 2 运行所有 13 个方法
python -m model.module2_localization.run --dataset berlin1 --input results/integration_test/module1_output.pkl --methods all --output results/integration_test/

# Step 3: Module 3 自适应选择
python -m model.module3_adaptive.run --dataset berlin1 --input results/integration_test/ --output results/integration_test/

# Step 4: Module 5 可视化
python -m model.module5_visualization.run --dataset berlin1 --module all --input results/integration_test/ --output_dir results/integration_test/viz/
```

### 步骤 8.2 — 修复集成问题
常见问题及修复：
- 路径不一致 → 统一使用 `pathlib.Path` + 相对路径
- 数据格式不匹配 → 在 `common/config_manager.py` 添加 schema 验证
- 缺失依赖 → 补 `requirements.txt`

### 步骤 8.3 — 验证 `condition.md` 自动生成
检查 `results/integration_test/condition.md` 是否包含：时间戳、数据集、所有参数、Python 版本、Git commit hash。

### 步骤 8.4 — Git 备份
- `git commit -m "v2: Phase 8 — integration test passed"`
- `git push origin master`

---

## Phase 9: P0 实验运行

### 步骤 9.1 — 基线对比（4 数据集 × 13 方法 = 52 组合）
```powershell
# 使用 batch_scheduler 生成并行脚本
python -m model.module4_experiments.run --mode baseline_comparison --datasets berlin1,berlin2,frankfurt1,frankfurt2 --methods all --parallel 4 --output results/baseline_comparison/
```

输出：
- `results/baseline_comparison/ranking.csv` — 每个数据集的 CEP50 排名
- `results/baseline_comparison/ranking.png` — 柱状图
- `results/baseline_comparison/radar.png` — 多维度雷达图
- `results/baseline_comparison/condition.md`

### 步骤 9.2 — P0 参数扫描
```powershell
# 单参数扫描
python -m model.module4_experiments.run --mode param_sweep --param fg_threshold --values 0.50,0.55,0.60,0.65,0.70,0.75,0.80 --datasets all --output results/sweep_fg_threshold/

python -m model.module4_experiments.run --mode param_sweep --param window_size --values 20,50,100,200 --datasets all --output results/sweep_window_size/
```

### 步骤 9.3 — 统计检验
```powershell
python -m model.module4_experiments.run --mode statistical_test --input results/baseline_comparison/ --output results/statistical_tests/
```

输出：`p_value_matrix.csv`（每种方法对的 Wilcoxon p-value）

### 步骤 9.4 — Git 备份
- `git add results/baseline_comparison/ results/sweep_*/ results/statistical_tests/`
- `git commit -m "v2: Phase 9 — P0 experiments complete"`
- `git push origin master`

---

## Phase 10: P1 消融与敏感性

### 步骤 10.1 — 系统消融实验

| 消融变量 | 变体 | 运行命令 |
|---------|------|---------|
| Module 1 阶段数 | 三阶段 (BCE→Blend→NLL) / 两阶段 (BCE→NLL) / 单阶段 (纯 BCE) | 修改 `config.yaml` 中 `loss.pure_bce_epochs` + `blend_epochs` |
| Module 2 MoG 使用 | 有 MoG (p_los+sigma+mu) / 无 MoG (仅 p_los 二分类) | `additional_info` 传不同字段 |
| Module 3 残差跟踪 | 有 / 无 `ResidualInnovationTracker` | `config.yaml` 中 `residual_tracking.enabled` |
| Module 3 场景检测 | 有 / 无 `SceneQualityDetector` | 只用创新评分 vs 综合评分 |
| Module 3 安全回退 | 有 / 无 fallback | `config.yaml` 中 `fallback.enabled` |

```powershell
python -m model.module4_experiments.run --mode ablation --datasets all --output results/ablation/
```

### 步骤 10.2 — 双参数敏感性分析

学习率 × lambda_bce 联合热力图：
```powershell
python -m model.module4_experiments.run --mode param_sweep_2d --param1 learning_rate --values1 1e-6,2e-6,5e-6,1e-5 --param2 lambda_bce --values2 0.2,0.4,0.6,0.8,1.0 --datasets berlin1 --output results/sweep_lr_lambda/
```

### 步骤 10.3 — Git 备份
- `git commit -m "v2: Phase 10 — P1 ablation + sensitivity"`
- `git push origin master`

---

## Phase 11: P2 扩展 + 最终报告

### 步骤 11.1 — 多起点优化对比
```powershell
python -m model.module4_experiments.run --mode param_sweep --param multistart --values 1,3,5,10 --method factor_graph --datasets all --output results/sweep_multistart/
```

### 步骤 11.2 — 交叉验证
```powershell
python -m model.module4_experiments.run --mode cross_validation --folds 4 --datasets all --output results/cross_validation/
```

### 步骤 11.3 — HK 数据外域测试
- 使用 `model_2/part1_GAT/model/` 中的 HK checkpoint
- 运行 M1(infer) → M2 → M3 → 评估
- 作为泛化性附录，不作为核心结论

### 步骤 11.4 — 最终报告生成
```powershell
python -m model.module5_visualization.run --mode generate_report --input results/ --output results/FINAL_REPORT.md
```

报告内容：
- 实验总览
- 基线排名表 + 柱状图
- 参数扫描热力图
- 消融实验结论
- 统计显著性矩阵
- 核心发现（Standard LS 在 3/4 城市最优，FG 仅在 Frankfurt1 有效）
- 投稿建议（目标期刊、需补充实验）
- 19 张规范化图表

### 步骤 11.5 — 最终 Git 备份
- `git add results/FINAL_REPORT.md results/cross_validation/ results/sweep_multistart/`
- `git commit -m "v2: Phase 11 — final report + P2 experiments"`
- `git tag -a v2.0 -m "Code restructuring complete — 5 modules, 13 baselines, full evaluation"`
- `git push origin master --tags`

---

## 约束与规则

| 规则 | 说明 |
|------|------|
| Git 备份 | 每个 Phase 结束后 commit + push |
| 禁止删除 | `model_2/`、`data/` 在任何情况下不修改、不删除 |
| 源码只读 | 从 `model_2/` 复制代码到 `model/`，不在 `model_2/` 内修改 |
| 中断恢复 | 每个 Phase 的输出独立可验证，中断后从中断点继续 |
| 编码 | 所有 `.py` UTF-8，所有 `.md` UTF-8 |
| 路径 | 所有路径使用 Windows 绝对路径或 `pathlib.Path` |

---

## 关键决策记录

| 决策 | 结论 | 原因 |
|------|------|------|
| Module 1 权重 | 使用已有 checkpoint，不重新训练 | 节省时间，已有结果足够 |
| 新基线实现优先级 | 先保证能跑通，性能调优放 P2 | 尽快完成框架验证 |
| 实验并行度 | 4 进程并行 | RTX 5060 + CPU 可控 |
| HK 数据处理 | P2 最后一环，非核心 | 已有诊断结论（F1=0.047） |
| 配置文件格式 | YAML | 可读性好，支持注释，PyYAML 零依赖 |
