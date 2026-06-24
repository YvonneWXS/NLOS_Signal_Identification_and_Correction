# GNSS NLOS定位系统 - 完整重构与评估Goal文档

**项目版本**：v2  
**目标受众**：Codex AI + 工程师  
**生成时间**：2026-06-24  
**实验数据基础**：result_v2.md, change_v2.md  
**评估范围**：欧洲四城市（暂不含香港）

---

## 一、项目总览

### 1.1 背景与现状

#### 已完成工作（v1）
- ✅ Module 1（NLOS GAT）：柏林、法兰克福四城市模型训练完成，F1 0.75-0.89
- ✅ Module 2（融合定位）：实现标准LS、WLS、因子图等5种定位方法，评估了CEP50性能
- ✅ Module 3（自适应选择）：残差跟踪 + 场景检测 + 自适应选择器，在欧洲数据上表现良好
- ✅ 初步的对比实验与可视化（part4, part5）

#### 存在的问题（v1的局限）
1. **代码结构散乱**：模块之间边界不清，难以维护和扩展
2. **缺乏标准接口**：模块间数据传递格式不统一，硬编码路径较多
3. **配置管理混乱**：参数分散在代码各处，难以批量修改
4. **实验框架不完整**：参数扫描、基线对比缺乏系统化设计
5. **文档不充分**：模块使用说明不足，新人上手困难
6. **可视化不规范**：图表命名、输出位置不统一

#### 本次重构目标（v2）
1. **代码规范化**：模块化、接口化、可测试化、完整文档化
2. **实验框架建立**：参数扫描引擎 + 基线对比系统 + 批量运行脚本
3. **可视化升级**：系统化、多维度、专业级输出
4. **成果评估**：判断是否足以投稿，明确补充实验需求

### 1.2 关键约束

| 约束条件 | 说明 |
|---------|------|
| **数据集** | 仅使用欧洲四城市（Berlin1/2, Frankfurt1/2），暂不含香港 |
| **模型** | 不重新训练Module 1，使用已有权重（exp_001-004） |
| **重点** | 代码结构 + 对比框架 + 可视化 + 评估 |
| **运行环境** | Windows（需PowerShell和.bat脚本支持） |

---

## 二、代码重构详细设计

### 2.1 目录结构规范

**根目录**：`D:\3_document\4_research\NLOS Signal Identification and Correction\model`

```
model/
├── module1_nlos/                    # 【模块1】NLOS感知与误差建模
│   ├── README.md                    # ⭐ 完整模块文档（必需）
│   ├── __init__.py
│   ├── config.yaml                  # 模块专属配置
│   ├── data_loader.py               # 数据加载（sky_mask插值、卫星几何）
│   ├── features.py                  # 11维特征 + 图构建
│   ├── model.py                     # GAT+MoG网络定义
│   ├── loss.py                      # 三阶段损失函数（BCE→Blend→NLL）
│   ├── trainer.py                   # 训练循环 + TensorBoard
│   ├── inference.py                 # 推理接口
│   ├── run.py                       # ⭐ 模块独立入口（python -m module1_nlos.run）
│   └── tests/
│       ├── __init__.py
│       ├── test_data_loader.py      # 测试数据加载
│       ├── test_model.py            # 测试模型前向传播
│       └── test_trainer.py          # 测试训练循环
│
├── module2_localization/            # 【模块2】融合定位
│   ├── README.md                    # ⭐ 完整模块文档
│   ├── __init__.py
│   ├── config.yaml
│   ├── base.py                      # LocalizationBase抽象类（统一接口）
│   ├── standard_ls.py               # 标准最小二乘
│   ├── wls.py                       # WLS（elevation/MoG加权）
│   ├── factor_graph.py              # 因子图优化（L-BFGS-B）
│   ├── factory.py                   # 定位方法工厂 + 注册机制
│   ├── raim.py                      # RAIM残差检测（基线）
│   ├── irls.py                      # 迭代加权LS（基线）
│   ├── kalman.py                    # EKF定位（基线）
│   ├── dnn.py                       # DNN端到端回归（基线）
│   ├── gat_e2e.py                   # GAT端到端定位（基线）
│   ├── ins_gnss.py                  # INS/GNSS松耦合和紧耦合（基线）
│   ├── platt_calibration.py         # Platt概率校准
│   ├── run.py                       # ⭐ 模块独立入口
│   └── tests/
│       ├── test_base.py
│       ├── test_ls.py
│       └── test_factor_graph.py
│
├── module3_adaptive/                # 【模块3】自适应选择
│   ├── README.md                    # ⭐ 完整模块文档
│   ├── __init__.py
│   ├── config.yaml
│   ├── tracker.py                   # ResidualInnovationTracker（滑动窗口）
│   ├── detector.py                  # SceneQualityDetector（三指标加权）
│   ├── selector.py                  # AdaptivePositionSelector（含安全回退）
│   ├── run.py                       # ⭐ 模块独立入口
│   └── tests/
│
├── module4_experiments/             # 【模块4】对比实验框架
│   ├── README.md                    # ⭐ 完整模块文档
│   ├── __init__.py
│   ├── config.yaml                  # 实验配置（数据集、参数扫描范围）
│   ├── param_sweep.py               # 参数扫描引擎 + 网格搜索
│   ├── baseline_runner.py           # 基线对比运行器
│   ├── batch_scheduler.py           # 批量实验调度器（.ps1生成）
│   ├── results_aggregator.py        # 结果聚合与导出（CSV/JSON）
│   ├── statistical_test.py          # Wilcoxon符号秩检验
│   ├── run.py                       # ⭐ 模块独立入口
│   ├── tests/
|   ├── baseline/                    # 基线代码文件夹，每个子文件夹里是一个完整的基线实验框架
│
├── module5_visualization/           # 【模块5】可视化模块
│   ├── README.md                    # ⭐ 完整模块文档
│   ├── __init__.py
│   ├── config.yaml                  # 可视化风格配置
│   ├── trajectory_viz.py            # 轨迹对比图
│   ├── error_analysis_viz.py        # 误差分析（CDF、箱线图、热力图）
│   ├── module1_viz.py               # Module 1可视化（混淆矩阵、p_los分布）
│   ├── module3_viz.py               # Module 3可视化（创新时间序列、场景质量）
│   ├── baseline_comparison_viz.py   # 基线对比可视化
│   ├── param_sweep_viz.py           # 参数扫描结果可视化（折线、热力图）
│   ├── generate_report.py           # 报告生成（汇聚所有可视化）
│   ├── run.py                       # ⭐ 模块独立入口
│   └── tests/
│
├── common/                          # 【公共库】被各模块共享
│   ├── README.md                    # ⭐ 公共库文档
│   ├── __init__.py
│   ├── coordinate.py                # LLA↔ECEF, ECEF↔ENU转换
│   ├── sp3_reader.py                # SP3星历解析与插值
│   ├── time_utils.py                # GPS时间转换工具
│   ├── metrics.py                   # CEP50/CEP95等评估指标
│   ├── logger.py                    # 统一日志系统
│   ├── config_manager.py            # 配置文件加载与验证
│   └── tests/
│
├── data/                            # 数据目录（软链接或实际数据，直接复制D:\3_document\4_research\NLOS Signal Identification and Correction\data文件夹下的内容，不要删除原本的文件夹）
│   ├── berlin1/
│   ├── berlin2/
│   ├── frankfurt1/
│   ├── frankfurt2/
│   └── sp3/
│
├── results/                         # 结果输出目录（自动生成）
│   ├── <dataset>_<timestamp>/       # 每次运行自动生成
│   │   ├── condition.md             # 运行条件记录
│   │   ├── module1/                 # 各模块输出
│   │   ├── module2/
│   │   ├── module3/
│   │   ├── metrics.json
│   │   └── visualizations/
│   └── experiments/
│       ├── param_sweep_<name>/      # 参数扫描结果
│       ├── baseline_comparison/     # 基线对比结果
│       └── summary.json             # 汇总表
│
├── scripts/                         # 批量运行脚本
│   ├── run_all_experiments.ps1      # PowerShell批量脚本（推荐）
│   ├── run_all_experiments.bat      # 批处理脚本（备选）
│   ├── param_sweep.ps1              # 参数扫描脚本
│   └── generate_report.ps1          # 报告生成脚本
│
├── requirements.txt                 # 环境依赖
├── setup.py                         # 安装脚本
├── pytest.ini                       # pytest配置
├── .gitignore
├── README.md                        # ⭐ 项目级总文档
└── CHANGELOG.md                     # 版本变更记录

```

### 2.2 模块 README.md 规范模板

每个模块的 `README.md` 必须包含以下8个章节，确保使用者无需阅读源码即可完整理解和使用：

#### **模板内容**

```markdown
# Module X: [模块名称]

## 1. 模块概述

### 1.1 功能定位
本模块负责 [简明功能描述]，在整个PI-PEM系统中的角色是 [角色说明]。

### 1.2 核心流程
```
输入 → [处理步骤1] → [处理步骤2] → 输出
```

### 1.3 输入与输出
- **输入**：
  - `xxx.json` - [字段说明]
  - `yyy.pkl` - [字段说明]
- **输出**：
  - `zzz.pkl` - 每个历元包含：{字段1, 字段2, ...}

## 2. 架构设计

### 2.1 模块内部结构
```
module_x/
├── submodule_a.py  → 功能A
├── submodule_b.py  → 功能B
└── submodule_c.py  → 功能C的工具类
```

### 2.2 核心类/函数依赖关系
[用表格或Mermaid图展示核心类和函数的依赖关系]

### 2.3 数据流示意
[详细描述数据从输入到输出的全过程]

## 3. 配置说明

### 3.1 config.yaml 参数表
| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| param1 | type | default | range | 说明（含调参建议） |
| ... | ... | ... | ... | ... |

### 3.2 配置示例
```yaml
module:
  param1: value1
  param2: value2
```

## 4. 使用方法

### 4.1 独立运行（命令行）
```bash
# 基本用法
python -m modelX_name.run --config config.yaml --dataset berlin1

# 显示帮助
python -m modelX_name.run --help

# 常见用法
python -m modelX_name.run --dataset berlin1,berlin2 --mode train_and_infer
python -m modelX_name.run --load_model checkpoint.pth --mode inference_only
```

### 4.2 作为库被调用
```python
from model.moduleX_name import CoreClass, load_data

# 示例代码
data = load_data('path/to/data')
result = CoreClass(config).process(data)
```

### 4.3 输入数据格式
- 文件路径：`data/<dataset>/`
- 数据结构：{字段1: ..., 字段2: ...}
- 格式要求：JSON / pickle / HDF5

### 4.4 输出结果说明
- **输出位置**：`results/<dataset>_<timestamp>/moduleX/`
- **文件清单**：
  - `output_primary.pkl` - 主要输出
  - `output_secondary.json` - 辅助输出
  - `metrics.json` - 评估指标
- **字段说明**：[详细列出输出中的每个字段]

## 5. 核心API参考

### 5.1 主要类
```python
class MainClass:
    """[类功能说明]"""
    
    def __init__(self, config: Dict):
        """
        参数：
            config: 配置字典
        """
        pass
    
    def process(self, data: Dict) -> Dict:
        """
        [方法说明]
        
        参数：
            data: 输入数据
        
        返回：
            处理后的结果
        
        异常：
            ValueError: [说明何时抛出]
        """
        pass
```

### 5.2 主要函数
```python
def load_data(path: str) -> Dict:
    """[函数说明]"""
    pass
```

## 6. 依赖关系说明

### 6.1 内部依赖
- 依赖 `Model.common.coordinate` 进行坐标转换
- 依赖 `Model.moduleY_name` 的输出作为输入

### 6.2 外部依赖
- `torch>=1.9.0`：神经网络框架
- `torch_geometric>=2.0`：图神经网络库
- `scipy>=1.7`：科学计算

### 6.3 依赖关系图
```
moduleX ← common, moduleY
  ↓
moduleZ
```

## 7. 测试与质量保证

### 7.1 如何运行测试
```bash
# 运行该模块的所有测试
pytest model/moduleX_name/tests/ -v

# 运行特定测试
pytest model/moduleX_name/tests/test_core.py::TestClassName::test_method_name -v

# 生成覆盖率报告
pytest model/moduleX_name/tests/ --cov=model.moduleX_name --cov-report=html
```

### 7.2 测试覆盖范围
- 数据加载器：正常/异常输入
- 核心处理逻辑：单元测试
- 集成测试：全流程测试

### 7.3 测试通过标准
- 单元测试通过率 ≥95%
- 核心功能覆盖率 ≥80%

## 8. 常见问题与故障排查

### Q1: [常见问题1]
**症状**：[问题表现]
**原因**：[根本原因]
**解决方案**：
```bash
# 具体步骤
```

### Q2: [常见问题2]
**症状**：...
**解决方案**：...

### Q3: 如何修改参数？
编辑 `config.yaml` 或命令行传入：
```bash
python -m moduleX_name.run --param1 value1 --param2 value2
```

### Q4: 如何调试？
启用debug日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 9. 性能指标与优化建议

### 9.1 典型运行时间
- 数据加载：~X秒
- 处理逻辑：~Y秒
- 总耗时：~Z秒（样本规模≈10k）

### 9.2 内存使用
- 峰值内存：~XXX MB
- 优化建议：使用梯度累积或批处理

## 10. 版本历史
- v2.0（2026-06-24）：重构，规范接口
- v1.0（2026-05-XX）：初版实现
```

### 2.3 统一接口设计

#### **Module 2 定位基类**

```python
# model/module2_localization/base.py

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np

class LocalizationBase(ABC):
    """
    定位方法的统一基类。
    所有定位方法（LS、WLS、FG等）都应继承此类，
    实现 solve() 方法，确保输入输出接口一致。
    """
    
    def __init__(self, config: Dict, name: str):
        """
        参数：
            config: 方法特有的配置字典
            name: 方法名称（用于日志和结果记录）
        """
        self.config = config
        self.name = name
        self.converged = False
        self.iterations = 0
    
    @abstractmethod
    def solve(self, 
              observations: np.ndarray,      # (N,) 伪距观测值
              sv_positions: np.ndarray,      # (N, 3) 卫星ECEF位置
              sv_systems: np.ndarray = None, # (N,) 星座ID
              additional_info: Dict = None   # 辅助信息（如MoG输出、IMU等）
              ) -> Tuple[np.ndarray, float, Dict]:
        """
        求解定位问题。
        
        返回：
            position: (3,) 接收机ECEF位置
            clock_bias: 时钟偏差（单位：米）
            details: 求解细节字典，包含：
                - 'converged': bool，是否收敛
                - 'iterations': int，迭代次数
                - 'residuals': (N,) 残差向量
                - 'covariance': (4,4) 协方差矩阵（可选）
                - 'dop': float，DOP值（可选）
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """返回方法的显示名称"""
        pass
    
    def validate_input(self, observations, sv_positions):
        """验证输入数据的合法性"""
        if len(observations) < 4:
            raise ValueError(f"至少需要4颗卫星，当前{len(observations)}颗")
        if observations.shape[0] != sv_positions.shape[0]:
            raise ValueError("观测值和卫星位置数量不匹配")
```

#### **Module 2 定位方法工厂**

```python
# model/module2_localization/factory.py

class LocalizationFactory:
    """
    定位方法工厂，支持动态注册和创建定位器。
    """
    
    _methods = {}
    
    @classmethod
    def register(cls, name: str, method_class):
        """注册新的定位方法"""
        cls._methods[name] = method_class
    
    @classmethod
    def create(cls, method_name: str, config: Dict) -> LocalizationBase:
        """创建指定的定位器实例"""
        if method_name not in cls._methods:
            raise ValueError(f"Unknown method: {method_name}")
        return cls._methods[method_name](config)
    
    @classmethod
    def list_methods(cls):
        """列出所有已注册的方法"""
        return list(cls._methods.keys())

# 模块加载时自动注册所有方法
from .standard_ls import StandardLS
from .wls import WLS_Elevation, WLS_MoG
from .factor_graph import FactorGraphMoG
from .raim import RAIM
# ... 其他方法

LocalizationFactory.register('standard_ls', StandardLS)
LocalizationFactory.register('wls_elevation', WLS_Elevation)
LocalizationFactory.register('wls_mog', WLS_MoG)
LocalizationFactory.register('fg_mog', FactorGraphMoG)
LocalizationFactory.register('raim', RAIM)
```

#### **统一运行接口**

```python
# model/module2_localization/run.py

def main():
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Module 2: 融合定位')
    parser.add_argument('--dataset', choices=['berlin1', 'berlin2', 'frankfurt1', 'frankfurt2'])
    parser.add_argument('--methods', nargs='+', 
                       choices=LocalizationFactory.list_methods() + ['all'],
                       default=['standard_ls', 'wls_mog', 'fg_mog'])
    parser.add_argument('--load_mog', type=Path, help='Module 1的MoG预测文件')
    parser.add_argument('--output_dir', type=Path, default='results/')
    args = parser.parse_args()
    
    # 运行定位
    for method_name in (args.methods if args.methods != ['all'] else LocalizationFactory.list_methods()):
        locator = LocalizationFactory.create(method_name, config)
        results = locator.solve(observations, sv_positions, additional_info={'mog': mog_pred})
        # 保存结果

if __name__ == '__main__':
    main()
```

### 2.4 TensorBoard 集成（Module 1）

```python
# model/module1_nlos/trainer.py

from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, config, log_dir='runs/'):
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def train_epoch(self, epoch):
        """训练一个epoch，并记录TensorBoard"""
        
        # 记录损失函数分量
        self.writer.add_scalars('Loss', {
            'bce': bce_loss.item(),
            'direction': direction_loss.item(),
            'mog_nll': nll_loss.item(),
            'total': total_loss.item(),
        }, epoch)
        
        # 记录梯度范数（监控梯度爆炸）
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = np.sqrt(total_norm)
        self.writer.add_scalar('Gradient/norm', total_norm, epoch)
        
        # 记录验证集指标
        self.writer.add_scalars('Validation', {
            'f1': val_f1,
            'accuracy': val_accuracy,
            'p_los_gap': p_los_gap,
        }, epoch)
        
        # 记录p_los分布直方图（每10个epoch）
        if epoch % 10 == 0:
            self.writer.add_histogram('p_los_dist', all_p_los, epoch)
    
    def close(self):
        self.writer.close()
```

### 2.5 结果输出规范

每次运行必须自动生成以下目录结构：

```
results/
├── berlin1_20260624_143022/          # <dataset>_<timestamp>/
│   ├── condition.md                  # 运行条件记录（必需）
│   ├── module1/
│   │   ├── model_best.pth
│   │   ├── predictions.pkl           # MoG预测结果
│   │   ├── metrics.json              # F1, Accuracy等
│   │   └── runs/                     # TensorBoard日志
│   │       └── events.out.tfevents.*
│   ├── module2/
│   │   ├── positioning_results.pkl   # 所有方法的定位结果
│   │   └── comparison.csv            # 各方法CEP50/CEP95
│   ├── module3/
│   │   └── adaptive_results.pkl      # 自适应选择结果
│   ├── metrics.json                  # 全局评估指标
│   └── visualizations/               # 输出图表
│       ├── trajectory_2d.png
│       ├── error_cdf.png
│       └── ...
│
└── experiments/
    ├── param_sweep_<name>/           # 参数扫描结果
    │   ├── summary.csv               # 汇总表格
    │   └── heatmap_<param1>_<param2>.png
    └── baseline_comparison/
        ├── summary.json
        └── ranking.png
```

#### **condition.md 模板**

```markdown
# Experiment Conditions

**Run ID**: berlin1_20260624_143022
**Dataset**: Berlin1 Potsdamer Platz
**Run Time**: 2026-06-24 14:30:22
**Duration**: 47 min 32 sec
**Git Commit**: abc123f (code version)

## Module 1: NLOS感知
- **模型**: GAT_V2025
- **训练数据**: 276个历元 × ~4k颗卫星
- **验证数据**: 276个历元 × ~4k颗卫星
- **配置参数**:
  - pure_bce_epochs: 8
  - blend_epochs: 25
  - learning_rate: 5e-6
  - batch_size: 32
  - lambda_bce: 0.6
  - lambda_direction: 1.0
  - lambda_sigma_sep: 5.0

## Module 2: 融合定位
- **输入**: Module 1 的MoG预测 + GNSS观测
- **方法**: Standard LS, WLS-elevation, WLS-MoG, Hard-threshold, FG-MoG+2A
- **配置参数**:
  - multistart: 3
  - platt_calibration: true
  - max_iterations: 10

## Module 3: 自适应选择
- **配置参数**:
  - window_size: 50
  - fg_threshold: 0.65
  - wls_threshold: 0.50
  - weights: [0.33, 0.33, 0.34]

## 环境信息
- Python: 3.9.13
- PyTorch: 1.13.0+cu117
- CUDA: 11.7
- Device: NVIDIA GeForce RTX 3090 (24GB)

## 关键结果
- Module 1 Val F1: 0.847
- Module 2 Standard LS CEP50: 904.5m
- Module 2 FG-MoG+2A CEP50: 984.2m
- Module 3 Adaptive CEP50: 902.3m
```

---

## 三、对比实验完整设计

### 3.1 参数扫描表（超完整版）

为了充分探索参数空间和验证假设，以下参数应进行系统扫描。参数扫描按优先级分为三档：

#### **P0优先级（必扫，核心参数）**

| 模块 | 参数 | 类型 | 扫描值 | 默认值 | 说明 |
|------|------|------|--------|--------|------|
| M1 | pure_bce_epochs | int | 0, 4, 8, 12, 16, 20 | 8 | 阶段1长度，探索初始化重要性 |
| M1 | blend_epochs | int | 0, 10, 20, 25, 30, 40 | 25 | 阶段2长度，探索平滑过渡的价值 |
| M1 | learning_rate | float | 1e-6, 2e-6, 5e-6, 1e-5, 2e-5 | 5e-6 | 训练稳定性 |
| M1 | lambda_bce | float | 0.2, 0.4, 0.6, 0.8, 1.0 | 0.6 | BCE权重，影响LOS/NLOS平衡 |
| M1 | hidden_dim | int | 64, 128, 256, 512 | 128 | 模型容量 |
| M2 | multistart | int | 1, 3, 5, 7 | 3 | FG优化起点数，影响收敛质量 |
| M3 | window_size | int | 10, 20, 30, 50, 100, 150, 200 | 50 | 残差窗口，影响自适应响应速度 |
| M3 | fg_threshold | float | 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80 | 0.65 | FG使用阈值，核心控制参数 |

#### **P1优先级（应扫，重要参数）**

| 模块 | 参数 | 类型 | 扫描值 | 默认值 | 说明 |
|------|------|------|--------|--------|------|
| M1 | lambda_direction | float | 0.0, 0.5, 1.0, 2.0, 5.0 | 1.0 | 方向损失权重 |
| M1 | lambda_sigma_sep | float | 0.0, 1.0, 5.0, 10.0 | 5.0 | sigma分离损失权重 |
| M1 | dropout | float | 0.0, 0.05, 0.1, 0.2 | 0.1 | 正则化强度 |
| M2 | platt_calibration | bool | True, False | True | 概率校准是否有效 |
| M3 | wls_threshold | float | 0.30, 0.40, 0.50, 0.60, 0.70 | 0.50 | WLS使用阈值 |
| M3 | min_history | int | 5, 10, 15, 20, 30 | 15 | 最小历史样本数 |

#### **P2优先级（可扫，增强版参数）**

| 模块 | 参数 | 类型 | 扫描值 | 默认值 | 说明 |
|------|------|------|--------|--------|------|
| M1 | num_heads | int | 4, 8, 12, 16 | 8 | 多头数 |
| M2 | max_iterations | int | 5, 10, 20, 50 | 10 | FG迭代次数上限 |
| M3 | weights_plos_gap | float | [0.2, 0.3, 0.4, 0.5, 0.6]（需满足权重和=1） | 0.33 | p_los_gap权重 |
| M3 | weights_pdop_ratio | float | 同上 | 0.33 | DOP权重 |
| M3 | weights_nlos_redundancy | float | 同上 | 0.34 | NLOS冗余权重 |

### 3.2 基线对比方案

#### **已有内部基线（5种）**

这些是当前系统中已实现的定位方法：

1. **Standard LS** - 标准最小二乘，所有卫星等权重
   - 文件：`model/module2_localization/standard_ls.py`
   - 参数：无特殊参数
   - 用途：性能基准

2. **WLS-elevation** - 仰角加权最小二乘
   - 权重公式：w = sin(elevation)
   - 文件：`model/module2_localization/wls.py` (elevation模式)
   - 参数：无特殊参数
   - 用途：观测几何质量基线

3. **WLS-MoG** - Module 1 MoG输出加权最小二乘
   - 权重公式：w = p_los / sigma²
   - 文件：`model/module2_localization/wls.py` (mog模式)
   - 参数：platt_calibration (True/False)
   - 用途：软权重基线

4. **Hard-threshold** - 二值阈值剔除
   - 权重公式：w = 1 if p_los > 0.5 else 0
   - 文件：`model/module2_localization/wls.py` (hard_threshold模式)
   - 参数：threshold (0.4-0.6)
   - 用途：硬剔除的坏例子（预期最差）

5. **FG-MoG+2A** - 因子图优化 + 2个启发式
   - 算法：L-BFGS-B非线性优化
   - 文件：`model/module2_localization/factor_graph.py`
   - 参数：multistart, max_iterations, convergence_threshold
   - 用途：高级方法基线

#### **需要新增的外部基线（8种）**

为了进行充分的对比和验证，需实现以下外部基线：

**信号质量加权（2种）**

| # | 方法名 | 实现要点 | 预期结论 |
|---|--------|---------|---------|
| 6 | C/N0-weighted LS | 权重 = C/N0 / max(C/N0) | 验证深度学习是否优于纯手工特征 |
| 7 | SNR-weighted LS | 权重 = max(0, (SNR-threshold)/range) | 同上，用SNR代替C/N0 |

**鲁棒统计方法（2种）**

| # | 方法名 | 实现要点 | 预期结论 |
|---|--------|---------|---------|
| 8 | RAIM (Receiver Autonomous Integrity Monitoring) | (1) 解算定位; (2) 检测残差异常; (3) 剔除> threshold的卫星; (4) 重新解算 | 硬剔除 vs 软权重的差异 |
| 9 | IRLS (Iterative Reweighted LS) | 用Huber/M估计器自适应调整权重，迭代收敛 | 鲁棒统计对NLOS的效果 |

**时序方法（1种）**

| # | 方法名 | 实现要点 | 预期结论 |
|---|--------|---------|---------|
| 10 | EKF (Extended Kalman Filter) | 运动模型: 恒速或随机游走; 观测方程: GNSS伪距 | 时序处理的益处（vs批处理） |

**深度学习方法（2种）**

| # | 方法名 | 实现要点 | 预期结论 |
|---|--------|---------|---------|
| 11 | DNN端到端 | 输入11维特征 → 3层FC → 输出(x,y,z,clk) | 端到端 vs 模块化 |
| 12 | GAT端到端定位 | 输入图结构 + 特征 → GAT → 输出位置 | 模块化GAT的价值 |

**组合导航基线（1种）**

| # | 方法名 | 实现要点 | 预期结论 |
|---|--------|---------|---------|
| 13 | INS/GNSS松耦合 | IMU预积分(Δv, Δθ) + EKF融合LS定位 | 多传感器融合的上界 |

#### **基线详细实现规范**

```python
# model/module2_localization/new_baseline.py

from .base import LocalizationBase

class CNO_WeightedLS(LocalizationBase):
    """C/N0加权最小二乘"""
    
    def __init__(self, config):
        super().__init__(config, 'C/N0-weighted LS')
    
    def solve(self, observations, sv_positions, sv_systems=None, additional_info=None):
        # 从additional_info中提取C/N0
        cn0_values = additional_info['cn0']  # (N,)
        
        # 计算权重（归一化到[0.1, 1]，防止权重为0）
        cn0_max = np.max(cn0_values)
        weights = np.clip(cn0_values / cn0_max, 0.1, 1.0)
        
        # 迭代WLS
        x_est = np.array([6378000, -2200000, 2400000])
        clock_bias = 0
        
        for iteration in range(20):
            distances = np.linalg.norm(sv_positions - x_est, axis=1)
            direction = (sv_positions - x_est) / distances[:, None]
            
            H = np.column_stack([-direction, np.ones(len(observations))])
            residuals = observations - (distances + clock_bias)
            
            # 加权正规方程
            W = np.diag(weights)
            HTW = H.T @ W
            try:
                delta = np.linalg.solve(HTW @ H, HTW @ residuals)
                x_est = x_est + delta[:3]
                clock_bias = clock_bias + delta[3]
                
                if np.linalg.norm(delta) < 0.01:
                    self.converged = True
                    self.iterations = iteration + 1
                    break
            except np.linalg.LinAlgError:
                break
        
        return x_est, clock_bias, {
            'converged': self.converged,
            'iterations': self.iterations,
            'residuals': residuals,
        }
    
    def get_name(self):
        return 'C/N0-weighted LS'
```

### 3.3 实验矩阵设计

#### **主实验矩阵 (完全因子设计)**

```
实验维度：
  - 数据集：berlin1, berlin2, frankfurt1, frankfurt2 (4个)
  - 模块1参数：6 × 6 × 5 × 5 (P0优先级的参数组合)
  - 方法：13种基线方法 (内部5 + 新增8)
  - Module 3参数：7 × 5 (fg_threshold × window_size)

总实验数：4 × (6×6×5×5) × 13 × (7×5) = 超过 200万次实验 ⚠️

优化策略：
  1. 分阶段扫描：
     - Phase 1: 固定M1, M3参数，对比13种方法 (4 × 13 = 52次)
     - Phase 2: 最优M2方法固定，扫描M1参数 (4 × 180 = 720次)
     - Phase 3: 最优M1配置固定，扫描M3参数 (4 × 35 = 140次)
     - Phase 4: 最优参数组合进行敏感性分析 (4 × 20 = 80次)
  
  2. 并行化：
     - 4个数据集并行运行（4个GPU或CPU核心）
     - 结果自动汇总
```

#### **单参数扫描方案**

```bash
# P0优先级 - 核心参数扫描
python -m model.module4_experiments.param_sweep \
    --dataset berlin1,berlin2,frankfurt1,frankfurt2 \
    --param pure_bce_epochs \
    --values 0 4 8 12 16 20 \
    --fix_other_params default \
    --output_dir results/param_sweep_pbc_epochs/

python -m model.module4_experiments.param_sweep \
    --dataset berlin1,berlin2,frankfurt1,frankfurt2 \
    --param fg_threshold \
    --values 0.50 0.55 0.60 0.65 0.70 0.75 0.80 \
    --output_dir results/param_sweep_fg_threshold/
```

#### **双参数联合扫描方案**

```bash
# 重点：学习率 × λ_bce 的联合效应
python -m model.module4_experiments.param_sweep \
    --dataset berlin1 \
    --param1 learning_rate \
    --values1 1e-6 2e-6 5e-6 1e-5 \
    --param2 lambda_bce \
    --values2 0.2 0.4 0.6 0.8 1.0 \
    --output_dir results/param_sweep_lr_lambda/
```

### 3.4 基线对比运行脚本

```python
# model/module4_experiments/baseline_runner.py

class BaselineComparisonRunner:
    """基线对比运行器"""
    
    def run_all_baselines(self, dataset, output_dir):
        """在指定数据集上运行所有基线方法"""
        
        baseline_configs = {
            'standard_ls': {},
            'wls_elevation': {},
            'wls_mog': {'platt_calibration': True},
            'hard_threshold': {'threshold': 0.5},
            'fg_mog': {'multistart': 3},
            'cno_weighted': {},
            'snr_weighted': {},
            'raim': {'threshold': 3.0},
            'irls': {'robustness_param': 1.5},
            'ekf': {'process_noise': 0.1},
            'dnn_e2e': {'load_model': 'dnn_e2e_model.pth'},
            'gat_e2e': {'load_model': 'gat_e2e_model.pth'},
            'ins_gnss': {'fusion_type': 'loose'},
        }
        
        results_summary = {}
        
        for method_name, method_config in baseline_configs.items():
            print(f"Running baseline: {method_name}...")
            
            try:
                locator = LocalizationFactory.create(method_name, method_config)
                results = locator.solve(observations, sv_positions, additional_info)
                
                # 计算评估指标
                metrics = compute_metrics(results['position'], gt_position)
                
                results_summary[method_name] = {
                    'cep50': metrics['cep50'],
                    'cep95': metrics['cep95'],
                    'mean_error': metrics['mean_error'],
                    'rmse': metrics['rmse'],
                    'converged': results['converged'],
                    'iterations': results.get('iterations', 0),
                    'runtime': results.get('runtime', 0),
                }
            
            except Exception as e:
                print(f"Error running {method_name}: {e}")
                results_summary[method_name] = {'error': str(e)}
        
        # 保存结果
        import json
        with open(f"{output_dir}/baseline_comparison.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # 生成对比表
        self.generate_comparison_table(results_summary, output_dir)
        
        return results_summary
    
    def generate_comparison_table(self, results, output_dir):
        """生成对比表格"""
        import pandas as pd
        
        df = pd.DataFrame(results).T
        df.to_csv(f"{output_dir}/baseline_comparison.csv")
        
        # 打印排名
        print("\n" + "="*60)
        print("Baseline Methods Ranking (by CEP50)")
        print("="*60)
        ranking = df.sort_values('cep50')
        for rank, (method, row) in enumerate(ranking.iterrows(), 1):
            print(f"{rank}. {method:20s} CEP50={row['cep50']:8.1f}m  " 
                  f"CEP95={row['cep95']:8.1f}m")
```

### 3.5 批量运行脚本

#### **PowerShell脚本（推荐）**

```powershell
# scripts/run_all_experiments.ps1

# 配置
$datasetList = @("berlin1", "berlin2", "frankfurt1", "frankfurt2")
$paramSweepList = @(
    @{name="pure_bce_epochs"; values=@(0, 4, 8, 12, 16, 20)},
    @{name="fg_threshold"; values=@(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)},
    @{name="window_size"; values=@(10, 20, 50, 100, 200)}
)

$outputDir = "results/experiments"
$pythonExe = "python"

# 记录开始时间
$startTime = Get-Date
Write-Host "Starting comprehensive experiments at $startTime" -ForegroundColor Green

# Phase 1: 基线对比（所有方法，所有数据集）
Write-Host "`n[Phase 1] Running baseline comparison..." -ForegroundColor Cyan
foreach ($dataset in $datasetList) {
    Write-Host "  Dataset: $dataset"
    & $pythonExe -m model.module4_experiments.run `
        --dataset $dataset `
        --mode baseline_comparison `
        --output $outputDir/baseline_$dataset
}

# Phase 2: 参数扫描 (P0优先级)
Write-Host "`n[Phase 2] Running P0 parameter sweeps..." -ForegroundColor Cyan
foreach ($paramConfig in $paramSweepList) {
    $paramName = $paramConfig.name
    $paramValues = $paramConfig.values -join ","
    
    foreach ($dataset in $datasetList) {
        Write-Host "  Dataset: $dataset, Param: $paramName"
        & $pythonExe -m model.module4_experiments.param_sweep `
            --dataset $dataset `
            --param $paramName `
            --values $paramValues `
            --output $outputDir/sweep_$paramName/$dataset
    }
}

# Phase 3: 敏感性分析 (双参数)
Write-Host "`n[Phase 3] Running sensitivity analysis..." -ForegroundColor Cyan
$sensitivityPairs = @(
    @{p1="learning_rate"; v1=@(1e-6, 2e-6, 5e-6, 1e-5); p2="lambda_bce"; v2=@(0.2, 0.4, 0.6, 0.8)},
    @{p1="fg_threshold"; v1=@(0.50, 0.65, 0.80); p2="window_size"; v2=@(20, 50, 100)}
)

foreach ($pair in $sensitivityPairs) {
    Write-Host "  Analyzing $($pair.p1) × $($pair.p2)"
    & $pythonExe -m model.module4_experiments.param_sweep `
        --dataset berlin1 `
        --param1 $pair.p1 --values1 ($pair.v1 -join ",") `
        --param2 $pair.p2 --values2 ($pair.v2 -join ",") `
        --output $outputDir/sensitivity_$($pair.p1)_$($pair.p2)
}

# Phase 4: 结果汇总与报告生成
Write-Host "`n[Phase 4] Aggregating results..." -ForegroundColor Cyan
& $pythonExe -m model.module4_experiments.run `
    --mode aggregate_results `
    --input_dir $outputDir `
    --output $outputDir/FINAL_REPORT

# 记录结束时间
$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host "`nAll experiments completed in $($duration.TotalHours)h $($duration.Minutes)m" -ForegroundColor Green
```

#### **批处理脚本（备选）**

```batch
:: scripts/run_all_experiments.bat

@echo off
setlocal enabledelayedexpansion

set PYTHON=python
set OUTPUT_DIR=results/experiments
set DATASETS=berlin1 berlin2 frankfurt1 frankfurt2

echo Starting comprehensive experiments...
echo Timestamp: %date% %time%

:: Phase 1: Baseline comparison
echo.
echo [Phase 1] Running baseline comparison...
for %%D in (%DATASETS%) do (
    echo  Dataset: %%D
    %PYTHON% -m model.module4_experiments.run ^
        --dataset %%D ^
        --mode baseline_comparison ^
        --output %OUTPUT_DIR%/baseline_%%D
)

:: Phase 2: Parameter sweeps
echo.
echo [Phase 2] Running parameter sweeps...
for %%D in (%DATASETS%) do (
    echo  Dataset: %%D
    %PYTHON% -m model.module4_experiments.param_sweep ^
        --dataset %%D ^
        --param fg_threshold ^
        --values 0.50 0.55 0.60 0.65 0.70 0.75 0.80 ^
        --output %OUTPUT_DIR%/sweep_fg_threshold/%%D
)

echo.
echo Experiments completed!
pause
```

### 3.6 统计显著性检验

```python
# model/module4_experiments/statistical_test.py

from scipy.stats import wilcoxon, mannwhitneyu
import numpy as np

class StatisticalTester:
    """统计显著性检验"""
    
    @staticmethod
    def wilcoxon_signed_rank_test(method1_errors, method2_errors, alpha=0.05):
        """
        Wilcoxon符号秩检验，比较两个方法的差异是否显著。
        适用于配对样本（同一组历元上的两种方法）。
        """
        differences = method1_errors - method2_errors
        
        statistic, p_value = wilcoxon(differences)
        
        is_significant = p_value < alpha
        
        return {
            'test': 'Wilcoxon Signed-Rank',
            'statistic': statistic,
            'p_value': p_value,
            'significant': is_significant,
            'interpretation': (
                f"Method 1 significantly {'better' if np.mean(differences) < 0 else 'worse'} "
                f"than Method 2 (p={p_value:.4f})"
                if is_significant else
                "No significant difference between methods"
            )
        }
    
    @staticmethod
    def mann_whitney_u_test(method1_errors, method2_errors, alpha=0.05):
        """
        Mann-Whitney U检验，当两组样本量或来源不同时使用。
        """
        statistic, p_value = mannwhitneyu(method1_errors, method2_errors, alternative='two-sided')
        
        is_significant = p_value < alpha
        
        return {
            'test': 'Mann-Whitney U',
            'statistic': statistic,
            'p_value': p_value,
            'significant': is_significant,
        }
    
    @staticmethod
    def generate_statistical_report(results_dict, output_file):
        """
        为所有方法对生成统计检验报告。
        """
        methods = list(results_dict.keys())
        report = []
        
        # 两两比较
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                errors1 = results_dict[method1]['errors']
                errors2 = results_dict[method2]['errors']
                
                test_result = StatisticalTester.wilcoxon_signed_rank_test(
                    np.array(errors1), np.array(errors2)
                )
                
                report.append({
                    'method1': method1,
                    'method2': method2,
                    'mean_error1': np.mean(errors1),
                    'mean_error2': np.mean(errors2),
                    'improvement': (np.mean(errors1) - np.mean(errors2)) / np.mean(errors1) * 100,
                    **test_result
                })
        
        import json
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 打印报告
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE TEST REPORT")
        print("="*80)
        for item in report:
            print(f"\n{item['method1']} vs {item['method2']}:")
            print(f"  Mean Error: {item['mean_error1']:.1f}m vs {item['mean_error2']:.1f}m")
            print(f"  Improvement: {item['improvement']:.2f}%")
            print(f"  p-value: {item['p_value']:.4f} {'✓ SIGNIFICANT' if item['significant'] else '✗ NOT significant'}")
```

---

## 四、可视化模块详细设计

### 4.1 完整可视化清单（19张图）

#### **模块1可视化（欧洲四城）**

| # | 图表名 | 类型 | 数据源 | 命名规范 | 说明 |
|---|--------|------|--------|---------|------|
| 1 | Module 1 混淆矩阵 | Heatmap | 分类结果 | `confusion_matrix_<dataset>.png` | 2×2矩阵，显示LOS/NLOS分类准确度 |
| 2 | p_los分布 | 直方图 | MoG输出 | `p_los_distribution_<dataset>.png` | LOS/NLOS堆叠直方图，横轴p_los [0,1] |
| 3 | p_los Gap对比 | 柱状图 | MoG输出统计 | `p_los_gap_comparison.png` | 四城市p_los_gap的柱状图对比 |
| 4 | 训练曲线（多指标） | 折线图 | TensorBoard日志 | `training_curves_<dataset>.png` | 包含Loss、F1、Accuracy三条线 |

#### **定位精度可视化（欧洲四城，多基线）**

| # | 图表名 | 类型 | 数据源 | 命名规范 | 说明 |
|---|--------|------|--------|---------|------|
| 5 | CEP50对比柱状图 | 柱状分组 | Module 2结果 | `cep50_comparison_all_methods.png` | X轴=4城市，每个城市显示13种方法的柱子 |
| 6 | CEP95对比柱状图 | 柱状分组 | Module 2结果 | `cep95_comparison_all_methods.png` | 同上，CEP95指标 |
| 7 | 误差CDF曲线 | 折线图（多线）| Module 2结果 | `cdf_curves_berlin1.png` | 同一城市内13种方法的CDF，不同颜色 |
| 8 | 方法排名热力图 | Heatmap | Module 2结果 | `method_ranking_heatmap.png` | 行=城市，列=方法，色阶=CEP50（低值更好） |
| 9 | 基线方法运行时间 | 散点图 | Module 2结果 | `runtime_vs_accuracy_scatter.png` | 横轴=运行时间(s)，纵轴=CEP50(m)，大小=方法名 |

#### **自适应模块可视化（欧洲四城）**

| # | 图表名 | 类型 | 数据源 | 命名规范 | 说明 |
|---|--------|------|--------|---------|------|
| 10 | 算法选择分布 | 饼图 | Module 3结果 | `algorithm_distribution_<dataset>.png` | 显示LS/WLS/FG三种算法的使用比例 |
| 11 | 残差创新时间序列 | 折线图+阴影 | Module 3结果 | `residual_innovation_<dataset>.png` | 横轴=历元索引，纵轴=创新量，正负阴影表示FG优劣 |
| 12 | 创新值直方图 | 直方图 | Module 3结果 | `innovation_distribution_<dataset>.png` | 显示创新值分布，中心在0代表FG和LS平衡 |

#### **跨城市对比可视化**

| # | 图表名 | 类型 | 数据源 | 命名规范 | 说明 |
|---|--------|------|--------|---------|------|
| 13 | 轨迹叠加（四城）| 地图 | 定位结果 | `trajectory_overlay_all_cities.png` | 四城市真值轨迹和各算法估计轨迹叠加，色标表示城市 |
| 14 | NLOS率 vs 定位改善 | 散点图 | 汇总数据 | `nlos_rate_vs_improvement.png` | 横轴=NLOS率(%)，纵轴=改善幅度(%)，含趋势线 |
| 15 | 城市特征对比表 | 表格（PNG） | 数据集统计 | `city_characteristics_table.png` | 表格形式显示4城市的卫星数、NLOS率、轨迹长度等 |

#### **参数扫描结果可视化**

| # | 图表名 | 类型 | 数据源 | 命名规范 | 说明 |
|---|--------|------|--------|---------|------|
| 16 | 单参数扫描折线图 | 折线图（多线） | 参数扫描结果 | `param_sweep_<param_name>.png` | 每条线代表一个城市，横轴参数值，纵轴CEP50 |
| 17 | 双参数扫描热力图 | Heatmap | 参数扫描结果 | `param_heatmap_<param1>_<param2>.png` | 行=param1，列=param2，色阶=CEP50 |
| 18 | 参数敏感性雷达图 | Radar chart | 参数扫描结果 | `sensitivity_radar_<dataset>.png` | 多边形表示各参数变化对CEP50的影响幅度 |

#### **消融与基线实验可视化**

| # | 图表名 | 类型 | 数据源 | 命名规范 | 说明 |
|---|--------|------|--------|---------|------|
| 19 | 误差箱线图（按方法） | Boxplot分面 | Module 2结果 | `error_boxplot_by_city.png` | 4行（城市） × 13列（方法）的小提琴图或箱线图 |

### 4.2 命名规范与组织

#### **文件命名规则**

```
<figure_type>_<dataset_or_param>_<method_or_detail>.png

示例：
  cep50_comparison_all_methods.png        (多城市多方法对比)
  p_los_distribution_berlin1.png          (单城市单模块)
  param_sweep_learning_rate.png           (参数扫描)
  param_heatmap_lr_lambda_bce.png        (双参数热力图)
  confusion_matrix_frankfurt2.png         (单城市单方法)
```

#### **输出目录结构**

```
results/
└── visualizations/
    ├── module1/
    │   ├── confusion_matrix_berlin1.png
    │   ├── p_los_distribution_berlin1.png
    │   └── ...（其他城市）
    ├── module2_localization/
    │   ├── cep50_comparison_all_methods.png
    │   ├── cdf_curves_berlin1.png
    │   └── method_ranking_heatmap.png
    ├── module3_adaptive/
    │   ├── algorithm_distribution_berlin1.png
    │   └── residual_innovation_berlin1.png
    ├── cross_city_analysis/
    │   ├── trajectory_overlay_all_cities.png
    │   └── nlos_rate_vs_improvement.png
    └── param_sweep/
        ├── param_sweep_learning_rate.png
        ├── param_heatmap_lr_lambda.png
        └── sensitivity_radar_berlin1.png
```

### 4.3 可视化生成代码框架

```python
# model/module5_visualization/run.py

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['berlin1', 'berlin2', 'frankfurt1', 'frankfurt2', 'all'])
    parser.add_argument('--module', choices=['1', '2', '3', 'all'])
    parser.add_argument('--output_dir', default='results/visualizations/')
    args = parser.parse_args()
    
    # Module 1 可视化
    if args.module in ['1', 'all']:
        from .module1_viz import visualize_module1
        visualize_module1(args.dataset, args.output_dir)
    
    # Module 2 可视化
    if args.module in ['2', 'all']:
        from .module2_localization_viz import visualize_localization
        visualize_localization(args.dataset, args.output_dir)
    
    # Module 3 可视化
    if args.module in ['3', 'all']:
        from .module3_viz import visualize_adaptive
        visualize_adaptive(args.dataset, args.output_dir)
    
    # 跨模块可视化
    from .cross_module_viz import visualize_cross_city
    visualize_cross_city(args.output_dir)

if __name__ == '__main__':
    main()
```

---

## 五、实验评估框架

### 5.1 评估维度与判断标准

#### **维度 A：实验充分性**

**问题1：当前实验是否足以支撑核心结论？**

评估标准：
- ✓ **A1 Pass**：至少3个数据集、至少5种基线方法、统计显著性检验完整
- ⚠️ **A2 Partial**：数据集 < 3个，或基线方法 < 5种，或缺乏统计检验
- ✗ **A3 Fail**：仅1个数据集，或基线方法 < 3种

**当前状态**（基于result_v2.md）：
- 数据集：4个欧洲城市 ✓
- 基线方法：5个内部基线（已有）
- 新增基线：需实现8个
- 统计检验：需补充

**结论**：目前处于**A2 Partial**，需补充新增基线和统计检验。

---

**问题2：基线选择是否全面？有无关键遗漏？**

评估标准：
- ✓ **B1 Comprehensive**：覆盖经典方法、深度学习、传感器融合等多个类别
- ⚠️ **B2 Partial**：某些类别缺失
- ✗ **B3 Narrow**：仅围绕一种思路（如仅有加权方法）

**当前状态**：
- 经典方法：LS、WLS、FG ✓
- 信号质量：elevation加权（有），C/N0/SNR（无）
- 鲁棒统计：无（RAIM、IRLS需新增）
- 时序方法：无（EKF需新增）
- 深度学习：Module 1是GAT分类，定位是回归（需补充）
- 多传感器：无（INS/GNSS需新增）

**结论**：目前处于**B2 Partial**，缺少信号质量、鲁棒统计、时序、深度学习定位、多传感器等方向的对比。

---

**问题3：消融实验是否充分覆盖各模块贡献？**

评估标准：
- ✓ **C1 Complete**：对Module 1、2、3各有2-3个消融变体（如纯BCE、无MoG等）
- ⚠️ **C2 Partial**：仅对部分模块有消融
- ✗ **C3 Missing**：无系统的消融实验

**当前状态**（基于change_v2.md）：
- Module 1消融：无（尚未进行）
- Module 2消融：隐含在基线对比中（LS vs WLS vs FG）
- Module 3消融：无（参数敏感性实验需做）

**结论**：目前处于**C3 Missing**，需设计系统的消融实验，如：
- Module 1：纯BCE vs 三阶段（已有对比）
- Module 2：with/without MoG加权、with/without多起点
- Module 3：with/without残差跟踪、with/without场景检测

---

**问题4：统计分析是否完备？**

评估标准：
- ✓ **D1 Complete**：有p-value、信心区间、改善幅度、显著性判断
- ⚠️ **D2 Partial**：有平均值和CEP，但缺统计检验
- ✗ **D3 Missing**：仅有单点数据，无聚合统计

**当前状态**（基于result_v2.md）：
- CEP50/CEP95：有 ✓
- 平均误差、标准差：无
- 显著性检验：无
- 误差分布（CDF、箱线图）：无

**结论**：目前处于**D2 Partial**，需补充标准差、误差棒、显著性检验。

---

#### **维度 B：结果质量**

**问题1：CEP50数值与同领域SOTA相比处于什么水平？**

参考值（文献）：
- 城市GNSS定位（无NLOS处理）：800-1200m
- 城市GNSS定位（有NLOS检测）：400-800m
- 高精度多传感器融合：50-200m

**当前系统性能**（基于result_v2.md）：
- Standard LS（无NLOS处理）：904-525m（berlin1-frankfurt1），平均656m
- FG-MoG+2A（有NLOS处理）：984-464m（berlin1-frankfurt1），平均669m（略劣）
- Adaptive-M3：902-467m，平均685m（接近LS）

**评价**：
- ✓ 在可接受范围内（400-1000m）
- ⚠️ 但未如预期显著优于LS（反而略劣）
- ✗ 远不及高端方案（多传感器+滤波可达50-200m）

---

**问题2：自适应选择的提升幅度（+2% ~ +11%）是否具有实际工程意义？**

工程价值评估：
- **+2-3%**：边际改进，学术意义大于工程意义
- **+5-10%**：中等改进，具有工程价值
- **+20%+**：显著改进，生产级应用价值高

**当前改进幅度**（基于result_v2.md）：
- Frankfurt1：FG-MoG+2A vs LS = (525-464)/525 = +11.6% ✓ 中等改进
- 其他城市：改进 < 5% ⚠️ 边际改进

**评价**：
- Frankfurt1有实际意义
- 其他城市的改进有限
- 平均而言，自适应相对于简单LS的增益有限

---

**问题3：HK数据集F1=0.047的失效案例：应视为局限性讨论还是根本性缺陷？**

分析（基于result_v2.md）：
- HK NLOS率仅2.7%，欧洲为25-52%
- 模型在极端类别不平衡（97.3% LOS）下崩溃
- **局限性还是缺陷？**：
  - 学术视角：这是模型的已知局限（不是缺陷），应在论文中明确说明
  - 工程视角：NLOS检测在低NLOS环境下价值有限（缺陷）

**建议**：
- 在论文中明确：本工作适用于 NLOS > 20% 的城市环境
- 作为消融或讨论部分，说明NLOS检测的边界条件

---

#### **维度 C：可复现性与透明度**

**问题1：实验设置是否足够详细以供复现？**

检查清单：
- [ ] 数据集路径、格式、数据划分方式
- [ ] 模型初始化种子、超参数
- [ ] 训练过程（epoch数、学习率调度）
- [ ] 评估指标（CEP定义、ECEF vs ENU）

**当前状态**：
- condition.md中记录了关键参数 ✓
- 但缺少数据划分、种子、评估指标定义的详细说明 ⚠️

---

**问题2：随机种子、数据划分、模型初始化是否固定？**

关键因素：
- PyTorch种子
- NumPy种子
- 数据集划分（时间顺序、随机种子）
- 模型初始化（kaiming、xavier等）

**当前状态**：需确认是否在 common/config_manager.py 中全局设置种子

---

### 5.2 评估输出模板

根据以上评估，给出三种可能的结论：

#### **情况 A：可以投稿**

```markdown
# 评估结论：推荐投稿

## 总体评价
当前实验充分且结果质量达到学术发表标准。

## 优势
1. **实验设计完整**：4城市 × 13方法 × 参数扫描，覆盖面广
2. **方法创新性**：自适应选择机制在Frankfurt1达到+11.6%改进
3. **代码规范化**：模块化结构、统一接口、完整文档
4. **结论明确**：NLOS检测在高NLOS环境有效，但在低NLOS环境失效

## 局限性与改进建议
1. **论文中明确说明**：适用条件为NLOS > 20%的城市环境
2. **补充实验**（P2级别）：
   - 新增8个外部基线的对比（运行时间±2周）
   - Wilcoxon显著性检验（运行时间±3天）
   - 参数敏感性分析（运行时间±1周）

## 目标期刊建议
- **顶级期刊**（GPS Solutions, IEEE GNSS）：需补充新增基线和显著性检验
- **中档期刊**（Sensors, Remote Sensing）：当前实验足以支撑投稿

## 核心贡献点
1. PI-PEM三模块系统的系统化实现与对比
2. NLOS意识的自适应定位选择机制
3. 完整的代码重构与模块化设计
```

#### **情况 B：需要补充实验**

```markdown
# 评估结论：需要补充实验后投稿

## 缺口分析
| 维度 | 当前状态 | 不足之处 | 优先级 |
|------|--------|--------|--------|
| 基线方法 | 5个内部 | 缺8个外部基线 | P0 |
| 消融实验 | 无系统消融 | 需3-5个消融变体 | P1 |
| 统计检验 | 无 | 需Wilcoxon + CI | P1 |
| 参数敏感性 | 无 | 需双参数热力图 | P2 |

## 补充实验清单

### P0优先级（必补）
**补充外部基线**（实现8个基线方法）
- 预期工时：2-3周
- 预期产出：基线对比表、排名图表
- 价值：验证Module 1的竞争力，支撑创新性声称

**添加统计检验**（Wilcoxon符号秩检验）
- 预期工时：3-5天
- 预期产出：p-value表、显著性标记
- 价值：提升论文严谨性

### P1优先级（强烈建议）
**系统消融实验**
- Module 1：三阶段 vs 两阶段 vs 单阶段
- Module 2：有MoG vs 无MoG
- Module 3：有/无残差跟踪，有/无场景检测
- 预期工时：1-2周
- 价值：明确各模块贡献度

### P2优先级（可选）
**参数敏感性分析**
- 双参数热力图
- 预期工时：1周

## 完成后的投稿路径
P0 + P1 完成 → 可投顶级期刊  
P0 完成 → 可投中档期刊  
P0 + P1 + P2 完成 → 可投顶级会议（IEEE ICCAS, ION）
```

#### **情况 C：需要重新设计框架**

```markdown
# 评估结论：框架需重新设计

## 核心瓶颈
1. **NLOS检测有效性有限**：欧洲4城NLOS > 25%时，FG方法相对LS改进<5%
2. **自适应选择逻辑不够鲁棒**：跨数据集参数泛化性差
3. **定位精度未达SOTA**：656m vs 400m（同类SOTA方案）

## 问题根源分析
- [ ] Module 1：MoG输出的mu和sigma没有充分利用（仅在WLS中用p_los）
- [ ] Module 2：因子图优化不稳定（多起点仍无法保证最优解）
- [ ] Module 3：场景质量检测指标选择不当（DOP膨胀与NLOS无强相关）

## 重新设计建议
### 方向 1：增强Module 1输出
- 目前：仅输出p_los作为二分类
- 改进：同时输出mu、sigma、异常度评分（三维输出）
- 预期效果：为Module 2提供更丰富的先验

### 方向 2：替换Module 2为学习的定位
- 目前：纯解析方法（LS/WLS/FG）
- 改进：用GAT端到端学习定位（见baseline 12）
- 预期：消除复杂的多起点优化，直接学习鲁棒的定位函数

### 方向 3：重新设计Module 3的自适应逻辑
- 目前：基于残差创新和场景质量评分
- 改进：基于历元级的不确定性预测（从Module 1学习）
- 预期：更稳健的算法选择

## 重新设计的预期时间表
- 设计与原型：2周
- 实现与集成：3周
- 实验与验证：4周
- 论文撰写：2周
- **总计：3月**
```

---

## 六、执行计划与里程碑

### 6.1 分阶段工作计划

#### **阶段 1：**

**目标**：完成模块化重构，建立标准接口

| 任务 | 工作量 | 产出 | 验收标准 |
|------|--------|------|---------|
| Module 1重构 | 3天 | 规范化trainer、数据加载、推理接口 | 能独立运行、TensorBoard输出正确 |
| Module 2重构 + 基类 | 3天 | LocalizationBase、工厂模式 | 5个基线都能正确注册和调用 |
| Module 3重构 | 2天 | 独立run.py，配置分离 | 配置改变能正确反映在结果中 |
| Module 4框架搭建 | 3天 | param_sweep、batch_scheduler骨架 | 支持CLI调用、结果自动聚合 |
| Module 5框架搭建 | 2天 | 可视化入口、目录结构 | 能生成示意图 |
| 公共库整理 | 2天 | common/目录、坐标变换、指标计算 | 各模块能正确import |
| **小计** | **15天** | | |

#### **阶段 2：基线方法实现**

**目标**：实现8个新基线方法

| 基线方法 | 工作量 | 验收标准 |
|---------|--------|---------|
| CNO加权 + SNR加权 | 1天 | 运行成功，CEP50合理 |
| RAIM + IRLS | 2天 | 迭代正常收敛 |
| EKF | 2天 | IMU预积分正确，融合逻辑清晰 |
| DNN + GAT端到端 | 2天 | 模型能加载训练，输出位置维度正确 |
| INS/GNSS | 1天 | 松耦合和紧耦合都能运行 |
| **小计** | **8天** | |

#### **阶段 3：对比实验执行**

**目标**：运行P0和P1优先级实验

| 实验类型 | 运行时间 | 并行性  |
|---------|--------|--------|
| Phase 1：基线对比（4数据集 × 13方法） | 2h/数据集 | 4并行 |
| Phase 2：参数P0扫描 | 2h/数据集 | 4并行 |
| Phase 3：敏感性分析（双参数） | 4h | 1串行 |
| Phase 4：统计检验 | - | - | 0.5h |
| **总计** | | | **~8.5h** |

运行策略：
- 启动基线对比（自动化，无人值守）
- 参数扫描（分批运行）
- 统计检验 + 结果汇总



#### **阶段 4：可视化与报告**

**目标**：生成19张图表和综合报告

| 任务  | 产出 |
|------|--------|
| 批量生成可视化 | 19张规范命名的PNG |
| 生成对比表格 | 基线排名、参数热力图等 |
| 撰写实验报告 | markdown格式，包含所有图表和结论 |
| 评估与反思 | 结论是否投稿、需补充什么 |

### 6.2 风险与应对

| 风险 | 影响 | 应对方案 |
|------|------|---------|
| 基线实现bug | 高 | 逐个单元测试，对标论文结果 |
| 参数扫描耗时过长 | 中 | 分阶段运行，优先运行P0 |
| 新基线性能异常 | 中 | 与简单LS对标，确保输出合理性 |
| 模块间数据格式不兼容 | 高 | 提前定义统一接口，单元测试 |

---

## 七、附录

### 7.1 配置文件模板

#### **module1_nlos/config.yaml**

```yaml
# Module 1: NLOS感知与误差建模 配置文件

dataset:
  name: berlin1
  path: data/berlin1/
  train_val_split: 0.7  # 70% 训练，30% 验证
  
model:
  architecture: gat_mog  # 图神经网络 + 混合高斯输出
  in_features: 11        # 卫星特征维数
  hidden_features: 128   # 隐藏层维数
  num_layers: 2          # GAT层数
  num_heads: 8           # 多头数
  dropout: 0.1
  
training:
  num_epochs: 100
  batch_size: 32
  use_block_diagonal_batching: true  # 处理可变卫星数
  learning_rate: 5.0e-6
  weight_decay: 1.0e-4
  gradient_clip: 10.0    # 梯度裁剪
  optimizer: adam        # 或 adamw
  
loss:
  # 三阶段训练配置
  pure_bce_epochs: 8          # 阶段1：纯BCE
  blend_epochs: 25            # 阶段2：BCE→NLL平滑过渡
  # 阶段3自动为 100 - 8 - 25 = 67个epoch
  
  # 损失权重
  lambda_bce: 0.6
  lambda_direction: 1.0       # 方向约束（sigma_los > sigma_nlos）
  lambda_sigma_sep: 5.0       # sigma分离约束
  
  # MoG参数
  mu_nlos_target: 0.30        # mu_nlos目标值（km）
  lambda_mu_reg: 0.20         # mu回归损失权重
  
early_stopping:
  patience: 15                # 验证F1连续15个epoch无改进则停止
  delta: 0.001                # 改进阈值（0.1%）
  
data:
  normalize_features: true
  feature_range: [0, 1]       # 特征归一化范围
  
logging:
  use_tensorboard: true
  tensorboard_dir: runs/
  log_frequency: 10           # 每10个epoch记录一次
  
device: cuda:0
```

#### **module2_localization/config.yaml**

```yaml
# Module 2: 融合定位 配置文件

methods:
  - standard_ls
  - wls_elevation
  - wls_mog
  - hard_threshold
  - fg_mog
  - cno_weighted
  - snr_weighted
  - raim
  - irls
  - ekf
  - dnn_e2e
  - gat_e2e
  - ins_gnss

# 各方法的特定参数
standard_ls: {}

wls:
  platt_calibration: true     # 是否进行Platt概率校准

hard_threshold:
  threshold: 0.5              # p_los > 0.5 认为是LOS

factor_graph:
  multistart: 3               # 多起点优化的起点数
  max_iterations: 10
  convergence_threshold: 0.01
  optimization_method: l_bfgs_b

raim:
  detection_threshold: 3.0    # 异常卫星检测阈值（标准差）

irls:
  max_iterations: 10
  robustness_param: 1.5       # Huber/M估计器的鲁棒性参数
  convergence_threshold: 0.01

ekf:
  process_noise: 0.1          # 过程噪声强度
  measurement_noise: 1.0      # 观测噪声强度
  initial_state: [6378000, -2200000, 2400000, 0]  # 初始位置和钟偏差

evaluation:
  metrics:
    - cep50
    - cep95
    - mean_error
    - rmse
    - median_error
```

#### **module3_adaptive/config.yaml**

```yaml
# Module 3: 自适应选择 配置文件

adaptive_selector:
  window_size: 50             # 滑动窗口大小（历元数）
  min_history: 15             # 最少历史样本数
  
thresholds:
  fg_threshold: 0.65          # 使用FG的综合评分阈值
  wls_threshold: 0.50         # 使用WLS的综合评分阈值
  # 评分 >= fg_threshold → 选FG
  # fg_threshold > 评分 >= wls_threshold → 选WLS
  # 评分 < wls_threshold → 选LS

scene_quality_detection:
  # 三个关键指标的权重（和为1）
  weight_plos_gap: 0.33       # p_los LOS/NLOS分离度
  weight_pdop_ratio: 0.33     # DOP比率（加权vs均匀）
  weight_nlos_redundancy: 0.34 # NLOS卫星冗余度
  
  # 各指标的阈值
  plos_gap_threshold: 0.4     # p_los_gap > 0.4认为质量HIGH
  pdop_ratio_threshold: 1.1   # 加权DOP/均匀DOP < 1.1
  nlos_redundancy_threshold: 3 # 有效NLOS卫星数 >= 3

fallback:
  enabled: true
  fallback_ratio: 1.05        # 如果选择方法相对LS误差>5%则回退
  
residual_tracking:
  enabled: true               # 是否使用残差创新跟踪
  innovation_alpha: 0.5       # 创新值指数平滑系数
```

### 7.2 运行命令示例

```bash
# 1. 完整流程（从数据预处理到评估）
python -m model.module1_nlos.run \
    --config model/module1_nlos/config.yaml \
    --dataset berlin1 \
    --output results/complete_run_berlin1/

# 2. 仅运行基线对比
python -m model.module4_experiments.run \
    --dataset berlin1,berlin2,frankfurt1,frankfurt2 \
    --mode baseline_comparison \
    --output results/baseline_comparison/

# 3. 单参数扫描（fg_threshold）
python -m model.module4_experiments.param_sweep \
    --dataset berlin1 \
    --param fg_threshold \
    --values 0.50 0.55 0.60 0.65 0.70 0.75 0.80 \
    --output results/sweep_fg_threshold/

# 4. 双参数热力图
python -m model.module4_experiments.param_sweep \
    --dataset berlin1 \
    --param1 learning_rate --values1 1e-6 2e-6 5e-6 1e-5 \
    --param2 lambda_bce --values2 0.2 0.4 0.6 0.8 1.0 \
    --output results/heatmap_lr_lambda/

# 5. 统计检验
python -m model.module4_experiments.run \
    --mode statistical_test \
    --input_dir results/baseline_comparison/ \
    --output results/statistical_test_report.json

# 6. 可视化生成
python -m model.module5_visualization.run \
    --dataset all \
    --module all \
    --output_dir results/visualizations/

# 7. 生成综合报告
python -m model.module4_experiments.run \
    --mode generate_report \
    --input_dir results/experiments/ \
    --output results/FINAL_REPORT.md
```

### 7.3 PowerShell批处理脚本示例

```powershell
# scripts/quick_experiment.ps1
# 快速运行一个完整实验周期（用于测试）

param(
    [string]$Dataset = "berlin1",
    [string]$OutputDir = "results/quick_test"
)

$Python = "python"
$StartTime = Get-Date

Write-Host "Quick Experiment: $Dataset" -ForegroundColor Green
Write-Host "Output: $OutputDir" -ForegroundColor Green
Write-Host "Started: $StartTime`n" -ForegroundColor Green

# Step 1: 基线对比（1条命令）
Write-Host "[1/4] Running baseline comparison..." -ForegroundColor Cyan
& $Python -m model.module4_experiments.run `
    --dataset $Dataset `
    --mode baseline_comparison `
    --output $OutputDir/baseline

# Step 2: 关键参数扫描（3条命令，并行运行）
Write-Host "[2/4] Running parameter sweeps..." -ForegroundColor Cyan
$jobs = @()
$jobs += Start-Job -ScriptBlock { & python -m model.module4_experiments.param_sweep `
    --dataset $args[0] --param fg_threshold --values 0.50 0.65 0.80 `
    --output $args[1]/sweep_fg_threshold } -ArgumentList $Dataset, $OutputDir
$jobs += Start-Job -ScriptBlock { & python -m model.module4_experiments.param_sweep `
    --dataset $args[0] --param window_size --values 20 50 100 200 `
    --output $args[1]/sweep_window_size } -ArgumentList $Dataset, $OutputDir

Wait-Job -Job $jobs | Receive-Job

# Step 3: 可视化
Write-Host "[3/4] Generating visualizations..." -ForegroundColor Cyan
& $Python -m model.module5_visualization.run `
    --dataset $Dataset `
    --module all `
    --output_dir $OutputDir/visualizations

# Step 4: 报告
Write-Host "[4/4] Generating report..." -ForegroundColor Cyan
& $Python -m model.module4_experiments.run `
    --mode generate_report `
    --input_dir $OutputDir `
    --output $OutputDir/report.md

$EndTime = Get-Date
$Duration = $EndTime - $StartTime
Write-Host "`nCompleted in $($Duration.TotalMinutes)m" -ForegroundColor Green
```

---

## 八、核心设计原则总结

为了确保重构质量，遵循以下核心设计原则：

1. **模块化与解耦**
   - 各模块独立可运行，通过清晰的数据接口（pickle/JSON/HDF5）交互
   - 模块间无硬编码依赖，通过工厂模式和配置文件管理

2. **接口优先**
   - 所有定位方法继承 LocalizationBase 统一接口
   - 统一的输入输出格式确保互换性

3. **可重复性与可扩展性**
   - 所有参数通过配置文件管理，支持批量修改
   - 完整的日志记录（condition.md），支持精确复现

4. **用户友好性**
   - 每个模块都有独立的 run.py 和详细的 README.md
   - 支持命令行接口，支持单个模块独立运行

5. **质量保证**
   - 单元测试覆盖核心逻辑（>80%）
   - 集成测试覆盖端到端流程
   - 统计显著性检验支持

---

**文档完成**。本Goal文档为代码重构和对比实验的完整指南，包含了详细的设计方案、实现规范、执行计划和评估框架。按照此文档进行实施，可确保重构质量和实验的充分性与可复现性。