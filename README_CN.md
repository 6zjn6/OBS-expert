# OBS-Expert: 基于大语言模型与知识图谱的甲骨文字解读框架

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT">
</p>

<p align="center">
  <a href="README_CN.md"><img src="https://img.shields.io/badge/lang-中文-red.svg" alt="中文"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/lang-English-blue.svg" alt="English"></a>
</p>
<p align="center">
  <em>一个自动化甲骨文字分析的研究框架，融合视觉模型、Neo4j 知识图谱与多模型 LLM 管线，生成古汉字的学术性释义。</em>
</p>


---

## 概述

甲骨文是目前已知最早的成体系汉字，刻写于商代（约公元前 1200 年）的龟甲和兽骨之上。解读甲骨文需要深厚的古文字学功底。本项目探究大语言模型在视觉部首识别和结构化知识增强下，能否生成准确的甲骨文字释义，并系统对比不同增强策略的效果。

### 系统架构

```
OB_Radix（源数据集：CSV + 图像）
    |  sync_data.py
    v
experiment/data/（工作副本）
    |
    |---> PrototypeClassifier -- DinoV2 特征提取 -- 部首预测
    |---> KG_construct --------- Neo4j 图谱（部首 <-> 字符）
    |---> chatgpt_rag ---------- RAG 管线 + 工具调用代理
    |---> cache_manager -------- LRU + 磁盘缓存
    |
    v
experiment/exp*/run*.py（管线执行）
    |
    v
output/（结果 CSV）
```

## 数据集：OB_Radix

由于体积限制，仓库仅包含少量示例文件（位于 `OB_Radix/`）。完整数据集包含：

| 文件 | 说明 |
|------|------|
| `character_explanations_CN.csv` | 字符 -> 学术释义（中文） |
| `character_explanations.csv` | 字符 -> 学术释义 |
| `character_analysis.csv` | 字符 -> 类型（象形字/会意字/形声字）+ 分析理据 |
| `radical_explanation.csv` | 部首 -> 定义 + 关联字符 |

**图像目录：**

- `img_zi/{字符}/` — 甲骨字图像（`.jpg`）及提取的部首切片（`.png`）
- `organized_radicals/{部首}/` — 部首原型样本，用于 PrototypeClassifier 训练

## 实验

### 实验一：部首识别

**研究问题：** 基于视觉的原型匹配能否有效识别甲骨文部首？

使用 **DinoV2** 主干网络提取 768 维特征向量，通过余弦相似度与学习到的原型进行匹配分类。

- **输入：** 部首切片图像
- **输出：** Top-k 预测部首类别及相似度分数

```bash
cd experiment/exp1
python run1.py
```

### 实验二：字形分类

**研究问题：** 知识图谱增强是否能提升字形分类准确率？

将甲骨字分为三种类型：

| 类型 | 说明 |
|------|------|
| 象形字 | 以图形描摹事物形象 |
| 会意字 | 组合多个语义部件表达含义 |
| 形声字 | 一个部件表义，一个部件表音 |

对比两条管线：

| 管线 | 脚本 | 说明 |
|------|------|------|
| 基线 | `run2_baseline.py` | 图像 -> LLM -> 类型预测 |
| 生成模块 | `run2_generation_module.py` | 图像 -> PrototypeClassifier -> 知识图谱检索 -> LLM -> 类型预测 |

```bash
cd experiment/exp2
python run2_baseline.py
python run2_generation_module.py
```

### 实验三：释义生成（主实验）

**研究问题：** LLM 能否生成准确的甲骨文释义？基线、知识图谱增强和多智能体方案的效果如何？

数据集按 7:3 划分：已见部分用于构建知识图谱，未见部分用于测试生成质量。

| 管线 | 脚本 | 说明 |
|------|------|------|
| 基线 | `run_baseline.py` | 纯 LLM 生成（无外部知识） |
| 原型 + 知识图谱 | `run_prototype_kg.py` | DinoV2 部首预测 -> 知识图谱检索 -> 增强 LLM 提示 |
| 多智能体 | `multi_agent_run.py` | 图像分析智能体（工具 + KG）-> 思考智能体（深度推理） |

**多智能体架构：**

```
甲骨字图像 + 部首图像
    |  [图像分析智能体 + KG 工具]
    v
部首预测 + 知识图谱检索结果
    |  [思考智能体，推理模式]
    v
5 条候选释义
```

```bash
cd experiment/exp3
python run_baseline.py
python run_prototype_kg.py
python multi_agent_run.py
```

每条管线为每个甲骨字生成 5 条候选释义。

### 补充实验

| 实验 | 脚本 | 说明 |
|------|------|------|
| 异体字分析 | `analyze_variants.py` | 查找同时出现在已见/未见集中的异体字 |
| 异体字处理 | `var_run.py` | 甲骨字 -> 现代汉字映射 |
| 英文版 | `exp3_English_version/` | 以英文提示词和输出复现实验三 |

英文版在 `output/` 下包含四个模型的预计算结果（按 `model_1/` 至 `model_4/` 组织）。

## 快速开始

### 环境要求

- **Python** >= 3.10
- **Neo4j** 数据库运行于 `localhost:7687`
- 推荐使用 **GPU** 进行 DinoV2 推理（支持 CPU 回退）

### 安装

```bash
git clone https://github.com/<your-org>/OBS_expert.git
cd OBS_expert
pip install -r requirements.txt
```

### 运行

```bash
# 1. 同步数据集到实验目录
python tools/sync_data.py --src OB_Radix --targets experiment/data --with-assets

# 2. 配置 LLM API
export LLM_API_KEY="your-api-key"
export LLM_MODEL="your-model-name"

# 3. 确保 Neo4j 已启动，然后运行实验
cd experiment/exp3
python run_prototype_kg.py
```

### LLM 配置

所有 LLM 调用均通过环境变量配置。任何兼容 OpenAI API 的大语言模型均可使用。

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_MODEL` | 模型标识 | -- |
| `LLM_BASE_URL` | API 端点（OpenAI 兼容） | `https://api.openai.com/v1` |
| `LLM_API_KEY` | API 密钥 | -- |
| `LLM_TEMPERATURE` | 采样温度 | `0.7` |
| `LLM_MAX_TOKENS` | 最大输出 token 数 | `4096` |
| `LLM_ENABLE_THINKING` | 启用推理模式（需模型支持） | `false` |
| `LLM_THINKING_MODELS` | 支持 thinking 模式的模型名称列表（逗号分隔） | -- |

## 项目结构

```
OBS_expert/
├── OB_Radix/                          # 源数据集
│   ├── img_zi/                        #   甲骨字图像（按字分目录）
│   └── organized_radicals/            #   部首原型图像（按部首分目录）
├── experiment/
│   ├── common/                        # 公共模块
│   │   ├── config.py                  #   模型加载与图像处理
│   │   ├── chatgpt.py                 #   LLM 视觉+文本 API 集成
│   │   ├── chatgpt_rag.py             #   RAG 管线与工具调用代理
│   │   ├── KG_construct.py            #   Neo4j 知识图谱构建
│   │   ├── PrototypeClassifier.py     #   基于 DinoV2 的部首分类
│   │   ├── cache_manager.py           #   LRU + 磁盘缓存
│   │   └── robust_csv_reader.py       #   鲁棒 CSV 解析
│   ├── exp1/                          # 实验一：部首识别
│   │   └── run1.py
│   ├── exp2/                          # 实验二：字形分类
│   │   ├── run2_baseline.py
│   │   └── run2_generation_module.py
│   ├── exp3/                          # 实验三：释义生成
│   │   ├── run_baseline.py
│   │   ├── run_prototype_kg.py
│   │   └── multi_agent_run.py
│   ├── supplementary/                 # 补充实验
│   │   ├── analyze_variants.py
│   │   ├── var_run.py
│   │   └── exp3_English_version/      #   英文版复现
│   └── data/                          # 同步的工作数据（自动生成）
├── tools/
│   └── sync_data.py                   # OB_Radix -> experiment/data 同步
├── requirements.txt
├── README.md                          # English
└── README_CN.md                       # 中文
```

## 版本控制建议

- 将 `OB_Radix/` 和 `experiment/common/` 纳入版本控制
- `experiment/data/` 由 `sync_data.py` 生成，默认忽略
- 实验目录下的 `output*/` 目录默认忽略

## 引用

如果本工作对您有帮助，请引用：

```bibtex
@article{obs_expert2025,
  title={OBS-Expert: Oracle Bone Script Interpretation with LLMs and Knowledge Graphs},
  author={},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 许可证

本项目采用 MIT 许可证。
