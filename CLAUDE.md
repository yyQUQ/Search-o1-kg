# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Search-o1-KG 是基于 Search-o1 框架构建的动态知识图谱增强推理框架。它集成了三个核心创新：
1. **动态知识图谱构建** - 从检索文档中实时构建知识图谱
2. **图神经网络推理** - 在知识图谱上进行路径推理和多跳推理
3. **跨模态对齐** - 文本-图谱对齐，提供可可视化的推理轨迹

该系统通过自动从搜索结果中抽取实体/关系、构建知识图谱，并执行基于图的推理，为大型语言模型提供更准确和可解释的答案。

## 常用开发命令

### 环境配置
```bash
# 安装依赖
pip install -r requirements.txt

# 下载 spaCy 模型
python -m spacy download en_core_web_sm

# 设置环境（安装依赖、创建配置）
python scripts/setup_environment.py --install

# 快速测试所有组件
python scripts/quick_start.py --test all
```

### 运行实验
```bash
# 主推理脚本（最重要的文件）
python scripts/run_search_o1_kg.py \
    --dataset_name gpqa \
    --split diamond \
    --model_path "microsoft/DialoGPT-medium" \
    --bing_subscription_key "$BING_SUBSCRIPTION_KEY" \
    --subset_num 10

# 使用 Jina API（可选）
python scripts/run_search_o1_kg.py \
    --dataset_name math500 \
    --split test \
    --model_path "your_model_path" \
    --bing_subscription_key "$BING_SUBSCRIPTION_KEY" \
    --jina_api_key "$JINA_API_KEY" \
    --max_search_limit 5 \
    --max_turn 10
```

### 数据管理
```bash
# 下载数据集
python scripts/download_datasets.py --dataset gpqa
python scripts/download_datasets.py --dataset math500

# 创建测试样本数据
python scripts/quick_start.py --create-samples

# 预处理和标准化数据
python scripts/preprocess_data.py --check-format
```

### 测试和评估
```bash
# 测试单个组件
python tests/test_kg.py
python tests/test_gnn.py
python tests/test_alignment.py

# 评估知识图谱性能
python scripts/evaluate_kg.py \
    --input outputs/gpqa.model.search_o1_kg/test.output.json \
    --data data/GPQA/diamond.json
```

## 高层架构设计

### 核心模块

1. **知识图谱构建** (`src/knowledge_graph/`)
   - `EntityExtractor`: 多模态实体抽取（文本、数学、化学）
   - `RelationExtractor`: 实体间关系抽取
   - `KnowledgeGraphBuilder`: 支持增量更新的图谱构建
   - `GraphStorage`: 多种存储格式（JSON、NetworkX、Pickle）

2. **图神经网络推理** (`src/gnn_reasoning/`)
   - `GraphReasoningEngine`: 核心推理引擎，包含路径查找
   - `GraphEmbeddingManager`: GNN 实现（GCN、GAT、GraphSAGE）
   - `EntityLinker`: 将问题实体链接到知识图谱
   - `PathFinder`: 多跳推理路径发现

3. **跨模态对齐** (`src/multimodal_alignment/`)
   - `MultimodalAligner`: 文本-图谱对齐，支持多种策略
   - `VisualizationEngine`: 推理轨迹可视化
   - `AlignmentUtils`: 对齐质量评估工具

### 关键集成点

主脚本 `scripts/run_search_o1_kg.py` 集成了：
- 原始 Search-o1 搜索和文档推理流水线
- 从检索文档动态构建知识图谱
- 在构建的知识图谱上进行基于 GNN 的推理
- 跨模态对齐以提供可解释的输出

### 数据流

1. **搜索阶段**: Bing/Web 搜索检索相关文档
2. **KG 构建**: 抽取实体/关系 → 构建知识图谱
3. **图推理**: 实体链接 → 路径查找 → GNN 推理
4. **对齐**: 将推理步骤映射到图谱节点以实现可解释性
5. **生成**: 产生带有支持证据的最终答案

## API 配置

### 必需的环境变量
```bash
BING_SUBSCRIPTION_KEY=your_bing_search_api_key
JINA_API_KEY=your_jina_api_key  # 可选但推荐
```

### 模型配置
- 默认模型: `microsoft/DialoGPT-medium`（用于测试）
- 生产模型: 任何 Hugging Face 聊天/指令模型
- 系统使用 vLLM 进行高效推理

## 重要实现细节

### 知识图谱集成
- 为每个问题从搜索结果动态构建 KG
- 支持多模态实体（文本、数学公式、化学结构）
- 使用增量更新避免从头重建
- 缓存在 `cache/kg_cache/` 中避免重复计算

### GNN 推理
- 实现多种 GNN 架构（GCN、GAT、GraphSAGE）
- 可配置的隐藏维度和层数
- 支持直推式和归纳式推理
- 基于路径的多跳问题推理

### 跨模态对齐
- 三种对齐策略：精确匹配、部分匹配、语义匹配
- 为对齐质量提供置信度分数
- 支持可视化的推理链
- 将文本推理步骤映射到图谱实体/关系

## 数据要求

所有数据集必须标准化为以下格式：
```json
[
  {
    "Question": "问题文本",
    "Answer": "答案",  // 问答任务
    "Correct Choice": "A"  // 多选题
  }
]
```

支持的数据集：GPQA、MATH500、HotpotQA、NQ、TriviaQA、AIME、AMC、LiveCodeBench

## 输出结构

结果保存在 `outputs/dataset.model.search_o1_kg/` 中：
- `test.info_extract.json`: 包含 KG 增强信息的详细推理记录
- `test.output.json`: 模型原始输出
- `test.metrics.json`: 性能指标

## 开发技巧

- 使用 `--subset_num 1` 进行快速调试
- 如果搜索结果看起来过时，检查缓存文件
- 使用 `watch -n 1 nvidia-smi` 监控 GPU 内存
- 如果内存受限，使用较小的 GNN 维度（`--gnn_hidden_dim 64`）
- 使用 `export PYTHONPATH=/path/to/src:$PYTHONPATH` 启用详细日志