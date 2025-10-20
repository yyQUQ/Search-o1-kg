# Search-o1-KG 文件结构详解

## 📁 完整目录结构

```
search-o1-kg/
├── 📄 README.md                   # 项目主文档
├── 📄 USAGE_GUIDE.md             # 详细使用指南
├── 📄 PROJECT_SUMMARY.md         # 项目总结报告
├── 📄 FILE_STRUCTURE.md          # 文件结构说明（本文档）
├── 📄 requirements.txt           # Python依赖包列表
├── 📄 LICENSE                    # MIT开源许可证
├── 📄 .env                       # 环境变量配置文件（需要创建）
├── 📄 .gitignore                 # Git忽略文件配置
│
├── 📁 src/                       # 核心源代码模块
│   ├── 📄 __init__.py            # 包初始化文件
│   │
│   ├── 📁 knowledge_graph/       # 🔬 知识图谱构建模块
│   │   ├── 📄 __init__.py        # 导出知识图谱相关类
│   │   ├── 📄 entity_extractor.py # 🏗️ 实体抽取器（核心文件）
│   │   ├── 📄 relation_extractor.py # 🔗 关系抽取器
│   │   ├── 📄 graph_builder.py   # 🏛️ 知识图谱构建器
│   │   └── 📄 graph_storage.py   # 💾 图谱存储系统
│   │
│   ├── 📁 gnn_reasoning/         # 🧠 图神经网络推理模块
│   │   ├── 📄 __init__.py        # 导出推理相关类
│   │   ├── 📄 graph_neural_network.py # 🔮 GNN实现（GCN/GAT/GraphSAGE）
│   │   ├── 📄 reasoning_engine.py # ⚙️ 推理引擎（核心文件）
│   │   ├── 📄 path_finder.py      # 🔍 路径查找器
│   │   └── 📄 entity_linker.py    # 🔗 实体链接器
│   │
│   └── 📁 multimodal_alignment/  # 🎨 跨模态对齐模块
│       ├── 📄 __init__.py        # 导出对齐相关类
│       ├── 📄 multimodal_aligner.py # 🔄 对齐器（核心文件）
│       ├── 📄 visualization_engine.py # 📊 可视化引擎
│       └── 📄 alignment_utils.py  # 🛠️ 对齐工具
│
├── 📁 scripts/                   # 🚀 脚本目录
│   ├── 📄 run_search_o1_kg.py    # 🎯 主推理脚本（最重要文件）
│   ├── 📄 download_datasets.py  # 📥 数据集下载脚本
│   ├── 📄 setup_environment.py   # 🔧 环境配置脚本
│   ├── 📄 quick_start.py         # ⚡ 快速启动演示
│   ├── 📄 preprocess_data.py     # 🔄 数据预处理脚本
│   ├── 📄 evaluate_kg.py         # 📊 知识图谱评估脚本
│   └── 📄 visualize_reasoning.py # 📈 推理可视化脚本
│
├── 📁 data/                      # 📊 数据目录
│   ├── 📁 GPQA/                  # 🧪 GPQA科学问答数据集
│   │   ├── 📄 diamond.json       # 钻石级数据
│   │   └── 📄 diamond_sample.json # 样本数据
│   │
│   ├── 📁 MATH500/               # 🔢 MATH500数学推理数据集
│   │   ├── 📄 test.json          # 测试数据
│   │   └── 📄 test_sample.json   # 样本数据
│   │
│   ├── 📁 QA_Datasets/           # ❓ 问答数据集
│   │   ├── 📄 nq.json            # Natural Questions
│   │   ├── 📄 triviaqa.json      # TriviaQA
│   │   ├── 📄 hotpotqa.json      # HotpotQA
│   │   └── 📄 *_sample.json      # 各数据集样本
│   │
│   └── 📁 LIVECODEBENCH/         # 💻 LiveCodeBench代码数据集
│       └── 📄 test.json          # 测试数据
│
├── 📁 cache/                     # 💾 缓存目录
│   ├── 📄 search_cache.json      # 搜索结果缓存
│   ├── 📄 url_cache.json         # URL内容缓存
│   └── 📁 kg_cache/              # 知识图谱缓存
│       ├── 📄 kg_*.json          # 知识图谱文件
│       └── 📄 embeddings_*.pkl   # 图嵌入文件
│
├── 📁 outputs/                   # 📤 输出目录
│   ├── 📁 runs.baselines/        # 🏁 基线模型结果
│   ├── 📁 runs.qa/               # ❓ 问答任务结果
│   ├── 📁 runs.analysis/         # 🔍 分析结果
│   └── 📁 dataset.model.search_o1_kg/ # 📈 具体实验结果
│       ├── 📄 test.info_extract.json # 详细推理记录
│       ├── 📄 test.output.json   # 模型原始输出
│       └── 📄 test.metrics.json   # 性能指标
│
├── 📁 configs/                   # ⚙️ 配置文件目录
│   ├── 📄 config.json            # 主配置文件
│   ├── 📄 gnn_config.json        # GNN配置
│   └── 📄 dataset_configs/       # 数据集配置
│       ├── 📄 gpqa.json         # GPQA配置
│       └── 📄 math500.json      # MATH500配置
│
├── 📁 tests/                     # 🧪 测试目录
│   ├── 📄 test_kg.py            # 知识图谱测试
│   ├── 📄 test_gnn.py           # GNN测试
│   ├── 📄 test_alignment.py     # 对齐测试
│   └── 📄 test_integration.py   # 集成测试
│
├── 📁 logs/                      # 📝 日志目录
│   ├── 📄 setup_report.json      # 环境配置报告
│   ├── 📄 experiment_*.log      # 实验日志
│   └── 📄 error_*.log           # 错误日志
│
└── 📁 docs/                      # 📚 文档目录
    ├── 📄 api_reference.md      # API参考文档
    ├── 📄 examples.md           # 使用示例
    └── 📄 theory.md             # 理论背景
```

## 🎯 核心文件详解

### 1. 主要运行文件

#### `scripts/run_search_o1_kg.py` - 🎯 **最重要文件**
- **功能**: 主推理脚本，集成所有模块
- **作用**:
  - 加载和配置模型
  - 执行搜索增强推理
  - 集成知识图谱推理
  - 生成最终答案
- **使用**: `python scripts/run_search_o1_kg.py --dataset_name gpqa --model_path "model_path" --bing_subscription_key "key"`

#### `src/knowledge_graph/entity_extractor.py` - 🏗️ **核心组件**
- **功能**: 从文本中自动抽取实体
- **特点**:
  - 支持多模态实体（文本、数学、化学）
  - 基于预训练NER模型和规则模式
  - 实体消歧和置信度计算
- **主要类**: `EntityExtractor`

#### `src/gnn_reasoning/reasoning_engine.py` - ⚙️ **推理核心**
- **功能**: 基于知识图谱的推理引擎
- **特点**:
  - 实体链接和关系预测
  - 多跳推理和路径查找
  - 推理步骤生成
- **主要类**: `GraphReasoningEngine`

#### `src/multimodal_alignment/multimodal_aligner.py` - 🔄 **对齐核心**
- **功能**: 文本推理步骤与图谱节点对齐
- **特点**:
  - 多种对齐策略
  - 可视化溯源
  - 质量评估
- **主要类**: `MultimodalAligner`

### 2. 配置文件

#### `.env` - 环境变量配置
```bash
# API密钥
BING_SUBSCRIPTION_KEY=your_bing_key_here
JINA_API_KEY=your_jina_key_here

# 模型路径
DEFAULT_MODEL_PATH=microsoft/DialoGPT-medium

# 缓存目录
CACHE_DIR=./cache
```

#### `requirements.txt` - 依赖包列表
包含所有必需的Python包及其版本号

#### `configs/config.json` - 主配置文件
```json
{
  "api_keys": {...},
  "model_paths": {...},
  "gnn_config": {...},
  "inference_config": {...}
}
```

### 3. 数据文件

#### 数据格式标准
所有数据集都需要转换为以下格式：
```json
[
  {
    "Question": "问题文本",
    "Answer": "答案文本",        // 问答任务
    "Correct Choice": "A"       // 多选题
  }
]
```

#### 缓存文件
- `search_cache.json`: 搜索结果缓存，避免重复搜索
- `url_cache.json`: URL内容缓存，避免重复下载
- `kg_cache/`: 知识图谱和嵌入文件缓存

### 4. 输出文件

#### `outputs/dataset.model.search_o1_kg/`
- `test.info_extract.json`: 详细推理记录，包含每步的KG增强信息
- `test.output.json`: 模型原始输出文本
- `test.metrics.json`: 性能评估指标

### 5. 工具脚本

#### `scripts/setup_environment.py` - 🔧 **环境配置**
- **功能**: 自动安装依赖、配置环境
- **使用**: `python scripts/setup_environment.py --install`

#### `scripts/download_datasets.py` - 📥 **数据下载**
- **功能**: 下载和标准化各种数据集
- **使用**: `python scripts/download_datasets.py --dataset gpqa`

#### `scripts/quick_start.py` - ⚡ **快速测试**
- **功能**: 测试所有核心功能是否正常
- **使用**: `python scripts/quick_start.py --test all`

## 🔄 工作流程

### 1. 环境配置流程
```
setup_environment.py → 安装依赖 → 创建配置 → 快速测试 → 验证成功
```

### 2. 数据准备流程
```
download_datasets.py → 下载原始数据 → 标准化格式 → 保存到data/目录
```

### 3. 推理执行流程
```
run_search_o1_kg.py → 加载模型 → 读取问题 → 搜索文档 → 构建KG → 推理 → 生成答案
```

### 4. 结果分析流程
```
推理输出 → 保存到outputs/ → 分析脚本 → 可视化 → 性能报告
```

## 🎨 模块关系图

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   run_search_   │    │  EntityExtractor │    │ KnowledgeGraph  │
│   o1_kg.py      │───▶│   (实体抽取)     │───▶│    Builder      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       ▼
         │              ┌─────────────────┐    ┌─────────────────┐
         │              │ RelationExtractor│    │ GraphStorage    │
         │              │   (关系抽取)     │    │   (图谱存储)     │
         │              └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Multimodal     │    │ GraphReasoning  │    │  Visualization  │
│  Aligner        │◀───│    Engine       │◀───│     Engine       │
│  (跨模态对齐)    │    │   (图推理)       │    │    (可视化)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📝 使用示例

### 快速开始
```bash
# 1. 环境配置
python scripts/setup_environment.py --install

# 2. 快速测试
python scripts/quick_start.py --test all

# 3. 创建样本数据
python scripts/quick_start.py --create-samples

# 4. 运行推理（使用样本数据）
python scripts/run_search_o1_kg.py \
    --dataset_name gpqa \
    --split diamond_sample \
    --model_path microsoft/DialoGPT-medium \
    --bing_subscription_key YOUR_KEY \
    --subset_num 2
```

### 完整实验
```bash
# 1. 下载完整数据集
python scripts/download_datasets.py --dataset gpqa

# 2. 运行大规模实验
python scripts/run_search_o1_kg.py \
    --dataset_name gpqa \
    --split diamond \
    --model_path YOUR_MODEL_PATH \
    --bing_subscription_key YOUR_KEY \
    --jina_api_key YOUR_JINA_KEY \
    --max_search_limit 5 \
    --max_turn 15

# 3. 分析结果
python scripts/evaluate_kg.py \
    --input outputs/gpqa.model.search_o1_kg/test.output.json \
    --data data/GPQA/diamond.json
```

## ⚠️ 注意事项

### 1. 文件权限
- 确保脚本有执行权限：`chmod +x scripts/*.py`
- 确保缓存目录可写：`chmod -R 755 cache/ outputs/`

### 2. 路径配置
- 所有路径都使用相对路径，基于项目根目录
- Python路径自动添加到sys.path

### 3. 依赖版本
- 严格按照requirements.txt中的版本号安装
- 某些包可能需要特定版本的CUDA

### 4. 资源需求
- GPU内存：至少8GB（推荐16GB+）
- 系统内存：至少16GB
- 磁盘空间：至少50GB

## 🆘 故障排除

### 常见问题及解决方案
1. **模块导入失败**: 检查Python路径和虚拟环境
2. **CUDA内存不足**: 减少batch_size或使用CPU模式
3. **API调用失败**: 检查密钥配置和网络连接
4. **数据格式错误**: 参考标准化格式重新处理

### 日志文件位置
- 环境配置日志：`logs/setup_report.json`
- 实验日志：`logs/experiment_*.log`
- 错误日志：`logs/error_*.log`

---

**文档更新时间**: 2025-01-19
**版本**: v1.0.0

如有问题，请参考 `USAGE_GUIDE.md` 或提交Issue。