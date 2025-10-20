# Search-o1-KG 使用指南

## 📋 目录
1. [环境配置](#环境配置)
2. [数据集下载与处理](#数据集下载与处理)
3. [项目文件结构说明](#项目文件结构说明)
4. [运行项目](#运行项目)
5. [API密钥配置](#api密钥配置)
6. [故障排除](#故障排除)
7. [结果分析](#结果分析)

## 🔧 环境配置

### 1. 系统要求
- Python 3.9+
- CUDA 11.0+ (推荐使用GPU)
- 至少16GB内存
- 50GB可用磁盘空间

### 2. 创建虚拟环境
```bash
# 创建conda环境
conda create -n search_o1_kg python=3.9
conda activate search_o1_kg

# 或者使用venv
python -m venv search_o1_kg_env
source search_o1_kg_env/bin/activate  # Linux/Mac
# search_o1_kg_env\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
cd /home/yy/projects/search-o1-kg

# 安装基础依赖
pip install -r requirements.txt

# 下载spaCy模型
python -m spacy download en_core_web_sm

# 安装额外的依赖（如果遇到问题）
pip install --upgrade pip
pip install setuptools wheel
```

### 4. 安装vLLM（用于高效推理）
```bash
# 安装vLLM（推荐GPU版本）
pip install vllm

# 如果CPU版本，可能需要额外配置
# export VLLM_USE_MODELSCOPE=1
```

## 📥 数据集下载与处理

### 1. 支持的数据集

#### 科学推理数据集
- **GPQA** (Graduate-Level Google-Proof Q&A)
- **MATH500** (数学推理数据集)
- **AIME** (American Invitational Mathematics Examination)
- **AMC** (American Mathematics Competitions)

#### 开放域问答数据集
- **NQ** (Natural Questions)
- **TriviaQA**
- **HotpotQA** (多跳问答)
- **2WikiMultihopQA**
- **MuSiQue**
- **Bamboogle**

#### 代码数据集
- **LiveCodeBench**

### 2. 数据集下载方法

#### 方法1: 使用Hugging Face Datasets（推荐）

```python
# 创建下载脚本 download_datasets.py
from datasets import load_dataset
import json
import os

# 下载数据集函数
def download_and_save_dataset(dataset_name, split, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if dataset_name == "gpqa":
        dataset = load_dataset("Idavidrein/gpqa", "diamond")
        data = dataset["train"]
    elif dataset_name == "math500":
        dataset = load_dataset("allenai/MathHub", "MATH500")
        data = dataset["test"]
    elif dataset_name == "nq":
        dataset = load_dataset("nq_open", "validation")
        data = dataset
    elif dataset_name == "triviaqa":
        dataset = load_dataset("trivia_qa", "unfiltered")
        data = dataset["validation"]
    elif dataset_name == "hotpotqa":
        dataset = load_dataset("hotpot_qa", "fullwiki")
        data = dataset["validation"]
    else:
        print(f"Dataset {dataset_name} not yet configured")
        return

    # 转换为标准格式
    processed_data = []
    for item in data:
        processed_item = {
            "Question": item["question"] if "question" in item else item.get("Question", ""),
            "Answer": item.get("answer", item.get("Answer", "")),
            "Correct Choice": item.get("correct_choice", item.get("Correct Choice", ""))
        }
        processed_data.append(processed_item)

    # 保存文件
    output_file = os.path.join(output_dir, f"{split}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"Dataset {dataset_name} saved to {output_file}")

# 下载示例
download_and_save_dataset("gpqa", "diamond", "./data/GPQA")
download_and_save_dataset("math500", "test", "./data/MATH500")
```

#### 方法2: 手动下载和处理

```bash
# 创建数据目录
mkdir -p data/{GPQA,MATH500,QA_Datasets,LIVECODEBENCH}

# GPQA数据集
cd data/GPQA
wget https://huggingface.co/datasets/Idavidrein/gpqa/resolve/main/diamond/train.jsonl
# 转换为JSON格式

# MATH500数据集
cd ../MATH500
wget https://huggingface.co/datasets/allenai/MathHub/resolve/main/MATH500/test.jsonl
# 转换为JSON格式
```

### 3. 数据格式标准化

所有数据集需要转换为以下格式：

```json
[
  {
    "Question": "问题文本",
    "Answer": "答案文本",  // 可选
    "Correct Choice": "正确选择"  // 多选题，可选
  }
]
```

### 4. 数据预处理脚本

创建 `scripts/preprocess_data.py`:

```python
import json
import os
import re

def standardize_dataset(input_file, output_file, dataset_type="qa"):
    """标准化数据集格式"""

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    standardized_data = []

    for item in data:
        standardized_item = {
            "Question": item.get("Question", item.get("question", "")),
        }

        # 根据数据类型添加相应字段
        if dataset_type == "multiple_choice":
            standardized_item["Correct Choice"] = item.get("Correct Choice", item.get("correct_choice", ""))
        elif dataset_type == "qa":
            standardized_item["Answer"] = item.get("Answer", item.get("answer", ""))

        # 清理问题文本
        question = standardized_item["Question"]
        question = re.sub(r'\s+', ' ', question).strip()
        standardized_item["Question"] = question

        standardized_data.append(standardized_item)

    # 保存标准化数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(standardized_data, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(standardized_data)} items from {input_file} to {output_file}")

# 使用示例
if __name__ == "__main__":
    # 处理不同类型的数据集
    standardize_dataset("raw_gpqa.json", "data/GPQA/diamond.json", "multiple_choice")
    standardize_dataset("raw_math500.json", "data/MATH500/test.json", "qa")
    standardize_dataset("raw_nq.json", "data/QA_Datasets/nq.json", "qa")
```

## 📁 项目文件结构说明

### 根目录文件

```
search-o1-kg/
├── README.md                   # 项目介绍文档
├── USAGE_GUIDE.md             # 使用指南（本文档）
├── PROJECT_SUMMARY.md         # 项目总结
├── requirements.txt           # Python依赖列表
├── LICENSE                    # 开源许可证（MIT）
```

### src/ 目录（核心模块）

```
src/
├── __init__.py                # 包初始化文件，导出主要类
├── knowledge_graph/           # 知识图谱构建模块
│   ├── __init__.py           # 导出图谱相关类
│   ├── entity_extractor.py   # 实体抽取器（核心文件）
│   ├── relation_extractor.py # 关系抽取器
│   ├── graph_builder.py      # 图谱构建器
│   └── graph_storage.py      # 图谱存储系统
├── gnn_reasoning/             # 图神经网络推理模块
│   ├── __init__.py           # 导出推理相关类
│   ├── graph_neural_network.py # GNN实现（GCN/GAT/GraphSAGE）
│   ├── reasoning_engine.py   # 推理引擎（核心文件）
│   ├── path_finder.py        # 路径查找器
│   └── entity_linker.py      # 实体链接器
└── multimodal_alignment/      # 跨模态对齐模块
    ├── __init__.py           # 导出对齐相关类
    ├── multimodal_aligner.py # 对齐器（核心文件）
    ├── visualization_engine.py # 可视化引擎
    └── alignment_utils.py    # 对齐工具
```

### scripts/ 目录

```
scripts/
├── run_search_o1_kg.py       # 主推理脚本（最重要的文件）
├── download_datasets.py      # 数据集下载脚本
├── preprocess_data.py        # 数据预处理脚本
├── evaluate_kg.py           # 知识图谱评估脚本
└── visualize_reasoning.py   # 推理可视化脚本
```

### 其他目录

```
data/                        # 数据目录
├── GPQA/                   # GPQA数据集
├── MATH500/                # MATH500数据集
├── QA_Datasets/            # 问答数据集
└── LIVECODEBENCH/          # 代码数据集

cache/                      # 缓存目录
├── search_cache.json       # 搜索结果缓存
├── url_cache.json         # URL内容缓存
└── kg_cache/              # 知识图谱缓存

outputs/                    # 输出目录
├── runs.baselines/        # 基线模型结果
├── runs.qa/              # 问答任务结果
└── runs.analysis/        # 分析结果

tests/                      # 测试目录
├── test_kg.py            # 知识图谱测试
├── test_gnn.py           # GNN测试
└── test_alignment.py     # 对齐测试

docs/                       # 文档目录
├── api_reference.md      # API参考文档
└── examples.md           # 使用示例
```

## 🚀 运行项目

### 1. 准备API密钥

#### Bing Search API
```bash
# 1. 访问 https://azure.microsoft.com/en-us/services/cognitive-services/bing-web-search-api/
# 2. 创建Azure账户并订阅Bing Search API
# 3. 获取订阅密钥
export BING_SUBSCRIPTION_KEY="your_bing_subscription_key_here"
```

#### Jina API（可选）
```bash
# 1. 访问 https://jina.ai/reader/
# 2. 注册并获取API密钥
export JINA_API_KEY="your_jina_api_key_here"
```

### 2. 准备模型

#### 使用Hugging Face模型
```bash
# 推荐的模型选择
export MODEL_PATH="microsoft/DialoGPT-medium"  # 示例
# 或使用其他大模型
export MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
export MODEL_PATH="Qwen/Qwen-7B-Chat"
```

### 3. 基本运行命令

```bash
# 基本推理命令
cd /home/yy/projects/search-o1-kg

python scripts/run_search_o1_kg.py \
    --dataset_name gpqa \
    --split diamond \
    --model_path "your_model_path" \
    --bing_subscription_key "your_bing_key" \
    --jina_api_key "your_jina_key" \
    --subset_num 10  # 先测试10个样本
```

### 4. 详细参数说明

#### 核心参数
```bash
--dataset_name          # 数据集名称 [gpqa, math500, aime, nq, hotpotqa等]
--split                 # 数据分割 [test, diamond, train, validation]
--model_path           # 预训练模型路径
--subset_num           # 处理样本数量（-1表示全部）
```

#### 搜索参数
```bash
--max_search_limit     # 最大搜索次数 [默认: 10]
--max_turn             # 最大推理轮次 [默认: 15]
--top_k                # 返回文档数量 [默认: 10]
--max_doc_len          # 文档最大长度 [默认: 3000]
--use_jina             # 是否使用Jina API [默认: True]
```

#### GNN参数
```bash
--gnn_hidden_dim       # GNN隐藏层维度 [默认: 128]
--gnn_output_dim       # GNN输出维度 [默认: 64]
--gnn_num_layers       # GNN层数 [默认: 3]
--gnn_dropout          # Dropout率 [默认: 0.1]
```

#### 采样参数
```bash
--temperature          # 采样温度 [默认: 0.7]
--top_p               # Top-p采样 [默认: 0.8]
--top_k_sampling      # Top-k采样 [默认: 20]
--repetition_penalty  # 重复惩罚 [默认: 1.05]
```

### 5. 不同数据集的运行示例

#### GPQA（科学问答）
```bash
python scripts/run_search_o1_kg.py \
    --dataset_name gpqa \
    --split diamond \
    --model_path "microsoft/DialoGPT-medium" \
    --bing_subscription_key "$BING_SUBSCRIPTION_KEY" \
    --jina_api_key "$JINA_API_KEY" \
    --max_search_limit 5 \
    --max_turn 15 \
    --subset_num 50
```

#### MATH500（数学推理）
```bash
python scripts/run_search_o1_kg.py \
    --dataset_name math500 \
    --split test \
    --model_path "microsoft/DialoGPT-medium" \
    --bing_subscription_key "$BING_SUBSCRIPTION_KEY" \
    --jina_api_key "$JINA_API_KEY" \
    --max_search_limit 3 \
    --max_turn 10 \
    --subset_num 30
```

#### HotpotQA（多跳问答）
```bash
python scripts/run_search_o1_kg.py \
    --dataset_name hotpotqa \
    --split validation \
    --model_path "microsoft/DialoGPT-medium" \
    --bing_subscription_key "$BING_SUBSCRIPTION_KEY" \
    --jina_api_key "$JINA_API_KEY" \
    --max_search_limit 10 \
    --max_turn 15 \
    --subset_num 20
```

## 🔑 API密钥配置

### 1. 环境变量方式（推荐）

创建 `.env` 文件：
```bash
# .env 文件内容
BING_SUBSCRIPTION_KEY=your_actual_bing_key_here
JINA_API_KEY=your_actual_jina_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here  # 如果需要私有模型
```

加载环境变量：
```bash
# 安装python-dotenv
pip install python-dotenv

# 在脚本开始处添加
from dotenv import load_dotenv
load_dotenv()
```

### 2. 命令行参数方式

```bash
# 直接在命令行中传递
python scripts/run_search_o1_kg.py \
    --dataset_name gpqa \
    --model_path "model_path" \
    --bing_subscription_key "your_key" \
    --jina_api_key "your_key"
```

### 3. 配置文件方式

创建 `config/config.json`:
```json
{
  "api_keys": {
    "bing_subscription_key": "your_key",
    "jina_api_key": "your_key"
  },
  "model_paths": {
    "default": "microsoft/DialoGPT-medium",
    "math": "microsoft/DialoGPT-medium"
  },
  "gnn_config": {
    "hidden_dim": 128,
    "output_dim": 64,
    "num_layers": 3,
    "dropout": 0.1
  }
}
```

## 🔧 故障排除

### 1. 常见错误及解决方案

#### CUDA内存不足
```bash
# 错误: CUDA out of memory
# 解决方案:
python scripts/run_search_o1_kg.py \
    --subset_num 1 \  # 减少样本数量
    --gnn_hidden_dim 64 \  # 减少GNN维度
    --max_doc_len 1000  # 减少文档长度
```

#### 模型加载失败
```bash
# 错误: model not found
# 解决方案:
# 1. 检查模型路径是否正确
# 2. 确保有网络连接访问Hugging Face
# 3. 使用本地模型路径
export MODEL_PATH="/path/to/your/local/model"
```

#### API调用失败
```bash
# 错误: API key invalid
# 解决方案:
# 1. 检查API密钥是否正确
# 2. 确认API配额是否充足
# 3. 检查网络连接
```

#### 依赖包冲突
```bash
# 错误: package version conflict
# 解决方案:
pip install --upgrade pip
pip install --force-reinstall -r requirements.txt
```

### 2. 调试模式

```bash
# 启用详细日志
export PYTHONPATH=/home/yy/projects/search-o1-kg/src:$PYTHONPATH
python -u scripts/run_search_o1_kg.py --dataset_name gpqa --subset_num 1 --model_path "model_path" --bing_subscription_key "key" 2>&1 | tee debug.log
```

### 3. 性能监控

```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 监控内存使用
htop

# 监控进程
ps aux | grep python
```

## 📊 结果分析

### 1. 输出文件说明

运行完成后，会在 `outputs/` 目录生成以下文件：

```
outputs/
├── dataset_name.model_name.search_o1_kg/
│   ├── test.info_extract.json           # 详细推理记录
│   ├── test.output.json                 # 模型输出
│   └── test.metrics.json               # 评估指标
```

### 2. 查看推理结果

```python
# 创建分析脚本 analyze_results.py
import json

def analyze_results(output_file):
    with open(output_file, 'r') as f:
        results = json.load(f)

    print(f"总共处理了 {len(results)} 个样本")

    for i, result in enumerate(results[:5]):  # 查看前5个结果
        print(f"\n样本 {i+1}:")
        print(f"输入长度: {len(result['prompt'])}")
        print(f"输出长度: {len(result.get('raw_output', ''))}")
        print(f"是否使用KG增强: {result.get('kg_enhanced', False)}")
        if 'extracted_info' in result:
            print(f"提取信息长度: {len(result['extracted_info'])}")

# 使用示例
analyze_results("outputs/gpqa.model_name.search_o1_kg/test.info_extract.json")
```

### 3. 知识图谱分析

```python
# 创建KG分析脚本 analyze_kg.py
import sys
sys.path.append('/home/yy/projects/search-o1-kg/src')
from knowledge_graph import KnowledgeGraphBuilder

def analyze_knowledge_graph(documents):
    builder = KnowledgeGraphBuilder()
    kg = builder.build_graph_from_documents(documents)

    print("知识图谱统计:")
    print(f"实体数量: {len(kg.entities)}")
    print(f"关系数量: {len(kg.relations)}")
    print(f"图节点数: {kg.graph.number_of_nodes()}")
    print(f"图边数: {kg.graph.number_of_edges()}")

    # 实体类型分布
    entity_types = {}
    for entity in kg.entities.values():
        entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1

    print("\n实体类型分布:")
    for entity_type, count in entity_types.items():
        print(f"  {entity_type}: {count}")

    return kg

# 使用示例
documents = ["Einstein worked at Princeton University.", "E=mc² is his famous equation."]
kg = analyze_knowledge_graph(documents)
```

### 4. 性能对比

创建性能对比脚本：

```python
# performance_comparison.py
import matplotlib.pyplot as plt
import json

def compare_performance():
    # 假设的结果数据
    models = ['Search-o1', 'Search-o1-KG']
    datasets = ['GPQA', 'MATH500', 'HotpotQA']

    # 示例数据（需要替换为实际结果）
    results = {
        'GPQA': [58.2, 62.1],
        'MATH500': [65.4, 68.7],
        'HotpotQA': [72.1, 75.8]
    }

    # 绘制对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(models))

    for i, dataset in enumerate(datasets):
        offset = i * 0.25
        ax.bar([xi + offset for xi in x], results[dataset], width=0.25,
               label=dataset, alpha=0.8)

    ax.set_xlabel('模型')
    ax.set_ylabel('准确率 (%)')
    ax.set_title('Search-o1 vs Search-o1-KG 性能对比')
    ax.set_xticks([xi + 0.25 for xi in x])
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/performance_comparison.png')
    plt.show()

if __name__ == "__main__":
    compare_performance()
```

## 📝 快速开始清单

运行项目前的检查清单：

- [ ] Python 3.9+ 环境已创建
- [ ] 所有依赖已安装 (`pip install -r requirements.txt`)
- [ ] spaCy模型已下载 (`python -m spacy download en_core_web_sm`)
- [ ] 数据集已下载并处理为正确格式
- [ ] API密钥已配置
- [ ] 模型路径已设置
- [ ] 缓存目录已创建 (`mkdir -p cache outputs`)

完成以上步骤后，运行以下命令测试：

```bash
cd /home/yy/projects/search-o1-kg
python scripts/run_search_o1_kg.py \
    --dataset_name gpqa \
    --split diamond \
    --subset_num 1 \
    --model_path "microsoft/DialoGPT-medium" \
    --bing_subscription_key "your_key"
```

如果运行成功，你应该看到类似输出：
```
开始Search-o1-KG推理，数据集: gpqa
加载了 X 条数据
初始化语言模型...
初始化知识图谱组件...
-------------- Turn 1 --------------
We have 1 sequences needing generation...
Generation completed, processing outputs...
Batch processing 1 sequences with enhanced KG reasoning...
Batch generation completed, assigning outputs to sequences...
Search-o1-KG inference completed!
```

## 🆘 获取帮助

如果遇到问题，可以：

1. 查看详细日志文件
2. 运行测试脚本验证安装
3. 检查GitHub Issues
4. 联系项目维护者

---

**祝您使用愉快！如有问题，请查阅文档或提交Issue。**