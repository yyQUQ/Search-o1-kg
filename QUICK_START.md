# 🚀 Search-o1-KG 快速启动指南

## ⚡ 5分钟快速体验

### 1. 环境准备
```bash
cd /home/yy/projects/search-o1-kg

# 创建虚拟环境
conda create -n search_o1_kg python=3.9
conda activate search_o1_kg

# 安装依赖
pip install -r requirements.txt

# 下载spaCy模型
python -m spacy download en_core_web_sm
```

### 2. 环境配置
```bash
# 自动配置环境
python scripts/setup_environment.py --install

# 或手动创建.env文件
echo "BING_SUBSCRIPTION_KEY=your_key_here" > .env
```

### 3. 快速测试
```bash
# 测试所有核心功能
python scripts/quick_start.py --test all

# 创建样本数据
python scripts/quick_start.py --create-samples
```

### 4. 运行演示
```bash
# 使用样本数据运行（无需真实API密钥）
python scripts/run_search_o1_kg.py \
    --dataset_name gpqa \
    --split diamond_sample \
    --model_path microsoft/DialoGPT-medium \
    --subset_num 1 \
    --max_search_limit 1
```

## 📋 完整使用流程

### 步骤1: 环境配置
```bash
# 克隆项目
git clone <your-repo-url>
cd search-o1-kg

# 环境配置
python scripts/setup_environment.py --install

# 验证安装
python scripts/quick_start.py --test all
```

### 步骤2: API密钥配置
```bash
# 编辑.env文件
nano .env

# 添加以下内容：
BING_SUBSCRIPTION_KEY=your_bing_key_here
JINA_API_KEY=your_jina_key_here  # 可选
```

### 步骤3: 数据准备
```bash
# 下载样本数据
python scripts/download_datasets.py --dataset sample

# 或下载完整数据集
python scripts/download_datasets.py --dataset gpqa
python scripts/download_datasets.py --dataset math500
```

### 步骤4: 运行实验
```bash
# GPQA科学问答
python scripts/run_search_o1_kg.py \
    --dataset_name gpqa \
    --split diamond \
    --model_path "your_model_path" \
    --bing_subscription_key "$BING_SUBSCRIPTION_KEY" \
    --subset_num 50

# 数学推理
python scripts/run_search_o1_kg.py \
    --dataset_name math500 \
    --split test \
    --model_path "your_model_path" \
    --bing_subscription_key "$BING_SUBSCRIPTION_KEY" \
    --subset_num 30
```

## 🎯 关键文件说明

| 文件路径 | 作用 | 使用方法 |
|---------|------|----------|
| `scripts/run_search_o1_kg.py` | 主推理脚本 | 核心运行文件 |
| `src/knowledge_graph/entity_extractor.py` | 实体抽取 | 自动识别实体 |
| `src/gnn_reasoning/reasoning_engine.py` | 图推理引擎 | 知识图谱推理 |
| `src/multimodal_alignment/multimodal_aligner.py` | 跨模态对齐 | 文本-图谱对齐 |

## 🔧 常用命令

### 环境管理
```bash
# 检查环境
python scripts/setup_environment.py --quick

# 重新安装
python scripts/setup_environment.py --install

# 测试功能
python scripts/quick_start.py --test all
```

### 数据管理
```bash
# 下载数据集
python scripts/download_datasets.py --dataset gpqa

# 创建样本数据
python scripts/quick_start.py --create-samples

# 查看数据格式
python scripts/preprocess_data.py --check-format
```

### 实验运行
```bash
# 小规模测试
python scripts/run_search_o1_kg.py --subset_num 1

# 中等规模实验
python scripts/run_search_o1_kg.py --subset_num 50

# 大规模实验
python scripts/run_search_o1_kg.py --subset_num -1  # 全部数据
```

## 📊 结果查看

### 输出文件位置
```
outputs/
└── dataset.model.search_o1_kg/
    ├── test.info_extract.json  # 详细推理记录
    ├── test.output.json        # 模型输出
    └── test.metrics.json      # 性能指标
```

### 分析结果
```python
# 创建分析脚本
import json

# 读取结果
with open("outputs/gpqa.model.search_o1_kg/test.info_extract.json", "r") as f:
    results = json.load(f)

print(f"处理了 {len(results)} 个样本")
for i, result in enumerate(results[:3]):
    print(f"样本 {i+1}: KG增强 = {result.get('kg_enhanced', False)}")
```

## ⚠️ 常见问题

### Q: 模型加载失败
```bash
# 解决方案：使用小模型测试
export MODEL_PATH="microsoft/DialoGPT-medium"
python scripts/run_search_o1_kg.py --model_path "$MODEL_PATH" --subset_num 1
```

### Q: GPU内存不足
```bash
# 解决方案：减少参数
python scripts/run_search_o1_kg.py \
    --subset_num 1 \
    --gnn_hidden_dim 64 \
    --max_doc_len 1000
```

### Q: API密钥错误
```bash
# 解决方案：检查.env文件
cat .env
# 确保密钥正确且没有额外空格
```

### Q: 依赖安装失败
```bash
# 解决方案：升级pip并重装
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## 📞 获取帮助

1. **查看详细文档**: `USAGE_GUIDE.md`
2. **文件结构说明**: `FILE_STRUCTURE.md`
3. **项目总结**: `PROJECT_SUMMARY.md`
4. **提交Issue**: GitHub Issues

## 🎉 成功指标

运行成功后，你应该看到：

```
开始Search-o1-KG推理，数据集: gpqa
加载了 X 条数据
初始化语言模型...
初始化知识图谱组件...
-------------- Turn 1 --------------
We have 1 sequences needing generation...
Generation completed, processing outputs...
Batch processing 1 sequences with enhanced KG reasoning...
知识图谱构建完成：X 个实体，Y 个关系
Search-o1-KG inference completed!
```

---

**🎯 恭喜！你已经成功运行了Search-o1-KG！**

下一步：
1. 尝试不同的数据集
2. 调整GNN参数
3. 分析推理结果
4. 撰写实验报告

**祝你使用愉快！** 🚀