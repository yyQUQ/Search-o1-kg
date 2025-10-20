"""
Search-o1-KG: 动态知识图谱增强的推理框架

基于Search-o1框架的核心创新点，结合动态知识图谱构建、
图神经网络推理和跨模态对齐技术，实现更强大的智能体RAG能力。

主要改进：
1. 动态知识图谱构建：从检索文档中自动抽取实体/关系/公式
2. 图神经网络推理：在推理链中引入GNN模块进行路径推理
3. 跨模态对齐：将文本推理步骤与图谱节点对齐，支持可视化溯源
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .knowledge_graph import KnowledgeGraphBuilder, EntityExtractor, RelationExtractor
from .gnn_reasoning import GraphReasoningEngine, GraphNeuralNetwork
from .multimodal_alignment import MultimodalAligner, VisualizationEngine

__all__ = [
    'KnowledgeGraphBuilder',
    'EntityExtractor',
    'RelationExtractor',
    'GraphReasoningEngine',
    'GraphNeuralNetwork',
    'MultimodalAligner',
    'VisualizationEngine'
]