"""
跨模态对齐器

实现文本推理步骤与知识图谱节点的对齐，
支持多种对齐策略和相似度计算方法。
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from ..knowledge_graph.graph_builder import KnowledgeGraph, Entity, Relation
from ..gnn_reasoning.reasoning_engine import ReasoningPath, ReasoningStep

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AlignmentResult:
    """对齐结果"""
    text_segment: str
    entity_id: str
    entity_text: str
    entity_type: str
    confidence: float
    alignment_type: str  # 'exact', 'partial', 'semantic'
    context: str = ""

@dataclass
class ReasoningAlignment:
    """推理对齐结果"""
    reasoning_step: ReasoningStep
    aligned_entities: List[AlignmentResult]
    aligned_relations: List[AlignmentResult]
    overall_confidence: float
    visualization_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.visualization_data is None:
            self.visualization_data = {}

class MultimodalAligner:
    """跨模态对齐器"""

    def __init__(self, kg: KnowledgeGraph):
        """
        初始化对齐器

        Args:
            kg: 知识图谱
        """
        self.kg = kg

        # 初始化文本特征提取器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        # 构建实体文本语料库
        self._build_entity_corpus()

        # 对齐配置
        self.alignment_config = {
            'exact_match_threshold': 0.9,
            'partial_match_threshold': 0.6,
            'semantic_similarity_threshold': 0.3,
            'context_window_size': 50,
            'max_alignments_per_step': 5
        }

        # 缓存相似度计算结果
        self.similarity_cache = {}

    def _build_entity_corpus(self):
        """构建实体文本语料库"""
        entity_texts = []
        self.entity_mapping = {}

        for entity_id, entity in self.kg.entities.items():
            # 组合实体文本和上下文
            text = f"{entity.text} {entity.context}"
            entity_texts.append(text)
            self.entity_mapping[len(entity_texts) - 1] = entity_id

        # 训练TF-IDF向量化器
        if entity_texts:
            self.entity_tfidf_matrix = self.tfidf_vectorizer.fit_transform(entity_texts)
        else:
            self.entity_tfidf_matrix = None

    def align_reasoning_step(self, reasoning_step: ReasoningStep, reasoning_text: str) -> ReasoningAlignment:
        """
        对齐推理步骤

        Args:
            reasoning_step: 推理步骤
            reasoning_text: 推理文本

        Returns:
            推理对齐结果
        """
        # 提取文本片段
        text_segments = self._extract_text_segments(reasoning_text)

        # 对齐实体
        aligned_entities = self._align_entities(text_segments)

        # 对齐关系
        aligned_relations = self._align_relations(text_segments, reasoning_step.relations)

        # 计算整体置信度
        overall_confidence = self._compute_overall_confidence(aligned_entities, aligned_relations)

        # 生成可视化数据
        visualization_data = self._generate_visualization_data(
            reasoning_step, aligned_entities, aligned_relations
        )

        return ReasoningAlignment(
            reasoning_step=reasoning_step,
            aligned_entities=aligned_entities,
            aligned_relations=aligned_relations,
            overall_confidence=overall_confidence,
            visualization_data=visualization_data
        )

    def _extract_text_segments(self, text: str) -> List[str]:
        """提取文本片段"""
        # 按句子分割
        sentences = re.split(r'[.!?]+', text)
        segments = [s.strip() for s in sentences if s.strip()]

        # 如果句子太少，按分号分割
        if len(segments) < 2:
            segments = re.split(r'[;]+', text)
            segments = [s.strip() for s in segments if s.strip()]

        return segments

    def _align_entities(self, text_segments: List[str]) -> List[AlignmentResult]:
        """对齐实体"""
        alignments = []

        for segment in text_segments:
            # 精确匹配
            exact_alignments = self._exact_match_entities(segment)
            alignments.extend(exact_alignments)

            # 部分匹配
            partial_alignments = self._partial_match_entities(segment)
            alignments.extend(partial_alignments)

            # 语义相似度匹配
            semantic_alignments = self._semantic_match_entities(segment)
            alignments.extend(semantic_alignments)

        # 去重和过滤
        filtered_alignments = self._filter_alignments(alignments)

        # 限制每步的对齐数量
        return filtered_alignments[:self.alignment_config['max_alignments_per_step']]

    def _exact_match_entities(self, text: str) -> List[AlignmentResult]:
        """精确匹配实体"""
        alignments = []
        text_lower = text.lower()

        for entity_id, entity in self.kg.entities.items():
            entity_text_lower = entity.text.lower()

            # 精确匹配
            if entity_text_lower == text_lower:
                alignment = AlignmentResult(
                    text_segment=text,
                    entity_id=entity_id,
                    entity_text=entity.text,
                    entity_type=entity.entity_type,
                    confidence=1.0,
                    alignment_type='exact',
                    context=self._get_context(entity_id, text)
                )
                alignments.append(alignment)
            elif entity_text_lower in text_lower or text_lower in entity_text_lower:
                # 包含匹配
                confidence = min(len(entity_text_lower), len(text_lower)) / max(len(entity_text_lower), len(text_lower))
                if confidence >= self.alignment_config['exact_match_threshold']:
                    alignment = AlignmentResult(
                        text_segment=text,
                        entity_id=entity_id,
                        entity_text=entity.text,
                        entity_type=entity.entity_type,
                        confidence=confidence,
                        alignment_type='exact',
                        context=self._get_context(entity_id, text)
                    )
                    alignments.append(alignment)

        return alignments

    def _partial_match_entities(self, text: str) -> List[AlignmentResult]:
        """部分匹配实体"""
        alignments = []
        text_words = set(text.lower().split())

        for entity_id, entity in self.kg.entities.items():
            entity_words = set(entity.text.lower().split())

            # 计算词汇重叠度
            overlap = len(text_words & entity_words)
            union = len(text_words | entity_words)

            if union > 0:
                jaccard_similarity = overlap / union
                if jaccard_similarity >= self.alignment_config['partial_match_threshold']:
                    alignment = AlignmentResult(
                        text_segment=text,
                        entity_id=entity_id,
                        entity_text=entity.text,
                        entity_type=entity.entity_type,
                        confidence=jaccard_similarity,
                        alignment_type='partial',
                        context=self._get_context(entity_id, text)
                    )
                    alignments.append(alignment)

        return alignments

    def _semantic_match_entities(self, text: str) -> List[AlignmentResult]:
        """语义匹配实体"""
        if self.entity_tfidf_matrix is None:
            return []

        alignments = []

        # 计算文本的TF-IDF向量
        try:
            text_vector = self.tfidf_vectorizer.transform([text])

            # 计算与所有实体的相似度
            similarities = cosine_similarity(text_vector, self.entity_tfidf_matrix).flatten()

            # 找到相似度超过阈值的实体
            for idx, similarity in enumerate(similarities):
                if similarity >= self.alignment_config['semantic_similarity_threshold']:
                    entity_id = self.entity_mapping[idx]
                    entity = self.kg.entities[entity_id]

                    alignment = AlignmentResult(
                        text_segment=text,
                        entity_id=entity_id,
                        entity_text=entity.text,
                        entity_type=entity.entity_type,
                        confidence=float(similarity),
                        alignment_type='semantic',
                        context=self._get_context(entity_id, text)
                    )
                    alignments.append(alignment)

        except Exception as e:
            logger.warning(f"语义匹配失败: {e}")

        return alignments

    def _align_relations(self, text_segments: List[str], target_relations: List[str]) -> List[AlignmentResult]:
        """对齐关系"""
        alignments = []

        for segment in text_segments:
            for target_relation in target_relations:
                # 检查文本中是否包含关系关键词
                if self._contains_relation_keywords(segment, target_relation):
                    # 查找相关的关系实例
                    related_relations = self._find_related_relations(target_relation)

                    for relation in related_relations:
                        alignment = AlignmentResult(
                            text_segment=segment,
                            entity_id=relation.subject + "_" + relation.object,  # 临时ID
                            entity_text=f"{relation.subject} -> {relation.object}",
                            entity_type=relation.predicate,
                            confidence=relation.confidence,
                            alignment_type='relation',
                            context=segment
                        )
                        alignments.append(alignment)

        return alignments

    def _contains_relation_keywords(self, text: str, relation: str) -> bool:
        """检查文本是否包含关系关键词"""
        relation_keywords = {
            'is_a': ['is', 'are', 'type of', 'kind of'],
            'works_at': ['works at', 'employed at', 'researcher at'],
            'located_in': ['located', 'situated', 'found in'],
            'causes': ['causes', 'leads to', 'results in'],
            'equals': ['equals', 'equal to', '='],
            'reacts_with': ['reacts with', 'combines with'],
            'produces': ['produces', 'yields', 'creates']
        }

        keywords = relation_keywords.get(relation, [relation])
        text_lower = text.lower()

        return any(keyword in text_lower for keyword in keywords)

    def _find_related_relations(self, relation_type: str) -> List[Relation]:
        """查找相关关系"""
        related_relations = []

        for relation in self.kg.relations:
            if relation.predicate == relation_type:
                related_relations.append(relation)

        return related_relations

    def _get_context(self, entity_id: str, text: str) -> str:
        """获取实体上下文"""
        entity = self.kg.entities.get(entity_id)
        if not entity:
            return ""

        # 简化实现：返回原始上下文
        return entity.context

    def _filter_alignments(self, alignments: List[AlignmentResult]) -> List[AlignmentResult]:
        """过滤对齐结果"""
        # 按置信度排序
        alignments.sort(key=lambda x: x.confidence, reverse=True)

        # 去重：相同实体的对齐只保留置信度最高的
        seen_entities = set()
        filtered = []

        for alignment in alignments:
            entity_key = (alignment.entity_id, alignment.text_segment)
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                filtered.append(alignment)

        return filtered

    def _compute_overall_confidence(self, entity_alignments: List[AlignmentResult], relation_alignments: List[AlignmentResult]) -> float:
        """计算整体置信度"""
        if not entity_alignments and not relation_alignments:
            return 0.0

        entity_confidence = np.mean([a.confidence for a in entity_alignments]) if entity_alignments else 0.0
        relation_confidence = np.mean([a.confidence for a in relation_alignments]) if relation_alignments else 0.0

        # 加权平均
        total_alignments = len(entity_alignments) + len(relation_alignments)
        if total_alignments == 0:
            return 0.0

        overall_confidence = (
            entity_confidence * len(entity_alignments) +
            relation_confidence * len(relation_alignments)
        ) / total_alignments

        return overall_confidence

    def _generate_visualization_data(self, reasoning_step: ReasoningStep, entity_alignments: List[AlignmentResult], relation_alignments: List[AlignmentResult]) -> Dict[str, Any]:
        """生成可视化数据"""
        visualization_data = {
            'step_id': reasoning_step.step_id,
            'nodes': [],
            'edges': [],
            'text_highlights': []
        }

        # 添加节点
        for alignment in entity_alignments:
            node_data = {
                'id': alignment.entity_id,
                'label': alignment.entity_text,
                'type': alignment.entity_type,
                'confidence': alignment.confidence,
                'color': self._get_node_color(alignment.entity_type),
                'source_text': alignment.text_segment
            }
            visualization_data['nodes'].append(node_data)

        # 添加边
        for alignment in relation_alignments:
            edge_data = {
                'source': alignment.entity_text.split(' -> ')[0],
                'target': alignment.entity_text.split(' -> ')[1],
                'label': alignment.entity_type,
                'confidence': alignment.confidence,
                'source_text': alignment.text_segment
            }
            visualization_data['edges'].append(edge_data)

        # 添加文本高亮
        for alignment in entity_alignments + relation_alignments:
            highlight = {
                'text': alignment.text_segment,
                'entity': alignment.entity_text,
                'confidence': alignment.confidence,
                'type': alignment.alignment_type
            }
            visualization_data['text_highlights'].append(highlight)

        return visualization_data

    def _get_node_color(self, entity_type: str) -> str:
        """获取节点颜色"""
        color_mapping = {
            'PERSON': '#ff6b6b',
            'ORG': '#4ecdc4',
            'LOC': '#45b7d1',
            'MATH': '#96ceb4',
            'CHEM': '#feca57',
            'TECH': '#9b59b6',
            'CONCEPT': '#3498db'
        }
        return color_mapping.get(entity_type, '#95a5a6')

    def align_reasoning_path(self, reasoning_path: ReasoningPath, reasoning_texts: List[str]) -> List[ReasoningAlignment]:
        """
        对齐整个推理路径

        Args:
            reasoning_path: 推理路径
            reasoning_texts: 推理文本列表

        Returns:
            推理对齐结果列表
        """
        alignments = []

        for i, step in enumerate(reasoning_path.steps):
            reasoning_text = reasoning_texts[i] if i < len(reasoning_texts) else ""
            alignment = self.align_reasoning_step(step, reasoning_text)
            alignments.append(alignment)

        return alignments

    def compute_alignment_quality(self, alignments: List[ReasoningAlignment]) -> Dict[str, float]:
        """
        计算对齐质量指标

        Args:
            alignments: 推理对齐结果列表

        Returns:
            质量指标字典
        """
        if not alignments:
            return {
                'overall_confidence': 0.0,
                'entity_coverage': 0.0,
                'relation_coverage': 0.0,
                'alignment_completeness': 0.0
            }

        # 整体置信度
        overall_confidence = np.mean([a.overall_confidence for a in alignments])

        # 实体覆盖率
        total_entity_alignments = sum(len(a.aligned_entities) for a in alignments)
        entity_coverage = total_entity_alignments / len(alignments) if alignments else 0.0

        # 关系覆盖率
        total_relation_alignments = sum(len(a.aligned_relations) for a in alignments)
        relation_coverage = total_relation_alignments / len(alignments) if alignments else 0.0

        # 对齐完整性（有对齐的步骤比例）
        aligned_steps = sum(1 for a in alignments if a.aligned_entities or a.aligned_relations)
        alignment_completeness = aligned_steps / len(alignments)

        return {
            'overall_confidence': overall_confidence,
            'entity_coverage': entity_coverage,
            'relation_coverage': relation_coverage,
            'alignment_completeness': alignment_completeness
        }

    def generate_alignment_summary(self, alignments: List[ReasoningAlignment]) -> str:
        """
        生成对齐摘要

        Args:
            alignments: 推理对齐结果列表

        Returns:
            对齐摘要文本
        """
        quality_metrics = self.compute_alignment_quality(alignments)

        summary = f"""
对齐摘要:
- 整体置信度: {quality_metrics['overall_confidence']:.2f}
- 实体覆盖率: {quality_metrics['entity_coverage']:.2f}
- 关系覆盖率: {quality_metrics['relation_coverage']:.2f}
- 对齐完整性: {quality_metrics['alignment_completeness']:.2f}

详细对齐结果:
"""

        for i, alignment in enumerate(alignments):
            summary += f"\n步骤 {alignment.reasoning_step.step_id + 1}:\n"
            summary += f"  推理: {alignment.reasoning_step.explanation}\n"
            summary += f"  置信度: {alignment.overall_confidence:.2f}\n"

            if alignment.aligned_entities:
                summary += "  对齐实体:\n"
                for entity_align in alignment.aligned_entities:
                    summary += f"    - {entity_align.entity_text} ({entity_align.entity_type}, {entity_align.confidence:.2f})\n"

            if alignment.aligned_relations:
                summary += "  对齐关系:\n"
                for relation_align in alignment.aligned_relations:
                    summary += f"    - {relation_align.entity_text} ({relation_align.confidence:.2f})\n"

        return summary

if __name__ == "__main__":
    # 测试代码
    from ..knowledge_graph.graph_builder import KnowledgeGraphBuilder
    from ..gnn_reasoning.reasoning_engine import ReasoningStep

    # 创建测试知识图谱
    builder = KnowledgeGraphBuilder()
    test_documents = [
        "Albert Einstein worked at Princeton University and developed the theory of relativity.",
        "E=mc² is the famous equation discovered by Einstein.",
        "Princeton University is located in New Jersey."
    ]

    kg = builder.build_graph_from_documents(test_documents)

    # 创建对齐器
    aligner = MultimodalAligner(kg)

    # 创建测试推理步骤
    reasoning_step = ReasoningStep(
        step_id=0,
        entities=[],
        relations=['works_at'],
        confidence=0.8,
        explanation="Einstein worked at Princeton University"
    )

    # 测试对齐
    reasoning_text = "Einstein was a theoretical physicist who worked at Princeton University"
    alignment = aligner.align_reasoning_step(reasoning_step, reasoning_text)

    print("对齐结果:")
    print(f"整体置信度: {alignment.overall_confidence:.2f}")
    print("对齐实体:")
    for entity_align in alignment.aligned_entities:
        print(f"  - {entity_align.entity_text} ({entity_align.entity_type}, {entity_align.confidence:.2f})")

    print("\n对齐测试完成！")