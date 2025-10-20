"""
图推理引擎

基于知识图谱和图神经网络实现复杂推理任务：
- 路径推理：在图谱中寻找推理路径
- 实体链接：链接问题中的实体到图谱
- 关系预测：预测实体间的关系
- 多跳推理：执行多步推理任务
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict, deque

from .graph_neural_network import GraphEmbeddingManager, GraphEmbeddingConfig
from ..knowledge_graph.graph_builder import KnowledgeGraph, Entity, Relation

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: int
    entities: List[str]
    relations: List[str]
    confidence: float
    explanation: str
    evidence: List[str] = None

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []

@dataclass
class ReasoningPath:
    """推理路径"""
    path_id: str
    question: str
    steps: List[ReasoningStep]
    final_answer: str
    confidence: float
    supporting_evidence: List[str] = None

    def __post_init__(self):
        if self.supporting_evidence is None:
            self.supporting_evidence = []

class GraphReasoningEngine:
    """图推理引擎"""

    def __init__(self, kg: KnowledgeGraph, config: GraphEmbeddingConfig = None):
        """
        初始化推理引擎

        Args:
            kg: 知识图谱
            config: 图嵌入配置
        """
        self.kg = kg
        self.config = config or GraphEmbeddingConfig()

        # 初始化图嵌入管理器
        self.embedding_manager = GraphEmbeddingManager(self.config)

        # 生成节点嵌入
        self.node_embeddings = self._generate_embeddings()

        # 初始化推理规则
        self.reasoning_rules = self._init_reasoning_rules()

        # 推理历史
        self.reasoning_history = []

    def _generate_embeddings(self) -> Dict[str, np.ndarray]:
        """生成节点嵌入"""
        logger.info("生成知识图谱节点嵌入...")

        # 基于实体类型生成初始特征
        node_features = self._generate_initial_features()

        # 使用GNN生成嵌入
        embeddings = self.embedding_manager.generate_node_embeddings(
            self.kg.graph,
            node_features,
            gnn_type='gcn'
        )

        logger.info(f"生成了 {len(embeddings)} 个节点嵌入")
        return embeddings

    def _generate_initial_features(self) -> Dict[str, np.ndarray]:
        """基于实体类型生成初始特征"""
        type_mapping = {
            'PERSON': [1, 0, 0, 0, 0],
            'ORG': [0, 1, 0, 0, 0],
            'LOC': [0, 0, 1, 0, 0],
            'MATH': [0, 0, 0, 1, 0],
            'CHEM': [0, 0, 0, 0, 1],
        }

        node_features = {}
        feature_dim = 32  # 默认特征维度

        for entity_id, entity in self.kg.entities.items():
            # 类型特征
            type_features = type_mapping.get(entity.entity_type, [0] * 5)

            # 扩展到目标维度
            initial_features = np.zeros(feature_dim)
            initial_features[:len(type_features)] = type_features

            # 添加一些随机噪声以增加多样性
            noise = np.random.randn(feature_dim - len(type_features)) * 0.1
            initial_features[len(type_features):] = noise

            node_features[entity_id] = initial_features

        return node_features

    def _init_reasoning_rules(self) -> Dict[str, List[List[str]]]:
        """初始化推理规则"""
        return {
            'transitivity': [
                ['is_a', 'is_a'],  # A is_a B, B is_a C => A is_a C
                ['part_of', 'part_of'],  # A part_of B, B part_of C => A part_of C
                ['located_in', 'located_in'],  # A located_in B, B located_in C => A located_in C
            ],
            'composition': [
                ['works_at', 'part_of'],  # A works_at B, B part_of C => A works_at C
                ['reacts_with', 'produces'],  # A reacts_with B, B produces C => A related_to C
            ],
            'causality': [
                ['causes', 'causes'],  # A causes B, B causes C => A causes C
                ['enables', 'causes'],  # A enables B, B causes C => A enables C
            ]
        }

    def link_entities(self, question: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        链接问题中的实体到知识图谱

        Args:
            question: 问题文本
            top_k: 返回top-k候选实体

        Returns:
            (实体ID, 置信度) 列表
        """
        # 简化实现：基于文本匹配
        question_lower = question.lower()
        candidates = []

        for entity_id, entity in self.kg.entities.items():
            entity_text = entity.text.lower()

            # 精确匹配
            if entity_text in question_lower:
                score = 1.0
                candidates.append((entity_id, score))
                continue

            # 部分匹配
            words = entity_text.split()
            match_count = sum(1 for word in words if word in question_lower)
            if match_count > 0:
                score = match_count / len(words)
                candidates.append((entity_id, score))

        # 按置信度排序
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:top_k]

    def find_reasoning_paths(self, start_entity: str, end_entity: str, max_length: int = 5) -> List[List[Tuple[str, str, str]]]:
        """
        寻找推理路径

        Args:
            start_entity: 起始实体ID
            end_entity: 目标实体ID
            max_length: 最大路径长度

        Returns:
            路径列表，每个路径是 (实体1, 关系, 实体2) 的列表
        """
        if start_entity not in self.kg.entities or end_entity not in self.kg.entities:
            return []

        # 使用BFS寻找路径
        paths = []
        queue = deque([(start_entity, [])])
        visited = set()

        while queue and len(paths) < 10:  # 限制搜索空间
            current_entity, current_path = queue.popleft()

            if current_entity in visited and current_entity != start_entity:
                continue

            visited.add(current_entity)

            # 找到目标实体
            if current_entity == end_entity and current_path:
                paths.append(current_path)
                continue

            # 路径长度限制
            if len(current_path) >= max_length:
                continue

            # 探索邻居
            for edge in self.kg.graph.edges(current_entity, data=True):
                neighbor = edge[1]
                if edge[0] != current_entity:
                    neighbor = edge[0]

                relation = edge[2].get('predicate', 'related_to')
                new_path = current_path + [(current_entity, relation, neighbor)]

                if neighbor not in visited or neighbor == end_entity:
                    queue.append((neighbor, new_path))

        return paths

    def apply_reasoning_rules(self, path: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """
        应用推理规则扩展路径

        Args:
            path: 原始路径

        Returns:
            扩展后的路径
        """
        extended_path = list(path)

        # 检查传递性规则
        for i in range(len(path) - 1):
            relation1 = path[i][1]
            relation2 = path[i + 1][1]

            for rule_type, rule_patterns in self.reasoning_rules.items():
                for pattern in rule_patterns:
                    if (relation1 == pattern[0] and relation2 == pattern[1]):
                        # 应用规则
                        inferred_relation = relation1  # 简化：保持第一个关系
                        logger.info(f"应用 {rule_type} 规则: {pattern} -> {inferred_relation}")
                        break

        return extended_path

    def predict_relation(self, entity1: str, entity2: str) -> List[Tuple[str, float]]:
        """
        预测两个实体间的关系

        Args:
            entity1: 实体1 ID
            entity2: 实体2 ID

        Returns:
            (关系, 置信度) 列表
        """
        if entity1 not in self.node_embeddings or entity2 not in self.node_embeddings:
            return []

        emb1 = self.node_embeddings[entity1]
        emb2 = self.node_embeddings[entity2]

        # 计算嵌入相似度
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        # 基于相似度生成关系预测
        candidate_relations = []

        # 检查已知关系
        for relation in self.kg.relations:
            if (relation.subject == self.kg.entities[entity1].text and
                relation.object == self.kg.entities[entity2].text):
                candidate_relations.append((relation.predicate, relation.confidence))
            elif (relation.subject == self.kg.entities[entity2].text and
                  relation.object == self.kg.entities[entity1].text):
                # 反向关系
                candidate_relations.append((f"reverse_{relation.predicate}", relation.confidence * 0.8))

        # 如果没有已知关系，基于相似度预测
        if not candidate_relations and similarity > 0.5:
            # 基于实体类型预测
            entity1_type = self.kg.entities[entity1].entity_type
            entity2_type = self.kg.entities[entity2].entity_type

            predicted_relations = self._predict_relations_by_type(entity1_type, entity2_type, similarity)
            candidate_relations.extend(predicted_relations)

        # 按置信度排序
        candidate_relations.sort(key=lambda x: x[1], reverse=True)

        return candidate_relations[:5]

    def _predict_relations_by_type(self, type1: str, type2: str, similarity: float) -> List[Tuple[str, float]]:
        """基于实体类型预测关系"""
        type_relations = {
            ('PERSON', 'ORG'): [('works_at', similarity)],
            ('PERSON', 'LOC'): [('located_in', similarity)],
            ('CHEM', 'CHEM'): [('reacts_with', similarity)],
            ('MATH', 'MATH'): [('equals', similarity), ('greater_than', similarity * 0.8)],
        }

        key = (type1, type2)
        reverse_key = (type2, type1)

        if key in type_relations:
            return type_relations[key]
        elif reverse_key in type_relations:
            return type_relations[reverse_key]
        else:
            return [('related_to', similarity * 0.5)]

    def reason_about_question(self, question: str, max_reasoning_steps: int = 10) -> ReasoningPath:
        """
        对问题进行推理

        Args:
            question: 问题文本
            max_reasoning_steps: 最大推理步骤数

        Returns:
            推理路径
        """
        logger.info(f"开始推理问题: {question}")

        # 第一步：链接实体
        linked_entities = self.link_entities(question, top_k=3)
        logger.info(f"链接到 {len(linked_entities)} 个实体")

        if len(linked_entities) < 2:
            return ReasoningPath(
                path_id="failed",
                question=question,
                steps=[],
                final_answer="无法找到足够的实体信息来回答问题",
                confidence=0.0
            )

        # 第二步：寻找推理路径
        start_entity = linked_entities[0][0]
        reasoning_steps = []
        current_entities = [start_entity]

        for step_id in range(max_reasoning_steps):
            if step_id == 0:
                # 第一步：从起始实体开始
                current_entity = start_entity
                explanation = f"从实体 '{self.kg.entities[current_entity].text}' 开始推理"
            else:
                # 选择下一个实体
                current_entity = self._select_next_entity(current_entities, linked_entities, reasoning_steps)
                explanation = f"继续推理到实体 '{self.kg.entities[current_entity].text}'"

            if not current_entity:
                break

            # 查找相关关系
            relations = self._find_relevant_relations(current_entity, current_entities, question)

            if relations:
                step = ReasoningStep(
                    step_id=step_id,
                    entities=[current_entity],
                    relations=relations,
                    confidence=0.8,
                    explanation=explanation,
                    evidence=[r.source_text for r in relations if hasattr(r, 'source_text')]
                )
                reasoning_steps.append(step)
                current_entities.append(current_entity)

            # 检查是否可以回答问题
            answer, confidence = self._generate_answer(reasoning_steps, question)
            if confidence > 0.6:
                break

        # 生成最终答案
        if not reasoning_steps:
            final_answer = "无法找到有效的推理路径"
            confidence = 0.0
        else:
            final_answer, confidence = self._generate_answer(reasoning_steps, question)

        reasoning_path = ReasoningPath(
            path_id=f"reasoning_{len(self.reasoning_history)}",
            question=question,
            steps=reasoning_steps,
            final_answer=final_answer,
            confidence=confidence,
            supporting_evidence=[evidence for step in reasoning_steps for evidence in step.evidence]
        )

        # 记录推理历史
        self.reasoning_history.append(reasoning_path)

        logger.info(f"推理完成，置信度: {confidence:.2f}")
        return reasoning_path

    def _select_next_entity(self, current_entities: List[str], linked_entities: List[Tuple[str, float]], reasoning_steps: List[ReasoningStep]) -> Optional[str]:
        """选择下一个推理实体"""
        # 优先选择已链接但未访问的实体
        visited_entities = set(step.entities[0] for step in reasoning_steps if step.entities)

        for entity_id, confidence in linked_entities:
            if entity_id not in visited_entities and entity_id not in current_entities:
                return entity_id

        # 如果都已访问，选择最相似的实体
        if current_entities:
            current_entity = current_entities[-1]
            similar_entities = self.embedding_manager.find_similar_nodes(
                current_entity, self.node_embeddings, top_k=5
            )
            for entity_id, similarity in similar_entities:
                if entity_id not in visited_entities:
                    return entity_id

        return None

    def _find_relevant_relations(self, entity: str, current_entities: List[str], question: str) -> List[str]:
        """查找相关关系"""
        relations = []

        # 查找与该实体相关的关系
        for relation in self.kg.relations:
            if (relation.subject == self.kg.entities[entity].text or
                relation.object == self.kg.entities[entity].text):
                relations.append(relation.predicate)

        return relations[:3]  # 限制关系数量

    def _generate_answer(self, reasoning_steps: List[ReasoningStep], question: str) -> Tuple[str, float]:
        """基于推理步骤生成答案"""
        if not reasoning_steps:
            return "无法生成答案", 0.0

        # 简化实现：基于最后一步生成答案
        last_step = reasoning_steps[-1]
        confidence = last_step.confidence

        # 基于问题类型生成答案模板
        question_lower = question.lower()
        if any(word in question_lower for word in ['what', 'who', 'where', 'when']):
            if last_step.entities:
                entity_id = last_step.entities[0]
                entity = self.kg.entities[entity_id]
                answer = f"根据推理，答案是 {entity.text} ({entity.entity_type})"
            else:
                answer = "无法确定具体答案"
        elif any(word in question_lower for word in ['how', 'why']):
            answer = f"基于以下推理步骤：{'; '.join(step.explanation for step in reasoning_steps)}"
        else:
            answer = f"根据知识图谱推理，置信度为 {confidence:.2f}"

        return answer, confidence

    def multi_hop_reasoning(self, query_entity: str, relation_chain: List[str], max_hops: int = 3) -> List[ReasoningPath]:
        """
        多跳推理

        Args:
            query_entity: 查询实体ID
            relation_chain: 关系链
            max_hops: 最大跳数

        Returns:
            推理路径列表
        """
        paths = []

        def recursive_search(current_entity: str, current_chain: List[str], hop_count: int, current_path: List[str]):
            if hop_count >= max_hops:
                return

            # 查找满足下一个关系的实体
            target_relation = relation_chain[hop_count] if hop_count < len(relation_chain) else None

            for neighbor in self.kg.graph.neighbors(current_entity):
                edge_data = self.kg.graph[current_entity][neighbor]
                predicate = edge_data.get('predicate', 'related_to')

                if target_relation and predicate != target_relation:
                    continue

                # 继续搜索
                new_path = current_path + [current_entity, predicate, neighbor]
                recursive_search(neighbor, relation_chain, hop_count + 1, new_path)

                # 如果到达目标，创建推理路径
                if hop_count == len(relation_chain) - 1:
                    reasoning_path = self._create_reasoning_path_from_path(new_path, query_entity)
                    paths.append(reasoning_path)

        # 开始搜索
        recursive_search(query_entity, relation_chain, 0, [])

        return paths

    def _create_reasoning_path_from_path(self, path: List[str], query_entity: str) -> ReasoningPath:
        """从路径创建推理路径"""
        steps = []

        for i in range(0, len(path), 3):
            if i + 2 < len(path):
                entity1, relation, entity2 = path[i], path[i+1], path[i+2]

                step = ReasoningStep(
                    step_id=len(steps),
                    entities=[entity1, entity2],
                    relations=[relation],
                    confidence=0.8,
                    explanation=f"{entity1} 通过 {relation} 关联到 {entity2}"
                )
                steps.append(step)

        final_answer = f"通过 {len(steps)} 步推理完成"
        confidence = 0.8

        return ReasoningPath(
            path_id=f"multi_hop_{len(self.reasoning_history)}",
            question=f"从 {query_entity} 开始的多跳推理",
            steps=steps,
            final_answer=final_answer,
            confidence=confidence
        )

if __name__ == "__main__":
    # 测试代码
    from ..knowledge_graph.graph_builder import KnowledgeGraphBuilder

    # 创建测试知识图谱
    builder = KnowledgeGraphBuilder()
    test_documents = [
        "Albert Einstein worked at Princeton University and developed the theory of relativity.",
        "E=mc² is the famous equation discovered by Einstein.",
        "Princeton University is located in New Jersey."
    ]

    kg = builder.build_graph_from_documents(test_documents)

    # 创建推理引擎
    config = GraphEmbeddingConfig(hidden_dim=32, output_dim=16, num_layers=2)
    engine = GraphReasoningEngine(kg, config)

    # 测试实体链接
    print("测试实体链接:")
    question = "Where did Einstein work?"
    linked_entities = engine.link_entities(question)
    print(f"问题: {question}")
    print(f"链接实体: {[(kg.entities[eid].text, conf) for eid, conf in linked_entities]}")

    # 测试推理
    print("\n测试推理:")
    reasoning_path = engine.reason_about_question(question)
    print(f"推理答案: {reasoning_path.final_answer}")
    print(f"推理置信度: {reasoning_path.confidence:.2f}")
    print("推理步骤:")
    for step in reasoning_path.steps:
        print(f"  {step.step_id}: {step.explanation}")

    print("\n推理测试完成！")