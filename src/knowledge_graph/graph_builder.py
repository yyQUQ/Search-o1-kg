"""
知识图谱构建器

整合实体和关系抽取结果，构建动态知识图谱。
支持增量更新、图谱合并和存储功能。
"""

import json
import logging
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import networkx as nx
import numpy as np
from datetime import datetime

from .entity_extractor import Entity, EntityExtractor
from .relation_extractor import Relation, RelationExtractor
from .graph_storage import GraphStorage

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeGraph:
    """知识图谱数据类"""
    entities: Dict[str, Entity]
    relations: List[Relation]
    entity_index: Dict[str, str]  # 实体名称到ID的映射
    graph: nx.DiGraph  # NetworkX有向图
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    def __post_init__(self):
        if not hasattr(self, 'created_at') or self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()

class KnowledgeGraphBuilder:
    """知识图谱构建器"""

    def __init__(self, storage_backend: Optional[GraphStorage] = None):
        """
        初始化知识图谱构建器

        Args:
            storage_backend: 图谱存储后端
        """
        self.storage = storage_backend or GraphStorage()
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()

        # 图谱配置
        self.config = {
            'min_entity_confidence': 0.5,
            'min_relation_confidence': 0.5,
            'max_entities_per_document': 100,
            'max_relations_per_document': 200,
            'entity_similarity_threshold': 0.8,
            'relation_similarity_threshold': 0.7
        }

    def build_graph_from_documents(self, documents: List[str], doc_ids: List[str] = None) -> KnowledgeGraph:
        """
        从文档列表构建知识图谱

        Args:
            documents: 文档内容列表
            doc_ids: 文档ID列表（可选）

        Returns:
            构建的知识图谱
        """
        logger.info(f"开始从 {len(documents)} 个文档构建知识图谱")

        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]

        # 第一步：抽取实体
        logger.info("第一步：抽取实体")
        entities_dict = self.entity_extractor.extract_entities_from_documents(documents)

        # 第二步：抽取关系
        logger.info("第二步：抽取关系")
        relations_dict = self.relation_extractor.extract_relations_from_documents(documents, entities_dict)

        # 第三步：合并和去重实体
        logger.info("第三步：合并和去重实体")
        merged_entities = self._merge_entities(entities_dict)

        # 第四步：构建图谱
        logger.info("第四步：构建图谱结构")
        graph = self._build_graph_structure(merged_entities, relations_dict)

        # 第五步：创建知识图谱对象
        kg = KnowledgeGraph(
            entities=merged_entities,
            relations=self._flatten_relations(relations_dict),
            entity_index=self._build_entity_index(merged_entities),
            graph=graph,
            metadata={
                'num_documents': len(documents),
                'document_ids': doc_ids,
                'config': self.config,
                'extraction_stats': self._compute_extraction_stats(entities_dict, relations_dict)
            },
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        logger.info(f"知识图谱构建完成：{len(merged_entities)} 个实体，{len(kg.relations)} 个关系")
        return kg

    def _merge_entities(self, entities_dict: Dict[str, List[Entity]]) -> Dict[str, Entity]:
        """合并和去重实体"""
        merged = {}
        entity_clusters = defaultdict(list)  # 相似实体聚类

        # 按文本聚类相似实体
        for doc_id, entities in entities_dict.items():
            for entity in entities:
                entity_key = entity.text.lower().strip()
                entity_clusters[entity_key].append(entity)

        # 合并每个聚类中的实体
        for entity_text, entity_list in entity_clusters.items():
            if len(entity_list) == 1:
                # 单个实体，直接使用
                entity = entity_list[0]
                entity_id = self._generate_entity_id(entity)
                merged[entity_id] = entity
            else:
                # 多个相似实体，需要合并
                merged_entity = self._merge_similar_entities(entity_list)
                entity_id = self._generate_entity_id(merged_entity)
                merged[entity_id] = merged_entity

        return merged

    def _merge_similar_entities(self, entities: List[Entity]) -> Entity:
        """合并相似实体"""
        if not entities:
            return None

        # 选择置信度最高的实体作为基础
        base_entity = max(entities, key=lambda e: e.confidence)

        # 合并属性
        all_types = set(e.entity_type for e in entities)
        all_contexts = [e.context for e in entities if e.context]
        all_properties = {}

        for entity in entities:
            if entity.properties:
                all_properties.update(entity.properties)

        # 如果有多种类型，选择最常见的
        if len(all_types) > 1:
            type_counts = defaultdict(int)
            for entity_type in all_types:
                type_counts[entity_type] += sum(1 for e in entities if e.entity_type == entity_type)
            base_entity.entity_type = max(type_counts, key=type_counts.get)

        # 合并上下文（选择最长的）
        if all_contexts:
            base_entity.context = max(all_contexts, key=len)

        # 合并属性
        base_entity.properties = all_properties

        # 更新置信度（取平均值）
        base_entity.confidence = sum(e.confidence for e in entities) / len(entities)

        return base_entity

    def _build_graph_structure(self, entities: Dict[str, Entity], relations_dict: Dict[str, List[Relation]]) -> nx.DiGraph:
        """构建图结构"""
        graph = nx.DiGraph()

        # 添加实体节点
        for entity_id, entity in entities.items():
            graph.add_node(
                entity_id,
                text=entity.text,
                type=entity.entity_type,
                confidence=entity.confidence,
                context=entity.context,
                properties=entity.properties or {}
            )

        # 添加关系边
        for relations in relations_dict.values():
            for relation in relations:
                # 查找对应的实体ID
                subject_id = self._find_entity_id_by_text(relation.subject, entities)
                object_id = self._find_entity_id_by_text(relation.object, entities)

                if subject_id and object_id:
                    graph.add_edge(
                        subject_id,
                        object_id,
                        predicate=relation.predicate,
                        confidence=relation.confidence,
                        source_text=relation.source_text,
                        subject_type=relation.subject_type,
                        object_type=relation.object_type
                    )

        return graph

    def _find_entity_id_by_text(self, text: str, entities: Dict[str, Entity]) -> Optional[str]:
        """根据文本查找实体ID"""
        text_lower = text.lower().strip()

        for entity_id, entity in entities.items():
            if entity.text.lower().strip() == text_lower:
                return entity_id

        return None

    def _build_entity_index(self, entities: Dict[str, Entity]) -> Dict[str, str]:
        """构建实体名称到ID的索引"""
        index = {}
        for entity_id, entity in entities.items():
            # 多种索引方式
            index[entity.text.lower().strip()] = entity_id
            index[entity.text.strip()] = entity_id

            # 添加小写版本
            lower_text = entity.text.lower()
            if lower_text != entity.text:
                index[lower_text] = entity_id

        return index

    def _flatten_relations(self, relations_dict: Dict[str, List[Relation]]) -> List[Relation]:
        """扁平化关系字典"""
        all_relations = []
        for relations in relations_dict.values():
            all_relations.extend(relations)
        return all_relations

    def _generate_entity_id(self, entity: Entity) -> str:
        """生成实体ID"""
        # 基于实体文本和类型生成唯一ID
        import hashlib
        text_hash = hashlib.md5(f"{entity.text}_{entity.entity_type}".encode()).hexdigest()[:8]
        return f"{entity.entity_type.lower()}_{text_hash}"

    def _compute_extraction_stats(self, entities_dict: Dict[str, List[Relation]], relations_dict: Dict[str, List[Relation]]) -> Dict[str, Any]:
        """计算抽取统计信息"""
        total_entities = sum(len(entities) for entities in entities_dict.values())
        total_relations = sum(len(relations) for relations in relations_dict.values())

        entity_types = defaultdict(int)
        for entities in entities_dict.values():
            for entity in entities:
                entity_types[entity.entity_type] += 1

        relation_types = defaultdict(int)
        for relations in relations_dict.values():
            for relation in relations:
                relation_types[relation.predicate] += 1

        return {
            'total_entities': total_entities,
            'total_relations': total_relations,
            'entity_type_distribution': dict(entity_types),
            'relation_type_distribution': dict(relation_types),
            'documents_processed': len(entities_dict)
        }

    def update_graph(self, kg: KnowledgeGraph, new_documents: List[str], doc_ids: List[str] = None) -> KnowledgeGraph:
        """
        增量更新知识图谱

        Args:
            kg: 现有知识图谱
            new_documents: 新文档列表
            doc_ids: 新文档ID列表

        Returns:
            更新后的知识图谱
        """
        logger.info(f"开始增量更新知识图谱，新增 {len(new_documents)} 个文档")

        # 从新文档构建子图
        new_kg = self.build_graph_from_documents(new_documents, doc_ids)

        # 合并图谱
        merged_kg = self._merge_graphs(kg, new_kg)

        # 更新元数据
        merged_kg.metadata['num_documents'] += len(new_documents)
        merged_kg.metadata['document_ids'].extend(new_kg.metadata['document_ids'])
        merged_kg.updated_at = datetime.now()

        logger.info(f"知识图谱更新完成：{len(merged_kg.entities)} 个实体，{len(merged_kg.relations)} 个关系")
        return merged_kg

    def _merge_graphs(self, kg1: KnowledgeGraph, kg2: KnowledgeGraph) -> KnowledgeGraph:
        """合并两个知识图谱"""
        # 合并实体
        all_entities = dict(kg1.entities)

        for entity_id, entity in kg2.entities.items():
            if entity_id in all_entities:
                # 合并相似实体
                merged_entity = self._merge_similar_entities([all_entities[entity_id], entity])
                all_entities[entity_id] = merged_entity
            else:
                # 添加新实体
                new_entity_id = self._generate_entity_id(entity)
                all_entities[new_entity_id] = entity

        # 合并关系
        all_relations = list(kg1.relations) + list(kg2.relations)

        # 重建图结构
        merged_graph = self._build_graph_structure(all_entities, {'merged': all_relations})
        merged_entity_index = self._build_entity_index(all_entities)

        # 合并元数据
        merged_metadata = dict(kg1.metadata)
        merged_metadata.update(kg2.metadata)

        return KnowledgeGraph(
            entities=all_entities,
            relations=all_relations,
            entity_index=merged_entity_index,
            graph=merged_graph,
            metadata=merged_metadata,
            created_at=min(kg1.created_at, kg2.created_at),
            updated_at=datetime.now()
        )

    def save_graph(self, kg: KnowledgeGraph, filepath: str) -> None:
        """保存知识图谱到文件"""
        self.storage.save_graph(kg, filepath)

    def load_graph(self, filepath: str) -> KnowledgeGraph:
        """从文件加载知识图谱"""
        return self.storage.load_graph(filepath)

    def query_graph(self, kg: KnowledgeGraph, query: str, query_type: str = 'entity') -> List[Dict]:
        """
        查询知识图谱

        Args:
            kg: 知识图谱
            query: 查询字符串
            query_type: 查询类型 ('entity', 'relation', 'path')

        Returns:
            查询结果列表
        """
        if query_type == 'entity':
            return self._query_entities(kg, query)
        elif query_type == 'relation':
            return self._query_relations(kg, query)
        elif query_type == 'path':
            return self._query_paths(kg, query)
        else:
            raise ValueError(f"不支持的查询类型: {query_type}")

    def _query_entities(self, kg: KnowledgeGraph, query: str) -> List[Dict]:
        """查询实体"""
        results = []
        query_lower = query.lower().strip()

        for entity_id, entity in kg.entities.items():
            if (query_lower in entity.text.lower() or
                any(query_lower in str(v).lower() for v in entity.properties.values())):
                results.append({
                    'entity_id': entity_id,
                    'entity': entity,
                    'relations': [r for r in kg.relations if r.subject == entity.text or r.object == entity.text]
                })

        return results

    def _query_relations(self, kg: KnowledgeGraph, query: str) -> List[Dict]:
        """查询关系"""
        results = []
        query_lower = query.lower().strip()

        for relation in kg.relations:
            if (query_lower in relation.predicate.lower() or
                query_lower in relation.subject.lower() or
                query_lower in relation.object.lower()):
                results.append({
                    'relation': relation,
                    'confidence': relation.confidence,
                    'source_text': relation.source_text
                })

        return results

    def _query_paths(self, kg: KnowledgeGraph, query: str) -> List[Dict]:
        """查询路径"""
        # 简化实现：查找两个实体之间的路径
        entities = query.split('->')
        if len(entities) != 2:
            return []

        source = entities[0].strip()
        target = entities[1].strip()

        source_id = kg.entity_index.get(source.lower())
        target_id = kg.entity_index.get(target.lower())

        if not source_id or not target_id:
            return []

        try:
            paths = list(nx.all_simple_paths(kg.graph, source_id, target_id, cutoff=5))
            results = []

            for path in paths:
                path_info = []
                for i in range(len(path) - 1):
                    edge_data = kg.graph[path[i]][path[i + 1]]
                    path_info.append({
                        'from': kg.entities[path[i]].text,
                        'predicate': edge_data['predicate'],
                        'to': kg.entities[path[i + 1]].text,
                        'confidence': edge_data['confidence']
                    })

                results.append({'path': path_info, 'length': len(path)})

            return results

        except nx.NetworkXNoPath:
            return []

    def get_graph_statistics(self, kg: KnowledgeGraph) -> Dict[str, Any]:
        """获取图谱统计信息"""
        stats = {
            'num_entities': len(kg.entities),
            'num_relations': len(kg.relations),
            'num_nodes': kg.graph.number_of_nodes(),
            'num_edges': kg.graph.number_of_edges(),
            'density': nx.density(kg.graph),
            'is_connected': nx.is_weakly_connected(kg.graph),
        }

        # 实体类型分布
        entity_types = defaultdict(int)
        for entity in kg.entities.values():
            entity_types[entity.entity_type] += 1
        stats['entity_type_distribution'] = dict(entity_types)

        # 关系类型分布
        relation_types = defaultdict(int)
        for relation in kg.relations:
            relation_types[relation.predicate] += 1
        stats['relation_type_distribution'] = dict(relation_types)

        # 度分布
        degrees = dict(kg.graph.degree())
        if degrees:
            stats['degree_stats'] = {
                'mean': np.mean(list(degrees.values())),
                'median': np.median(list(degrees.values())),
                'max': max(degrees.values()),
                'min': min(degrees.values())
            }

        return stats

if __name__ == "__main__":
    # 测试代码
    builder = KnowledgeGraphBuilder()

    test_documents = [
        """
        Albert Einstein was a theoretical physicist who developed the theory of relativity.
        He worked at Princeton University and published the famous equation E=mc².
        The energy E equals mass times the speed of light squared.
        """,
        """
        Hydrogen is the simplest element with the chemical symbol H.
        Water has the chemical formula H₂O.
        Hydrogen reacts with oxygen to produce water: 2H₂ + O₂ → 2H₂O.
        """
    ]

    # 构建知识图谱
    kg = builder.build_graph_from_documents(test_documents)

    # 打印统计信息
    stats = builder.get_graph_statistics(kg)
    print("知识图谱统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 查询示例
    print("\n查询实体 'Einstein':")
    results = builder.query_graph(kg, "Einstein", "entity")
    for result in results:
        print(f"  实体: {result['entity'].text} ({result['entity'].entity_type})")

    print("\n查询关系 'equals':")
    results = builder.query_graph(kg, "equals", "relation")
    for result in results:
        print(f"  关系: {result['relation'].subject} --{result['relation'].predicate}--> {result['relation'].object}")