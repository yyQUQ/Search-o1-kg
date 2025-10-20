"""
知识图谱存储模块

支持多种存储后端：
- JSON文件存储（简单、易用）
- NetworkX图格式（便于图算法）
- 可扩展支持数据库存储（Neo4j、MongoDB等）
"""

import json
import pickle
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import networkx as nx
from datetime import datetime

from .graph_builder import KnowledgeGraph, Entity, Relation

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphStorage:
    """知识图谱存储基类"""

    def __init__(self, storage_path: str = "./cache"):
        """
        初始化存储

        Args:
            storage_path: 存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)

    def save_graph(self, kg: KnowledgeGraph, filepath: str) -> None:
        """保存知识图谱"""
        raise NotImplementedError

    def load_graph(self, filepath: str) -> KnowledgeGraph:
        """加载知识图谱"""
        raise NotImplementedError

    def delete_graph(self, filepath: str) -> bool:
        """删除知识图谱文件"""
        try:
            file_path = self.storage_path / filepath
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"删除图谱文件失败: {e}")
            return False

    def list_graphs(self) -> List[str]:
        """列出所有存储的图谱文件"""
        try:
            return [f.name for f in self.storage_path.iterdir() if f.is_file()]
        except Exception as e:
            logger.error(f"列出图谱文件失败: {e}")
            return []

class JSONGraphStorage(GraphStorage):
    """JSON格式的图谱存储"""

    def save_graph(self, kg: KnowledgeGraph, filepath: str) -> None:
        """
        保存知识图谱为JSON格式

        Args:
            kg: 知识图谱对象
            filepath: 保存路径（相对于storage_path）
        """
        try:
            # 序列化图谱数据
            graph_data = {
                'entities': {
                    entity_id: {
                        'text': entity.text,
                        'entity_type': entity.entity_type,
                        'start_pos': entity.start_pos,
                        'end_pos': entity.end_pos,
                        'confidence': entity.confidence,
                        'context': entity.context,
                        'properties': entity.properties or {}
                    }
                    for entity_id, entity in kg.entities.items()
                },
                'relations': [
                    {
                        'subject': relation.subject,
                        'predicate': relation.predicate,
                        'object': relation.object,
                        'confidence': relation.confidence,
                        'source_text': relation.source_text,
                        'subject_type': relation.subject_type,
                        'object_type': relation.object_type,
                        'properties': relation.properties or {}
                    }
                    for relation in kg.relations
                ],
                'entity_index': kg.entity_index,
                'graph_data': {
                    'nodes': [
                        {
                            'id': node_id,
                            **kg.graph.nodes[node_id]
                        }
                        for node_id in kg.graph.nodes()
                    ],
                    'edges': [
                        {
                            'source': edge[0],
                            'target': edge[1],
                            **kg.graph.edges[edge]
                        }
                        for edge in kg.graph.edges()
                    ]
                },
                'metadata': kg.metadata,
                'created_at': kg.created_at.isoformat() if kg.created_at else None,
                'updated_at': kg.updated_at.isoformat() if kg.updated_at else None
            }

            # 保存到文件
            file_path = self.storage_path / filepath
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)

            logger.info(f"知识图谱已保存到: {file_path}")

        except Exception as e:
            logger.error(f"保存知识图谱失败: {e}")
            raise

    def load_graph(self, filepath: str) -> KnowledgeGraph:
        """
        从JSON文件加载知识图谱

        Args:
            filepath: 文件路径（相对于storage_path）

        Returns:
            知识图谱对象
        """
        try:
            file_path = self.storage_path / filepath
            with open(file_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)

            # 重建实体
            entities = {}
            for entity_id, entity_data in graph_data['entities'].items():
                entity = Entity(
                    text=entity_data['text'],
                    entity_type=entity_data['entity_type'],
                    start_pos=entity_data['start_pos'],
                    end_pos=entity_data['end_pos'],
                    confidence=entity_data['confidence'],
                    context=entity_data['context'],
                    properties=entity_data.get('properties', {})
                )
                entities[entity_id] = entity

            # 重建关系
            relations = []
            for relation_data in graph_data['relations']:
                relation = Relation(
                    subject=relation_data['subject'],
                    predicate=relation_data['predicate'],
                    object=relation_data['object'],
                    confidence=relation_data['confidence'],
                    source_text=relation_data['source_text'],
                    subject_type=relation_data['subject_type'],
                    object_type=relation_data['object_type'],
                    properties=relation_data.get('properties', {})
                )
                relations.append(relation)

            # 重建图结构
            graph = nx.DiGraph()

            # 添加节点
            for node_data in graph_data['graph_data']['nodes']:
                node_id = node_data.pop('id')
                graph.add_node(node_id, **node_data)

            # 添加边
            for edge_data in graph_data['graph_data']['edges']:
                source = edge_data.pop('source')
                target = edge_data.pop('target')
                graph.add_edge(source, target, **edge_data)

            # 重建元数据
            metadata = graph_data['metadata']
            created_at = None
            updated_at = None

            if graph_data.get('created_at'):
                created_at = datetime.fromisoformat(graph_data['created_at'])
            if graph_data.get('updated_at'):
                updated_at = datetime.fromisoformat(graph_data['updated_at'])

            kg = KnowledgeGraph(
                entities=entities,
                relations=relations,
                entity_index=graph_data['entity_index'],
                graph=graph,
                metadata=metadata,
                created_at=created_at,
                updated_at=updated_at
            )

            logger.info(f"知识图谱已从 {file_path} 加载")
            return kg

        except Exception as e:
            logger.error(f"加载知识图谱失败: {e}")
            raise

class NetworkXGraphStorage(GraphStorage):
    """NetworkX格式的图谱存储"""

    def save_graph(self, kg: KnowledgeGraph, filepath: str) -> None:
        """
        保存知识图谱为NetworkX格式

        Args:
            kg: 知识图谱对象
            filepath: 保存路径（相对于storage_path）
        """
        try:
            file_path = self.storage_path / filepath

            # 保存NetworkX图
            nx_graph_path = file_path.with_suffix('.graphml')
            nx.write_graphml(kg.graph, nx_graph_path)

            # 保存实体和关系数据
            data_path = file_path.with_suffix('.json')
            data = {
                'entities': {
                    entity_id: {
                        'text': entity.text,
                        'entity_type': entity.entity_type,
                        'start_pos': entity.start_pos,
                        'end_pos': entity.end_pos,
                        'confidence': entity.confidence,
                        'context': entity.context,
                        'properties': entity.properties or {}
                    }
                    for entity_id, entity in kg.entities.items()
                },
                'relations': [
                    {
                        'subject': relation.subject,
                        'predicate': relation.predicate,
                        'object': relation.object,
                        'confidence': relation.confidence,
                        'source_text': relation.source_text,
                        'subject_type': relation.subject_type,
                        'object_type': relation.object_type,
                        'properties': relation.properties or {}
                    }
                    for relation in kg.relations
                ],
                'entity_index': kg.entity_index,
                'metadata': kg.metadata,
                'created_at': kg.created_at.isoformat() if kg.created_at else None,
                'updated_at': kg.updated_at.isoformat() if kg.updated_at else None
            }

            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"知识图谱已保存到: {nx_graph_path} 和 {data_path}")

        except Exception as e:
            logger.error(f"保存知识图谱失败: {e}")
            raise

    def load_graph(self, filepath: str) -> KnowledgeGraph:
        """
        从NetworkX格式加载知识图谱

        Args:
            filepath: 文件路径（相对于storage_path）

        Returns:
            知识图谱对象
        """
        try:
            file_path = self.storage_path / filepath

            # 加载NetworkX图
            nx_graph_path = file_path.with_suffix('.graphml')
            graph = nx.read_graphml(nx_graph_path)

            # 加载数据
            data_path = file_path.with_suffix('.json')
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 重建实体
            entities = {}
            for entity_id, entity_data in data['entities'].items():
                entity = Entity(
                    text=entity_data['text'],
                    entity_type=entity_data['entity_type'],
                    start_pos=entity_data['start_pos'],
                    end_pos=entity_data['end_pos'],
                    confidence=entity_data['confidence'],
                    context=entity_data['context'],
                    properties=entity_data.get('properties', {})
                )
                entities[entity_id] = entity

            # 重建关系
            relations = []
            for relation_data in data['relations']:
                relation = Relation(
                    subject=relation_data['subject'],
                    predicate=relation_data['predicate'],
                    object=relation_data['object'],
                    confidence=relation_data['confidence'],
                    source_text=relation_data['source_text'],
                    subject_type=relation_data['subject_type'],
                    object_type=relation_data['object_type'],
                    properties=relation_data.get('properties', {})
                )
                relations.append(relation)

            # 重建元数据
            metadata = data['metadata']
            created_at = None
            updated_at = None

            if data.get('created_at'):
                created_at = datetime.fromisoformat(data['created_at'])
            if data.get('updated_at'):
                updated_at = datetime.fromisoformat(data['updated_at'])

            kg = KnowledgeGraph(
                entities=entities,
                relations=relations,
                entity_index=data['entity_index'],
                graph=graph,
                metadata=metadata,
                created_at=created_at,
                updated_at=updated_at
            )

            logger.info(f"知识图谱已从 {file_path} 加载")
            return kg

        except Exception as e:
            logger.error(f"加载知识图谱失败: {e}")
            raise

class PickleGraphStorage(GraphStorage):
    """Pickle格式的图谱存储（快速序列化）"""

    def save_graph(self, kg: KnowledgeGraph, filepath: str) -> None:
        """
        使用pickle保存知识图谱

        Args:
            kg: 知识图谱对象
            filepath: 保存路径（相对于storage_path）
        """
        try:
            file_path = self.storage_path / filepath
            with open(file_path, 'wb') as f:
                pickle.dump(kg, f)

            logger.info(f"知识图谱已保存到: {file_path}")

        except Exception as e:
            logger.error(f"保存知识图谱失败: {e}")
            raise

    def load_graph(self, filepath: str) -> KnowledgeGraph:
        """
        从pickle文件加载知识图谱

        Args:
            filepath: 文件路径（相对于storage_path）

        Returns:
            知识图谱对象
        """
        try:
            file_path = self.storage_path / filepath
            with open(file_path, 'rb') as f:
                kg = pickle.load(f)

            logger.info(f"知识图谱已从 {file_path} 加载")
            return kg

        except Exception as e:
            logger.error(f"加载知识图谱失败: {e}")
            raise

def get_storage_backend(backend_type: str = 'json', storage_path: str = "./cache") -> GraphStorage:
    """
    获取存储后端

    Args:
        backend_type: 存储类型 ('json', 'networkx', 'pickle')
        storage_path: 存储路径

    Returns:
        存储后端实例
    """
    backends = {
        'json': JSONGraphStorage,
        'networkx': NetworkXGraphStorage,
        'pickle': PickleGraphStorage
    }

    if backend_type not in backends:
        raise ValueError(f"不支持的存储类型: {backend_type}. 支持的类型: {list(backends.keys())}")

    return backends[backend_type](storage_path)

if __name__ == "__main__":
    # 测试代码
    from .graph_builder import KnowledgeGraphBuilder

    # 创建测试数据
    builder = KnowledgeGraphBuilder()
    test_documents = [
        "Albert Einstein developed the theory of relativity and worked at Princeton University.",
        "The equation E=mc² describes mass-energy equivalence."
    ]

    # 构建知识图谱
    kg = builder.build_graph_from_documents(test_documents)

    # 测试不同的存储后端
    storage_types = ['json', 'networkx', 'pickle']

    for storage_type in storage_types:
        print(f"\n测试 {storage_type} 存储:")
        storage = get_storage_backend(storage_type)

        # 保存
        filename = f"test_kg.{storage_type}"
        storage.save_graph(kg, filename)

        # 加载
        loaded_kg = storage.load_graph(filename)

        # 验证
        print(f"  原图谱实体数: {len(kg.entities)}")
        print(f"  加载图谱实体数: {len(loaded_kg.entities)}")
        print(f"  数据一致性: {len(kg.entities) == len(loaded_kg.entities)}")

    print("\n所有存储测试完成！")