"""
关系抽取器

从文档中自动抽取实体间的关系，包括：
- 属性关系 (is-a, part-of, located-in等)
- 功能关系 (causes, enables, requires等)
- 数学关系 (equals, greater-than, function-of等)
- 化学关系 (reacts-with, produces, decomposes-to等)
"""

import re
import json
import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import spacy
from .entity_extractor import Entity

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Relation:
    """关系数据类"""
    subject: str
    predicate: str
    object: str
    confidence: float
    source_text: str
    subject_type: str
    object_type: str
    properties: Dict = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

class RelationExtractor:
    """关系抽取器类"""

    def __init__(self):
        """初始化关系抽取器"""
        # 尝试加载spaCy模型
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("加载spaCy模型: en_core_web_sm")
        except OSError:
            logger.warning("spaCy模型未找到，将使用规则方法")
            self.nlp = None

        # 定义关系模式
        self.relation_patterns = self._init_relation_patterns()

        # 定义触发词
        self.trigger_words = self._init_trigger_words()

    def _init_relation_patterns(self) -> Dict[str, List[re.Pattern]]:
        """初始化关系识别模式"""
        patterns = {
            'is_a': [
                re.compile(r'(\w+(?:\s+\w+)*)\s+is\s+(?:a|an)\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
                re.compile(r'(\w+(?:\s+\w+)*)\s+(?:is|are)\s+(?:a|an)\s+(?:type|kind|form)\s+of\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
            ],
            'part_of': [
                re.compile(r'(\w+(?:\s+\w+)*)\s+is\s+(?:a|an)?\s*part\s+of\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
                re.compile(r'(\w+(?:\s+\w+)*)\s+(?:belongs|belong)\s+to\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
            ],
            'located_in': [
                re.compile(r'(\w+(?:\s+\w+)*)\s+is\s+(?:located|situated)\s+in\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
                re.compile(r'(\w+(?:\s+\w+)*)\s+(?:in|at)\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
            ],
            'causes': [
                re.compile(r'(\w+(?:\s+\w+)*)\s+(?:causes|caused|lead\s+to|leads\s+to)\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
                re.compile(r'(\w+(?:\s+\w+)*)\s+(?:result\s+in|results\s+in)\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
            ],
            'enables': [
                re.compile(r'(\w+(?:\s+\w+)*)\s+(?:enables|enable|allows|allow)\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
                re.compile(r'(\w+(?:\s+\w+)*)\s+(?:makes|make)\s+(\w+(?:\s+\w+)*)\s+possible', re.IGNORECASE),
            ],
            'equals': [
                re.compile(r'(\w+(?:\s+\w+)*)\s*=\s*(\w+(?:\s+\w+)*)'),
                re.compile(r'(\w+(?:\s+\w+)*)\s+(?:equals|equal)\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
            ],
            'greater_than': [
                re.compile(r'(\w+(?:\s+\w+)*)\s*>\s*(\w+(?:\s+\w+)*)'),
                re.compile(r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?greater\s+than\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
            ],
            'works_at': [
                re.compile(r'(\w+(?:\s+\w+)*)\s+(?:works|work)\s+(?:at|in|for)\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
                re.compile(r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:a|an)?\s*(?:researcher|scientist|professor|employee)\s+at\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
            ],
            'reacts_with': [
                re.compile(r'(\w+(?:\s+\w+)*)\s+(?:reacts|react)\s+with\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
                re.compile(r'(\w+(?:\s+\w+)*)\s*\+\s*(\w+(?:\s+\w+)*)\s*→'),
            ],
            'produces': [
                re.compile(r'(\w+(?:\s+\w+)*)\s+(?:produces|produce|yields|yield)\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
                re.compile(r'(\w+(?:\s+\w+)*)\s*→\s*(\w+(?:\s+\w+)*)'),
            ],
        }
        return patterns

    def _init_trigger_words(self) -> Dict[str, List[str]]:
        """初始化关系触发词"""
        return {
            'is_a': ['is', 'are', 'represents', 'represents a', 'is a type of', 'is a kind of'],
            'part_of': ['part of', 'belongs to', 'component of', 'contains', 'includes'],
            'located_in': ['located in', 'situated in', 'found in', 'in', 'at'],
            'causes': ['causes', 'leads to', 'results in', 'triggers', 'induces'],
            'enables': ['enables', 'allows', 'makes possible', 'facilitates'],
            'equals': ['equals', 'equal to', '=', 'is equal to'],
            'greater_than': ['greater than', 'larger than', '>', 'exceeds'],
            'works_at': ['works at', 'works in', 'employed at', 'researcher at'],
            'reacts_with': ['reacts with', 'reacts to', 'combines with'],
            'produces': ['produces', 'yields', 'generates', 'creates', '→'],
        }

    def extract_relations_with_patterns(self, text: str, entities: List[Entity]) -> List[Relation]:
        """使用规则模式抽取关系"""
        relations = []

        # 创建实体位置映射
        entity_positions = {}
        for entity in entities:
            entity_positions[(entity.start_pos, entity.end_pos)] = entity

        # 基于模式抽取关系
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    try:
                        subject_text = match.group(1).strip()
                        object_text = match.group(2).strip()

                        # 查找对应的实体
                        subject_entity = self._find_matching_entity(subject_text, entities, text)
                        object_entity = self._find_matching_entity(object_text, entities, text)

                        if subject_entity and object_entity:
                            relation = Relation(
                                subject=subject_entity.text,
                                predicate=relation_type,
                                object=object_entity.text,
                                confidence=0.7,
                                source_text=match.group(0),
                                subject_type=subject_entity.entity_type,
                                object_type=object_entity.entity_type
                            )
                            relations.append(relation)

                    except IndexError:
                        continue
                    except Exception as e:
                        logger.warning(f"关系抽取错误: {e}")
                        continue

        return relations

    def extract_relations_with_spacy(self, text: str, entities: List[Entity]) -> List[Relation]:
        """使用spaCy抽取关系"""
        if not self.nlp:
            return []

        relations = []
        doc = self.nlp(text)

        # 抽取主谓宾关系
        for sent in doc.sents:
            # 查找主语和宾语
            subjects = []
            objects = []
            predicate = None

            for token in sent:
                if token.dep_ in ['nsubj', 'nsubjpass']:
                    subject_text = token.text
                    subject_entity = self._find_matching_entity(subject_text, entities, text)
                    if subject_entity:
                        subjects.append(subject_entity)

                elif token.dep_ in ['dobj', 'iobj', 'pobj']:
                    object_text = token.text
                    object_entity = self._find_matching_entity(object_text, entities, text)
                    if object_entity:
                        objects.append(object_entity)

                elif token.pos_ == 'VERB':
                    predicate = token.lemma_

            # 创建关系
            if subjects and objects and predicate:
                for subject in subjects:
                    for obj in objects:
                        relation_type = self._map_verb_to_relation(predicate)
                        relation = Relation(
                            subject=subject.text,
                            predicate=relation_type,
                            object=obj.text,
                            confidence=0.6,
                            source_text=sent.text,
                            subject_type=subject.entity_type,
                            object_type=obj.entity_type
                        )
                        relations.append(relation)

        return relations

    def _find_matching_entity(self, text: str, entities: List[Entity], full_text: str) -> Optional[Entity]:
        """查找匹配的实体"""
        text_lower = text.lower().strip()

        # 精确匹配
        for entity in entities:
            if entity.text.lower().strip() == text_lower:
                return entity

        # 包含匹配
        for entity in entities:
            if text_lower in entity.text.lower() or entity.text.lower() in text_lower:
                return entity

        # 在原文中查找位置
        start_pos = full_text.lower().find(text_lower)
        if start_pos != -1:
            end_pos = start_pos + len(text)
            for entity in entities:
                if (entity.start_pos <= start_pos <= entity.end_pos or
                    entity.start_pos <= end_pos <= entity.end_pos):
                    return entity

        return None

    def _map_verb_to_relation(self, verb: str) -> str:
        """将动词映射到关系类型"""
        verb_mapping = {
            'be': 'is_a',
            'have': 'has_property',
            'work': 'works_at',
            'cause': 'causes',
            'enable': 'enables',
            'produce': 'produces',
            'react': 'reacts_with',
            'contain': 'contains',
            'belong': 'part_of',
        }
        return verb_mapping.get(verb, 'related_to')

    def extract_mathematical_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """抽取数学关系"""
        relations = []
        math_entities = [e for e in entities if e.entity_type == 'MATH']

        # 查找数学关系模式
        math_patterns = [
            (r'([^=]+)\s*=\s*([^=]+)', 'equals'),
            (r'([^+]+)\s*\+\s*([^+]+)', 'addition'),
            (r'([^-]+)\s*-\s*([^-]+)', 'subtraction'),
            (r'([^*]+)\s*\*\s*([^*]+)', 'multiplication'),
            (r'([^/]+)\s*/\s*([^/]+)', 'division'),
            (r'([^>]+)\s*>\s*([^>]+)', 'greater_than'),
            (r'([^<]+)\s*<\s*([^<]+)', 'less_than'),
        ]

        for pattern, relation_type in math_patterns:
            for match in re.finditer(pattern, text):
                try:
                    left_expr = match.group(1).strip()
                    right_expr = match.group(2).strip()

                    left_entity = self._find_matching_entity(left_expr, math_entities, text)
                    right_entity = self._find_matching_entity(right_expr, math_entities, text)

                    if left_entity and right_entity:
                        relation = Relation(
                            subject=left_entity.text,
                            predicate=relation_type,
                            object=right_entity.text,
                            confidence=0.9,
                            source_text=match.group(0),
                            subject_type='MATH',
                            object_type='MATH'
                        )
                        relations.append(relation)

                except Exception as e:
                    logger.warning(f"数学关系抽取错误: {e}")
                    continue

        return relations

    def extract_chemical_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """抽取化学关系"""
        relations = []
        chem_entities = [e for e in entities if e.entity_type == 'CHEM']

        # 化学反应模式
        reaction_patterns = [
            (r'([^+]+)\s*\+\s*([^+]+)\s*→\s*([^→]+)', 'reacts_to_produce'),
            (r'([^→]+)\s*→\s*([^→]+)', 'produces'),
            (r'([^=]+)\s*=\s*([^=]+)\s*\+\s*([^=]+)', 'decomposes_to'),
        ]

        for pattern, relation_type in reaction_patterns:
            for match in re.finditer(pattern, text):
                try:
                    if relation_type == 'reacts_to_produce':
                        reactant1 = match.group(1).strip()
                        reactant2 = match.group(2).strip()
                        product = match.group(3).strip()

                        r1_entity = self._find_matching_entity(reactant1, chem_entities, text)
                        r2_entity = self._find_matching_entity(reactant2, chem_entities, text)
                        p_entity = self._find_matching_entity(product, chem_entities, text)

                        if r1_entity and r2_entity and p_entity:
                            # 反应物关系
                            relation1 = Relation(
                                subject=r1_entity.text,
                                predicate='reacts_with',
                                object=r2_entity.text,
                                confidence=0.8,
                                source_text=match.group(0),
                                subject_type='CHEM',
                                object_type='CHEM'
                            )
                            relations.append(relation1)

                            # 产物关系
                            relation2 = Relation(
                                subject=r1_entity.text,
                                predicate='produces',
                                object=p_entity.text,
                                confidence=0.8,
                                source_text=match.group(0),
                                subject_type='CHEM',
                                object_type='CHEM'
                            )
                            relations.append(relation2)

                    else:
                        left = match.group(1).strip()
                        right = match.group(2).strip()

                        left_entity = self._find_matching_entity(left, chem_entities, text)
                        right_entity = self._find_matching_entity(right, chem_entities, text)

                        if left_entity and right_entity:
                            relation = Relation(
                                subject=left_entity.text,
                                predicate=relation_type,
                                object=right_entity.text,
                                confidence=0.8,
                                source_text=match.group(0),
                                subject_type='CHEM',
                                object_type='CHEM'
                            )
                            relations.append(relation)

                except Exception as e:
                    logger.warning(f"化学关系抽取错误: {e}")
                    continue

        return relations

    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        综合抽取关系

        Args:
            text: 输入文本
            entities: 实体列表

        Returns:
            关系列表
        """
        all_relations = []

        # 基于模式抽取
        pattern_relations = self.extract_relations_with_patterns(text, entities)
        all_relations.extend(pattern_relations)

        # 使用spaCy抽取
        spacy_relations = self.extract_relations_with_spacy(text, entities)
        all_relations.extend(spacy_relations)

        # 抽取数学关系
        math_relations = self.extract_mathematical_relations(text, entities)
        all_relations.extend(math_relations)

        # 抽取化学关系
        chem_relations = self.extract_chemical_relations(text, entities)
        all_relations.extend(chem_relations)

        # 去重和过滤
        filtered_relations = self._filter_relations(all_relations)

        return filtered_relations

    def _filter_relations(self, relations: List[Relation]) -> List[Relation]:
        """过滤和去重关系"""
        filtered = []
        seen_relations = set()

        for relation in relations:
            relation_key = (
                relation.subject.lower().strip(),
                relation.predicate,
                relation.object.lower().strip()
            )

            # 去重
            if relation_key in seen_relations:
                continue

            # 过滤置信度过低的关系
            if relation.confidence < 0.5:
                continue

            # 过滤自反关系
            if relation.subject.lower().strip() == relation.object.lower().strip():
                continue

            seen_relations.add(relation_key)
            filtered.append(relation)

        return filtered

    def extract_relations_from_documents(self, documents: List[str], entities_dict: Dict[str, List[Entity]]) -> Dict[str, List[Relation]]:
        """
        从多个文档中批量抽取关系

        Args:
            documents: 文档列表
            entities_dict: 文档ID到实体列表的映射

        Returns:
            文档ID到关系列表的映射
        """
        results = {}

        for doc_id, doc in enumerate(documents):
            entities_key = f"doc_{doc_id}"
            entities = entities_dict.get(entities_key, [])

            logger.info(f"正在处理文档 {doc_id + 1}/{len(documents)}")
            relations = self.extract_relations(doc, entities)
            results[entities_key] = relations
            logger.info(f"文档 {doc_id} 抽取到 {len(relations)} 个关系")

        return results

if __name__ == "__main__":
    # 测试代码
    from .entity_extractor import EntityExtractor

    extractor = RelationExtractor()
    entity_extractor = EntityExtractor()

    test_text = """
    Albert Einstein worked at Princeton University. He developed the famous equation E=mc².
    Hydrogen reacts with oxygen to produce water: 2H₂ + O₂ → 2H₂O.
    The energy E equals mass times the speed of light squared.
    """

    entities = entity_extractor.extract_entities(test_text)
    relations = extractor.extract_relations(test_text, entities)

    print("抽取的关系:")
    for relation in relations:
        print(f"- {relation.subject} --{relation.predicate}--> {relation.object} (置信度: {relation.confidence:.2f})")