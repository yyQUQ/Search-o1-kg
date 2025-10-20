"""
实体抽取器

从文档中自动抽取实体，包括：
- 文本实体（人名、地名、机构名等）
- 数学公式和符号
- 化学结构式
- 技术术语和概念
"""

import re
import json
import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import sympy as sp
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """实体数据类"""
    text: str
    entity_type: str  # 'PERSON', 'ORG', 'LOC', 'MATH', 'CHEM', 'TECH', 'CONCEPT'
    start_pos: int
    end_pos: int
    confidence: float
    context: str
    properties: Dict = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

class EntityExtractor:
    """实体抽取器类"""

    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        """
        初始化实体抽取器

        Args:
            model_name: 预训练NER模型名称
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化预训练模型
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"加载NER模型: {model_name}")
        except Exception as e:
            logger.warning(f"无法加载NER模型 {model_name}: {e}")
            self.tokenizer = None
            self.model = None

        # 定义实体类型模式
        self.entity_patterns = self._init_entity_patterns()

        # 数学公式模式
        self.math_patterns = self._init_math_patterns()

        # 化学式模式
        self.chem_patterns = self._init_chem_patterns()

    def _init_entity_patterns(self) -> Dict[str, List[re.Pattern]]:
        """初始化实体识别模式"""
        patterns = {
            'PERSON': [
                re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),  # 简单英文名字
                re.compile(r'\b[A-Z]\. [A-Z][a-z]+\b'),     # 缩写名字
            ],
            'ORG': [
                re.compile(r'\b[A-Z][a-z]+ (University|College|Institute|Company|Corp|Ltd|Inc)\b'),
                re.compile(r'\b[A-Z][a-z]+ (Lab|Laboratory|Center|Centre)\b'),
            ],
            'LOC': [
                re.compile(r'\b[A-Z][a-z]+, [A-Z]{2}\b'),    # 城市, 州
                re.compile(r'\b[A-Z][a-z]+ (City|Country|Nation)\b'),
            ],
            'TECH': [
                re.compile(r'\b[A-Z]+[a-z]*[0-9]*\b'),        # 技术术语 (GPT-4, V8等)
                re.compile(r'\b\w+\.js\b|\b\w+\.py\b|\b\w+\.java\b'),  # 文件扩展名
            ],
            'CONCEPT': [
                re.compile(r'\b\w+(?:tion|ment|ism|ity|ness|ology|graphy)\b', re.IGNORECASE),
            ]
        }
        return patterns

    def _init_math_patterns(self) -> List[re.Pattern]:
        """初始化数学公式识别模式"""
        return [
            re.compile(r'\$[^$]+\$'),                        # LaTeX数学公式 $...$
            re.compile(r'\\\([^)]+\\\)'),                    # LaTeX数学公式 \(...\)
            re.compile(r'\\\[[^\]]+\\\]'),                  # LaTeX数学公式 \[...\]
            re.compile(r'\\\\begin\{.*?\}.*?\\\\end\{.*?\}'), # LaTeX数学环境
            re.compile(r'[a-zA-Z]\^\d+|[a-zA-Z]_\{[^}]+\}'),  # 上标下标
            re.compile(r'\\[a-zA-Z]+\{[^}]*\}'),            # LaTeX命令
            re.compile(r'∫|∑|∏|√|∞|α|β|γ|δ|θ|λ|μ|π|σ|τ|φ|χ|ψ|ω'), # 数学符号
        ]

    def _init_chem_patterns(self) -> List[re.Pattern]:
        """初始化化学式识别模式"""
        return [
            re.compile(r'\b[A-Z][a-z]?\d*(?:[+-]?\d*)?\b'),  # 简单化学式
            re.compile(r'\b(?:H|He|Li|Be|B|C|N|O|F|Ne|Na|Mg|Al|Si|P|S|Cl|Ar|K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|Xe|Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|Fr|Ra|Ac|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr|Rf|Db|Sg|Bh|Hs|Mt|Ds|Rg|Cn|Nh|Fl|Mc|Lv|Ts|Og)\d*\b'),
            re.compile(r'\([A-Z][a-z]?\d*\)+\d*'),            # 化学基团 (CH3)2
            re.compile(r'[←→↔⇌]'),                           # 化学反应箭头
        ]

    def extract_entities_with_model(self, text: str) -> List[Entity]:
        """使用预训练模型抽取实体"""
        if not self.model or not self.tokenizer:
            return []

        try:
            # 分词和预测
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            # 解析预测结果
            entities = []
            current_entity = None

            for token_id, (token, pred_id) in enumerate(zip(tokens, predictions)):
                if token.startswith('##'):
                    continue

                label = self.model.config.id2label[pred_id]

                if label.startswith('B-'):  # 实体开始
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        'text': token,
                        'entity_type': label[2:],
                        'start_pos': text.find(token),
                        'tokens': [token_id]
                    }
                elif label.startswith('I-') and current_entity:  # 实体继续
                    current_entity['text'] += token.replace('##', '')
                    current_entity['tokens'].append(token_id)
                else:  # 非实体
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None

            if current_entity:
                entities.append(current_entity)

            # 转换为Entity对象
            result = []
            for ent in entities:
                entity = Entity(
                    text=ent['text'],
                    entity_type=ent['entity_type'],
                    start_pos=ent.get('start_pos', 0),
                    end_pos=ent.get('start_pos', 0) + len(ent['text']),
                    confidence=0.8,  # 默认置信度
                    context=text[max(0, ent.get('start_pos', 0)-50):min(len(text), ent.get('start_pos', 0)+len(ent['text'])+50)]
                )
                result.append(entity)

            return result

        except Exception as e:
            logger.error(f"模型实体抽取失败: {e}")
            return []

    def extract_entities_with_patterns(self, text: str) -> List[Entity]:
        """使用规则模式抽取实体"""
        entities = []

        # 抽取一般实体
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity = Entity(
                        text=match.group(),
                        entity_type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.6,
                        context=text[max(0, match.start()-30):min(len(text), match.end()+30)]
                    )
                    entities.append(entity)

        # 抽取数学公式
        for pattern in self.math_patterns:
            for match in pattern.finditer(text):
                formula_text = match.group()
                # 验证是否为有效数学表达式
                if self._validate_math_expression(formula_text):
                    entity = Entity(
                        text=formula_text,
                        entity_type='MATH',
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.9,
                        context=text[max(0, match.start()-30):min(len(text), match.end()+30)],
                        properties={'formula_type': self._classify_math_expression(formula_text)}
                    )
                    entities.append(entity)

        # 抽取化学式
        for pattern in self.chem_patterns:
            for match in pattern.finditer(text):
                chem_text = match.group()
                if self._validate_chemical_formula(chem_text):
                    entity = Entity(
                        text=chem_text,
                        entity_type='CHEM',
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.8,
                        context=text[max(0, match.start()-30):min(len(text), match.end()+30)]
                    )
                    entities.append(entity)

        return entities

    def _validate_math_expression(self, text: str) -> bool:
        """验证数学表达式"""
        try:
            # 简单的数学表达式验证
            cleaned = text.replace('$', '').replace('\\(', '').replace('\\)', '')
            if len(cleaned) < 2:
                return False
            # 尝试用sympy解析
            sp.sympify(cleaned)
            return True
        except:
            # 如果不能解析，检查是否包含数学符号
            math_symbols = set('∫∑∏√∞αβγδθλμπστφχψω+-*/=<>≤≥≠∈∉∪∩⊂⊃∧∨¬→←↔')
            return bool(set(text) & math_symbols)

    def _classify_math_expression(self, text: str) -> str:
        """分类数学表达式类型"""
        if '∫' in text:
            return 'integral'
        elif '∑' in text or '∏' in text:
            return 'summation'
        elif '=' in text and any(op in text for op in '+-*/'):
            return 'equation'
        elif '√' in text:
            return 'radical'
        else:
            return 'expression'

    def _validate_chemical_formula(self, text: str) -> bool:
        """验证化学式"""
        # 简单的化学式验证
        if re.match(r'^[A-Z][a-z]?\d*$', text):
            return True  # 单个化学元素
        elif re.match(r'^[A-Z][a-z]?\d*[A-Z][a-z]?\d*$', text):
            return True  # 两个元素的化合物
        elif '(' in text and ')' in text:
            return True  # 包含基团的化学式
        return False

    def extract_entities(self, text: str) -> List[Entity]:
        """
        综合抽取实体

        Args:
            text: 输入文本

        Returns:
            抽取的实体列表
        """
        all_entities = []

        # 使用预训练模型抽取
        model_entities = self.extract_entities_with_model(text)
        all_entities.extend(model_entities)

        # 使用规则模式抽取
        pattern_entities = self.extract_entities_with_patterns(text)
        all_entities.extend(pattern_entities)

        # 去重和过滤
        filtered_entities = self._filter_entities(all_entities)

        # 按位置排序
        filtered_entities.sort(key=lambda x: x.start_pos)

        return filtered_entities

    def _filter_entities(self, entities: List[Entity]) -> List[Entity]:
        """过滤和去重实体"""
        filtered = []
        seen_entities = set()

        for entity in entities:
            entity_key = (entity.text.lower().strip(), entity.entity_type)

            # 去重
            if entity_key in seen_entities:
                continue

            # 过滤过短的实体
            if len(entity.text.strip()) < 2:
                continue

            # 过滤置信度过低的实体
            if entity.confidence < 0.5:
                continue

            seen_entities.add(entity_key)
            filtered.append(entity)

        return filtered

    def extract_entities_from_documents(self, documents: List[str]) -> Dict[str, List[Entity]]:
        """
        从多个文档中批量抽取实体

        Args:
            documents: 文档列表

        Returns:
            文档ID到实体列表的映射
        """
        results = {}

        for doc_id, doc in enumerate(documents):
            logger.info(f"正在处理文档 {doc_id + 1}/{len(documents)}")
            entities = self.extract_entities(doc)
            results[f"doc_{doc_id}"] = entities
            logger.info(f"文档 {doc_id} 抽取到 {len(entities)} 个实体")

        return results

if __name__ == "__main__":
    # 测试代码
    extractor = EntityExtractor()

    test_text = """
    Albert Einstein worked at Princeton University.
    The famous equation E=mc² describes mass-energy equivalence.
    The chemical formula H₂O represents water.
    PyTorch is a popular machine learning framework.
    """

    entities = extractor.extract_entities(test_text)

    print("抽取的实体:")
    for entity in entities:
        print(f"- {entity.text} ({entity.entity_type}, 置信度: {entity.confidence:.2f})")