#!/usr/bin/env python3
"""
快速启动脚本

提供一键式测试和演示功能，帮助用户快速验证环境并体验项目功能。
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickStartDemo:
    """快速启动演示"""

    def __init__(self):
        self.project_root = project_root

    def create_sample_data(self) -> dict:
        """创建示例数据"""
        sample_data = {
            "Question": "Albert Einstein developed the theory of relativity while working at Princeton University. The famous equation E=mc² shows the relationship between mass and energy.",
            "ExpectedEntities": ["Albert Einstein", "Princeton University", "E=mc²", "theory of relativity"],
            "ExpectedRelations": ["worked at", "developed", "shows relationship"]
        }
        return sample_data

    def test_entity_extraction(self) -> bool:
        """测试实体抽取"""
        logger.info("测试实体抽取功能...")

        try:
            from knowledge_graph import EntityExtractor

            extractor = EntityExtractor()
            test_data = self.create_sample_data()

            entities = extractor.extract_entities(test_data["Question"])

            logger.info(f"✓ 抽取到 {len(entities)} 个实体:")
            for entity in entities:
                logger.info(f"  - {entity.text} ({entity.entity_type}, 置信度: {entity.confidence:.2f})")

            return len(entities) > 0

        except Exception as e:
            logger.error(f"✗ 实体抽取测试失败: {e}")
            return False

    def test_knowledge_graph_building(self) -> bool:
        """测试知识图谱构建"""
        logger.info("测试知识图谱构建功能...")

        try:
            from knowledge_graph import KnowledgeGraphBuilder

            builder = KnowledgeGraphBuilder()
            test_data = self.create_sample_data()

            kg = builder.build_graph_from_documents([test_data["Question"]])

            logger.info(f"✓ 构建知识图谱成功:")
            logger.info(f"  - 实体数量: {len(kg.entities)}")
            logger.info(f"  - 关系数量: {len(kg.relations)}")
            logger.info(f"  - 图节点数: {kg.graph.number_of_nodes()}")
            logger.info(f"  - 图边数: {kg.graph.number_of_edges()}")

            # 显示实体详情
            for entity_id, entity in kg.entities.items():
                logger.info(f"  实体: {entity.text} ({entity.entity_type})")

            # 显示关系详情
            for relation in kg.relations:
                logger.info(f"  关系: {relation.subject} --{relation.predicate}--> {relation.object}")

            return len(kg.entities) > 0

        except Exception as e:
            logger.error(f"✗ 知识图谱构建测试失败: {e}")
            return False

    def test_gnn_reasoning(self) -> bool:
        """测试图神经网络推理"""
        logger.info("测试图神经网络推理功能...")

        try:
            from knowledge_graph import KnowledgeGraphBuilder
            from gnn_reasoning import GraphReasoningEngine, GraphEmbeddingConfig

            # 构建测试图谱
            builder = KnowledgeGraphBuilder()
            test_data = self.create_sample_data()
            kg = builder.build_graph_from_documents([test_data["Question"]])

            # 创建推理引擎
            config = GraphEmbeddingConfig(
                hidden_dim=32,
                output_dim=16,
                num_layers=2,
                dropout=0.1
            )
            engine = GraphReasoningEngine(kg, config)

            # 测试实体链接
            question = "Where did Einstein work?"
            linked_entities = engine.link_entities(question, top_k=3)

            logger.info(f"✓ 链接到 {len(linked_entities)} 个实体:")
            for entity_id, confidence in linked_entities:
                entity = kg.entities.get(entity_id)
                if entity:
                    logger.info(f"  - {entity.text} (置信度: {confidence:.2f})")

            return len(linked_entities) > 0

        except Exception as e:
            logger.error(f"✗ 图神经网络推理测试失败: {e}")
            return False

    def test_multimodal_alignment(self) -> bool:
        """测试跨模态对齐"""
        logger.info("测试跨模态对齐功能...")

        try:
            from knowledge_graph import KnowledgeGraphBuilder
            from gnn_reasoning import GraphReasoningEngine, GraphEmbeddingConfig, ReasoningStep
            from multimodal_alignment import MultimodalAligner

            # 构建测试图谱
            builder = KnowledgeGraphBuilder()
            test_data = self.create_sample_data()
            kg = builder.build_graph_from_documents([test_data["Question"]])

            # 创建对齐器
            aligner = MultimodalAligner(kg)

            # 创建测试推理步骤
            reasoning_step = ReasoningStep(
                step_id=0,
                entities=[],
                relations=["worked_at"],
                confidence=0.8,
                explanation="Einstein worked at Princeton University"
            )

            # 测试对齐
            reasoning_text = "Einstein was a theoretical physicist who worked at Princeton University"
            alignment = aligner.align_reasoning_step(reasoning_step, reasoning_text)

            logger.info(f"✓ 对齐结果:")
            logger.info(f"  - 整体置信度: {alignment.overall_confidence:.2f}")
            logger.info(f"  - 对齐实体数量: {len(alignment.aligned_entities)}")
            logger.info(f"  - 对齐关系数量: {len(alignment.aligned_relations)}")

            for entity_align in alignment.aligned_entities:
                logger.info(f"    实体: {entity_align.entity_text} ({entity_align.entity_type}, {entity_align.confidence:.2f})")

            return alignment.overall_confidence > 0

        except Exception as e:
            logger.error(f"✗ 跨模态对齐测试失败: {e}")
            return False

    def run_full_demo(self) -> bool:
        """运行完整演示"""
        logger.info("开始完整功能演示...")

        test_results = {
            "entity_extraction": self.test_entity_extraction(),
            "knowledge_graph_building": self.test_knowledge_graph_building(),
            "gnn_reasoning": self.test_gnn_reasoning(),
            "multimodal_alignment": self.test_multimodal_alignment()
        }

        # 统计结果
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)

        logger.info(f"\n" + "="*50)
        logger.info("功能测试结果")
        logger.info("="*50)
        for test_name, result in test_results.items():
            status = "✓ 通过" if result else "✗ 失败"
            logger.info(f"{test_name}: {status}")

        logger.info(f"\n总体结果: {passed_tests}/{total_tests} 测试通过")

        if passed_tests == total_tests:
            logger.info("🎉 所有功能测试通过！环境配置成功！")
            self.show_next_steps()
        else:
            logger.warning("⚠️  部分功能测试失败，请检查环境配置")

        return passed_tests == total_tests

    def show_next_steps(self):
        """显示下一步操作"""
        logger.info("\n" + "="*50)
        logger.info("下一步操作指南")
        logger.info("="*50)
        logger.info("1. 配置API密钥:")
        logger.info("   - 编辑 .env 文件，添加Bing Search API密钥")
        logger.info("   - 可选: 添加Jina API密钥")
        logger.info("")
        logger.info("2. 下载完整数据集:")
        logger.info("   python scripts/download_datasets.py --dataset sample")
        logger.info("   python scripts/download_datasets.py --dataset gpqa")
        logger.info("")
        logger.info("3. 运行完整推理:")
        logger.info("   python scripts/run_search_o1_kg.py \\")
        logger.info("       --dataset_name gpqa \\")
        logger.info("       --split diamond_sample \\")
        logger.info("       --model_path microsoft/DialoGPT-medium \\")
        logger.info("       --bing_subscription_key YOUR_KEY \\")
        logger.info("       --subset_num 5")
        logger.info("")
        logger.info("4. 查看详细文档:")
        logger.info("   - 使用指南: USAGE_GUIDE.md")
        logger.info("   - 项目总结: PROJECT_SUMMARY.md")
        logger.info("   - README.md")

    def create_sample_dataset(self):
        """创建样本数据集"""
        logger.info("创建样本数据集...")

        sample_datasets = {
            "GPQA": [
                {
                    "Question": "What is the molecular formula of glucose?",
                    "Correct Choice": "C",
                    "A": "C5H10O5",
                    "B": "C12H22O11",
                    "C": "C6H12O6",
                    "D": "C3H6O3"
                },
                {
                    "Question": "In quantum mechanics, what principle states that certain pairs of physical properties cannot be simultaneously measured with arbitrary precision?",
                    "Correct Choice": "A",
                    "A": "Heisenberg uncertainty principle",
                    "B": "Pauli exclusion principle",
                    "C": "Schrödinger equation",
                    "D": "Planck's constant"
                }
            ],
            "MATH500": [
                {
                    "Question": "Solve for x: 2x + 5 = 15",
                    "Answer": "x = 5",
                    "Level": "1",
                    "Subject": "Algebra"
                },
                {
                    "Question": "What is the derivative of f(x) = x²?",
                    "Answer": "f'(x) = 2x",
                    "Level": "1",
                    "Subject": "Calculus"
                }
            ],
            "NQ": [
                {
                    "Question": "Who was the first president of the United States?",
                    "Answer": "George Washington"
                },
                {
                    "Question": "What is the capital of France?",
                    "Answer": "Paris"
                }
            ]
        }

        # 保存样本数据集
        data_dir = self.project_root / "data"
        data_dir.mkdir(exist_ok=True)

        for dataset_name, data in sample_datasets.items():
            if dataset_name == "GPQA":
                output_path = data_dir / "GPQA" / "diamond_sample.json"
            elif dataset_name == "MATH500":
                output_path = data_dir / "MATH500" / "test_sample.json"
            else:
                output_path = data_dir / "QA_Datasets" / f"{dataset_name.lower()}_sample.json"

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"✓ 样本数据集已创建: {output_path}")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Search-o1-KG 快速启动演示")
    parser.add_argument("--test", type=str, choices=["entity", "kg", "gnn", "alignment", "all"],
                        default="all", help="运行特定测试")
    parser.add_argument("--create-samples", action="store_true", help="创建样本数据集")

    args = parser.parse_args()

    demo = QuickStartDemo()

    if args.create_samples:
        demo.create_sample_dataset()
        return

    if args.test == "all":
        demo.run_full_demo()
    elif args.test == "entity":
        demo.test_entity_extraction()
    elif args.test == "kg":
        demo.test_knowledge_graph_building()
    elif args.test == "gnn":
        demo.test_gnn_reasoning()
    elif args.test == "alignment":
        demo.test_multimodal_alignment()

if __name__ == "__main__":
    main()