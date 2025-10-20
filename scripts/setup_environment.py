#!/usr/bin/env python3
"""
环境配置脚本

自动检查和配置运行环境，包括依赖安装、模型下载、API密钥配置等。
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    """环境配置器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.required_packages = [
            "torch", "transformers", "vllm", "networkx", "numpy", "scikit-learn",
            "spacy", "beautifulsoup4", "requests", "tqdm", "matplotlib", "sympy"
        ]
        self.required_models = ["en_core_web_sm"]

    def check_python_version(self) -> bool:
        """检查Python版本"""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 9:
            logger.info(f"Python版本检查通过: {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            logger.error(f"Python版本过低: {version.major}.{version.minor}.{version.micro}，需要Python 3.9+")
            return False

    def check_package(self, package_name: str) -> bool:
        """检查包是否已安装"""
        try:
            __import__(package_name)
            logger.info(f"✓ {package_name} 已安装")
            return True
        except ImportError:
            logger.warning(f"✗ {package_name} 未安装")
            return False

    def install_package(self, package_name: str) -> bool:
        """安装包"""
        try:
            logger.info(f"正在安装 {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            logger.info(f"✓ {package_name} 安装成功")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {package_name} 安装失败: {e}")
            return False

    def check_spacy_model(self, model_name: str) -> bool:
        """检查spaCy模型"""
        try:
            import spacy
            spacy.load(model_name)
            logger.info(f"✓ spaCy模型 {model_name} 已安装")
            return True
        except OSError:
            logger.warning(f"✗ spaCy模型 {model_name} 未安装")
            return False

    def install_spacy_model(self, model_name: str) -> bool:
        """安装spaCy模型"""
        try:
            logger.info(f"正在安装spaCy模型 {model_name}...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            logger.info(f"✓ spaCy模型 {model_name} 安装成功")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ spaCy模型 {model_name} 安装失败: {e}")
            return False

    def create_directories(self) -> bool:
        """创建必要的目录"""
        directories = [
            "cache", "outputs", "data/GPQA", "data/MATH500", "data/QA_Datasets",
            "data/LIVECODEBENCH", "logs", "configs", "tests"
        ]

        success = True
        for directory in directories:
            dir_path = self.project_root / directory
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✓ 创建目录: {directory}")
            except Exception as e:
                logger.error(f"✗ 创建目录失败 {directory}: {e}")
                success = False

        return success

    def create_config_template(self) -> bool:
        """创建配置文件模板"""
        config_template = {
            "api_keys": {
                "bing_subscription_key": "your_bing_subscription_key_here",
                "jina_api_key": "your_jina_api_key_here",
                "huggingface_token": "your_huggingface_token_here"
            },
            "model_paths": {
                "default": "microsoft/DialoGPT-medium",
                "math": "microsoft/DialoGPT-medium",
                "qa": "microsoft/DialoGPT-medium"
            },
            "gnn_config": {
                "hidden_dim": 128,
                "output_dim": 64,
                "num_layers": 3,
                "dropout": 0.1
            },
            "inference_config": {
                "max_search_limit": 5,
                "max_turn": 10,
                "top_k": 10,
                "max_doc_len": 3000,
                "temperature": 0.7,
                "top_p": 0.8
            }
        }

        config_path = self.project_root / "configs" / "config.json"
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_template, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ 配置文件模板已创建: {config_path}")
            return True
        except Exception as e:
            logger.error(f"✗ 创建配置文件失败: {e}")
            return False

    def create_env_file(self) -> bool:
        """创建.env文件模板"""
        env_content = """# API Keys
BING_SUBSCRIPTION_KEY=your_bing_subscription_key_here
JINA_API_KEY=your_jina_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Model Paths
DEFAULT_MODEL_PATH=microsoft/DialoGPT-medium

# Cache Directory
CACHE_DIR=./cache

# Log Level
LOG_LEVEL=INFO
"""

        env_path = self.project_root / ".env"
        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(env_content)
            logger.info(f"✓ .env文件模板已创建: {env_path}")
            return True
        except Exception as e:
            logger.error(f"✗ 创建.env文件失败: {e}")
            return False

    def check_gpu_availability(self) -> Dict[str, any]:
        """检查GPU可用性"""
        gpu_info = {
            "available": False,
            "device_count": 0,
            "device_name": None,
            "memory_total": None
        }

        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["device_count"] = torch.cuda.device_count()
                gpu_info["device_name"] = torch.cuda.get_device_name(0)
                gpu_info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                logger.info(f"✓ GPU可用: {gpu_info['device_name']} ({gpu_info['memory_total']:.1f}GB)")
            else:
                logger.warning("✗ GPU不可用，将使用CPU")
        except ImportError:
            logger.warning("✗ PyTorch未安装，无法检查GPU")

        return gpu_info

    def run_quick_test(self) -> bool:
        """运行快速测试"""
        logger.info("运行快速环境测试...")

        try:
            # 测试导入核心模块
            sys.path.append(str(self.project_root / "src"))

            from knowledge_graph import EntityExtractor, KnowledgeGraphBuilder
            from gnn_reasoning import GraphEmbeddingConfig
            from multimodal_alignment import MultimodalAligner

            # 测试实体抽取
            extractor = EntityExtractor()
            test_text = "Einstein worked at Princeton University."
            entities = extractor.extract_entities(test_text)

            logger.info(f"✓ 实体抽取测试通过: 抽取到 {len(entities)} 个实体")

            # 测试知识图谱构建
            builder = KnowledgeGraphBuilder()
            kg = builder.build_graph_from_documents([test_text])

            logger.info(f"✓ 知识图谱构建测试通过: 构建了 {len(kg.entities)} 个实体")

            return True

        except Exception as e:
            logger.error(f"✗ 快速测试失败: {e}")
            return False

    def install_requirements(self) -> bool:
        """安装requirements.txt中的依赖"""
        requirements_path = self.project_root / "requirements.txt"
        if not requirements_path.exists():
            logger.error("requirements.txt文件不存在")
            return False

        try:
            logger.info("正在安装requirements.txt中的依赖...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
            logger.info("✓ requirements.txt依赖安装完成")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ requirements.txt依赖安装失败: {e}")
            return False

    def setup_complete_environment(self) -> bool:
        """完整环境配置"""
        logger.info("开始完整环境配置...")

        success = True

        # 1. 检查Python版本
        if not self.check_python_version():
            success = False

        # 2. 安装依赖
        if not self.install_requirements():
            success = False

        # 3. 检查包安装情况
        missing_packages = []
        for package in self.required_packages:
            if not self.check_package(package):
                missing_packages.append(package)

        # 4. 安装缺失的包
        for package in missing_packages:
            if not self.install_package(package):
                success = False

        # 5. 检查spaCy模型
        for model in self.required_models:
            if not self.check_spacy_model(model):
                if not self.install_spacy_model(model):
                    success = False

        # 6. 创建目录
        if not self.create_directories():
            success = False

        # 7. 创建配置文件
        if not self.create_config_template():
            success = False

        if not self.create_env_file():
            success = False

        # 8. 检查GPU
        gpu_info = self.check_gpu_availability()

        # 9. 运行快速测试
        if success:
            if not self.run_quick_test():
                logger.warning("快速测试失败，但环境配置基本完成")

        # 10. 生成配置报告
        self.generate_setup_report(success, gpu_info)

        return success

    def generate_setup_report(self, success: bool, gpu_info: Dict[str, any]):
        """生成配置报告"""
        report = {
            "setup_success": success,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "gpu_info": gpu_info,
            "project_root": str(self.project_root),
            "next_steps": [
                "1. 配置API密钥 (在.env文件中)",
                "2. 下载数据集 (使用scripts/download_datasets.py)",
                "3. 选择合适的模型路径",
                "4. 运行测试: python scripts/run_search_o1_kg.py --dataset_name sample --subset_num 1"
            ]
        }

        report_path = self.project_root / "logs" / "setup_report.json"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            # 打印报告
            print("\n" + "="*50)
            print("环境配置报告")
            print("="*50)
            print(f"配置状态: {'✓ 成功' if success else '✗ 失败'}")
            print(f"Python版本: {report['python_version']}")
            print(f"GPU可用: {'是' if gpu_info['available'] else '否'}")
            if gpu_info['available']:
                print(f"GPU信息: {gpu_info['device_name']} ({gpu_info['memory_total']:.1f}GB)")
            print(f"项目路径: {report['project_root']}")
            print("\n下一步操作:")
            for step in report['next_steps']:
                print(f"  {step}")
            print("="*50)

        except Exception as e:
            logger.error(f"生成配置报告失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="配置Search-o1-KG运行环境")
    parser.add_argument("--quick", action="store_true", help="快速检查环境")
    parser.add_argument("--install", action="store_true", help="完整安装和配置")

    args = parser.parse_args()

    setup = EnvironmentSetup()

    if args.quick:
        logger.info("快速环境检查...")
        setup.check_python_version()
        for package in setup.required_packages:
            setup.check_package(package)
        for model in setup.required_models:
            setup.check_spacy_model(model)
        setup.check_gpu_availability()
        setup.run_quick_test()

    elif args.install:
        setup.setup_complete_environment()

    else:
        print("请指定 --quick 或 --install 参数")
        print("  --quick: 快速检查环境")
        print("  --install: 完整安装和配置环境")

if __name__ == "__main__":
    main()