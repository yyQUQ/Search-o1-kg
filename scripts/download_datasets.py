#!/usr/bin/env python3
"""
数据集下载脚本

支持下载和处理多种数据集，自动转换为Search-o1-KG所需格式。
"""

import os
import json
import requests
from datasets import load_dataset
from typing import Dict, List, Optional
import argparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """数据集下载器"""

    def __init__(self, base_dir: str = "./data"):
        self.base_dir = base_dir
        self.ensure_directories()

    def ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            "data/GPQA",
            "data/MATH500",
            "data/QA_Datasets",
            "data/LIVECODEBENCH",
            "cache"
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"确保目录存在: {directory}")

    def download_gpqa(self, split: str = "diamond") -> str:
        """下载GPQA数据集"""
        logger.info(f"下载GPQA数据集 - {split}")

        try:
            dataset = load_dataset("Idavidrein/gpqa", split)

            # 转换格式
            processed_data = []
            for item in dataset:
                processed_item = {
                    "Question": item["Question"],
                    "Correct Choice": item.get("Correct Choice", ""),
                    "A": item.get("A", ""),
                    "B": item.get("B", ""),
                    "C": item.get("C", ""),
                    "D": item.get("D", "")
                }
                processed_data.append(processed_item)

            # 保存文件
            output_path = f"data/GPQA/{split}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)

            logger.info(f"GPQA {split} 数据集已保存到: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"下载GPQA数据集失败: {e}")
            return None

    def download_math500(self) -> str:
        """下载MATH500数据集"""
        logger.info("下载MATH500数据集")

        try:
            dataset = load_dataset("allenai/MathHub", "MATH500", split="test")

            # 转换格式
            processed_data = []
            for item in dataset:
                processed_item = {
                    "Question": item["problem"],
                    "Answer": item.get("solution", ""),
                    "Level": item.get("level", ""),
                    "Subject": item.get("subject", "")
                }
                processed_data.append(processed_item)

            # 保存文件
            output_path = "data/MATH500/test.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)

            logger.info(f"MATH500数据集已保存到: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"下载MATH500数据集失败: {e}")
            return None

    def download_nq(self, split: str = "validation") -> str:
        """下载Natural Questions数据集"""
        logger.info(f"下载Natural Questions数据集 - {split}")

        try:
            dataset = load_dataset("nq_open", split=split)

            # 转换格式
            processed_data = []
            for item in dataset:
                processed_item = {
                    "Question": item["question"],
                    "Answer": item.get("answer", "")
                }
                processed_data.append(processed_item)

            # 保存文件
            output_path = f"data/QA_Datasets/nq.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Natural Questions数据集已保存到: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"下载Natural Questions数据集失败: {e}")
            return None

    def download_triviaqa(self, split: str = "validation") -> str:
        """下载TriviaQA数据集"""
        logger.info(f"下载TriviaQA数据集 - {split}")

        try:
            dataset = load_dataset("trivia_qa", "unfiltered", split=split)

            # 转换格式
            processed_data = []
            for item in dataset:
                processed_item = {
                    "Question": item["question"],
                    "Answer": item.get("answer", {}).get("value", "")
                }
                processed_data.append(processed_item)

            # 保存文件
            output_path = f"data/QA_Datasets/triviaqa.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)

            logger.info(f"TriviaQA数据集已保存到: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"下载TriviaQA数据集失败: {e}")
            return None

    def download_hotpotqa(self, split: str = "validation") -> str:
        """下载HotpotQA数据集"""
        logger.info(f"下载HotpotQA数据集 - {split}")

        try:
            dataset = load_dataset("hotpot_qa", "fullwiki", split=split)

            # 转换格式
            processed_data = []
            for item in dataset:
                processed_item = {
                    "Question": item["question"],
                    "Answer": item.get("answer", ""),
                    "Type": item.get("type", ""),
                    "Level": item.get("level", "")
                }
                processed_data.append(processed_item)

            # 保存文件
            output_path = f"data/QA_Datasets/hotpotqa.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)

            logger.info(f"HotpotQA数据集已保存到: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"下载HotpotQA数据集失败: {e}")
            return None

    def create_sample_dataset(self, dataset_name: str, sample_size: int = 10):
        """创建样本数据集用于测试"""
        logger.info(f"创建 {dataset_name} 样本数据集")

        sample_data = []

        if dataset_name == "gpqa":
            sample_data = [
                {
                    "Question": "What is the molecular formula of glucose?",
                    "Correct Choice": "C",
                    "A": "C5H10O5",
                    "B": "C12H22O11",
                    "C": "C6H12O6",
                    "D": "C3H6O3"
                },
                {
                    "Question": "In quantum mechanics, what is the Heisenberg uncertainty principle?",
                    "Correct Choice": "A",
                    "A": "The inability to simultaneously measure certain pairs of physical properties",
                    "B": "The wave-particle duality of light",
                    "C": "The quantization of energy levels",
                    "D": "The spin of electrons"
                }
            ]
        elif dataset_name == "math500":
            sample_data = [
                {
                    "Question": "Solve for x: 2x + 5 = 15",
                    "Answer": "x = 5",
                    "Level": "1",
                    "Subject": "Algebra"
                },
                {
                    "Question": "What is the derivative of x^2?",
                    "Answer": "2x",
                    "Level": "1",
                    "Subject": "Calculus"
                }
            ]
        elif dataset_name == "nq":
            sample_data = [
                {
                    "Question": "Who was the first president of the United States?",
                    "Answer": "George Washington"
                },
                {
                    "Question": "What is the capital of France?",
                    "Answer": "Paris"
                }
            ]

        # 限制样本数量
        sample_data = sample_data[:sample_size]

        # 保存样本数据
        if dataset_name in ["gpqa"]:
            output_path = f"data/GPQA/diamond_sample.json"
        elif dataset_name in ["math500"]:
            output_path = "data/MATH500/test_sample.json"
        else:
            output_path = f"data/QA_Datasets/{dataset_name}_sample.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)

        logger.info(f"样本数据集已保存到: {output_path}")
        return output_path

def main():
    parser = argparse.ArgumentParser(description="下载Search-o1-KG所需的数据集")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["gpqa", "math500", "nq", "triviaqa", "hotpotqa", "sample"],
                        help="要下载的数据集名称")
    parser.add_argument("--split", type=str, default="validation",
                        help="数据集分割")
    parser.add_argument("--sample_size", type=int, default=10,
                        help="样本数据集大小")
    parser.add_argument("--base_dir", type=str, default="./data",
                        help="数据保存基础目录")

    args = parser.parse_args()

    downloader = DatasetDownloader(args.base_dir)

    if args.dataset == "gpqa":
        downloader.download_gpqa(args.split)
    elif args.dataset == "math500":
        downloader.download_math500()
    elif args.dataset == "nq":
        downloader.download_nq(args.split)
    elif args.dataset == "triviaqa":
        downloader.download_triviaqa(args.split)
    elif args.dataset == "hotpotqa":
        downloader.download_hotpotqa(args.split)
    elif args.dataset == "sample":
        # 创建多个样本数据集
        for dataset in ["gpqa", "math500", "nq"]:
            downloader.create_sample_dataset(dataset, args.sample_size)

if __name__ == "__main__":
    main()