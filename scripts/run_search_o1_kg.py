"""
Search-o1-KG: 动态知识图谱增强的推理框架

基于Search-o1框架，集成动态知识图谱构建、图神经网络推理和跨模态对齐功能。
支持智能体RAG + 文档推理模块的增强版本。
"""

import os
import json
import time
import re
import argparse
import numpy as np
import torch
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm

# 导入原Search-o1模块
import sys
sys.path.append('/home/yy/projects/Search-o1-main/scripts')
from bing_search import bing_web_search, extract_relevant_info, fetch_page_content
from evaluate import run_evaluation, extract_answer
from prompts import get_webpage_to_reasonchain_instruction

# 导入我们的知识图谱模块
sys.path.append('/home/yy/projects/search-o1-kg/src')
from knowledge_graph import KnowledgeGraphBuilder, EntityExtractor, RelationExtractor
from gnn_reasoning import GraphReasoningEngine, GraphEmbeddingConfig
from multimodal_alignment import MultimodalAligner

# 导入transformers和vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 特殊标记
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
BEGIN_KG_REASONING = "<|begin_kg_reasoning|>"
END_KG_REASONING = "<|end_kg_reasoning|>"

class SearchO1KG:
    """Search-o1-KG主类"""

    def __init__(self, args):
        self.args = args

        # 设置路径
        self.cache_dir = './cache'
        self.output_dir = self._setup_output_dir()

        # 初始化缓存
        self.search_cache = {}
        self.url_cache = {}
        self.kg_cache = {}

        # 初始化模型
        self._initialize_models()

        # 初始化知识图谱组件
        self._initialize_kg_components()

        # 加载数据
        self._load_data()

    def _setup_output_dir(self) -> str:
        """设置输出目录"""
        if 'qwq' in self.args.model_path.lower():
            if self.args.dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode']:
                output_dir = f'./outputs/{self.args.dataset_name}.qwq.search_o1_kg'
            else:
                output_dir = f'./outputs/runs.qa/{self.args.dataset_name}.qwq.search_o1_kg'
        else:
            model_short_name = self.args.model_path.split('/')[-1].lower().replace('-instruct', '')
            output_dir = f'./outputs/runs.baselines/{self.args.dataset_name}.{model_short_name}.search_o1_kg'

        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _initialize_models(self):
        """初始化模型"""
        print("初始化语言模型...")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        # 加载LLM
        self.llm = LLM(
            model=self.args.model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.95,
        )

        print("模型初始化完成")

    def _initialize_kg_components(self):
        """初始化知识图谱组件"""
        print("初始化知识图谱组件...")

        # 知识图谱构建器
        self.kg_builder = KnowledgeGraphBuilder()

        # 图嵌入配置
        self.gnn_config = GraphEmbeddingConfig(
            hidden_dim=self.args.gnn_hidden_dim,
            output_dim=self.args.gnn_output_dim,
            num_layers=self.args.gnn_num_layers,
            dropout=self.args.gnn_dropout
        )

        print("知识图谱组件初始化完成")

    def _load_data(self):
        """加载数据集"""
        print(f"加载数据集: {self.args.dataset_name}")

        # 设置数据路径
        if self.args.dataset_name == 'livecode':
            self.data_path = f'./data/LiveCodeBench/{self.args.split}.json'
        elif self.args.dataset_name in ['math500', 'gpqa', 'aime', 'amc']:
            self.data_path = f'./data/{self.args.dataset_name.upper()}/{self.args.split}.json'
        else:
            self.data_path = f'./data/QA_Datasets/{self.args.dataset_name}.json'

        # 加载数据
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.filtered_data = json.load(f)

        print(f"加载了 {len(self.filtered_data)} 条数据")

    def _get_kg_enhanced_instruction(self, base_instruction: str) -> str:
        """获取知识图谱增强的指令"""
        kg_instruction = base_instruction + f"""

你有额外的知识图谱推理能力：
- 使用 {BEGIN_KG_REASONING} 开始知识图谱推理
- 使用 {END_KG_REASONING} 结束知识图谱推理
- 知识图谱可以帮助你发现实体间的关系和推理路径

示例：
{BEGIN_KG_REASONING}
从问题中识别关键实体：Einstein, Princeton University
在知识图谱中查找关系：Einstein --works_at--> Princeton University
基于图谱关系推理：Einstein worked at Princeton University
{END_KG_REASONING}

现在结合搜索和知识图谱来回答问题。
"""
        return kg_instruction

    def _build_knowledge_graph_from_documents(self, documents: List[str], search_queries: List[str]) -> str:
        """从文档构建知识图谱并生成推理结果"""
        if not documents:
            return "没有找到相关文档来构建知识图谱。"

        try:
            # 构建知识图谱
            kg = self.kg_builder.build_graph_from_documents(documents)

            # 创建推理引擎
            reasoning_engine = GraphReasoningEngine(kg, self.gnn_config)

            # 创建对齐器
            aligner = MultimodalAligner(kg)

            # 选择一个搜索查询进行推理
            if search_queries:
                query = search_queries[0]

                # 执行推理
                reasoning_path = reasoning_engine.reason_about_question(query)

                # 生成推理结果
                reasoning_result = f"""
知识图谱推理结果：
- 问题: {reasoning_path.question}
- 答案: {reasoning_path.final_answer}
- 置信度: {reasoning_path.confidence:.2f}

推理步骤：
"""
                for i, step in enumerate(reasoning_path.steps):
                    reasoning_result += f"\n步骤 {i+1}: {step.explanation}"
                    if step.entities:
                        entity_names = [kg.entities[eid].text for eid in step.entities if eid in kg.entities]
                        reasoning_result += f"\n  相关实体: {', '.join(entity_names)}"
                    if step.relations:
                        reasoning_result += f"\n  相关关系: {', '.join(step.relations)}"

                # 添加图谱统计信息
                stats = reasoning_engine.get_graph_statistics(kg)
                reasoning_result += f"\n\n知识图谱统计：
- 实体数量: {stats['num_entities']}
- 关系数量: {stats['num_relations']}
- 节点数量: {stats['num_nodes']}
- 边数量: {stats['num_edges']}"

                return reasoning_result
            else:
                return f"知识图谱构建完成，包含 {len(kg.entities)} 个实体和 {len(kg.relations)} 个关系。"

        except Exception as e:
            print(f"知识图谱构建失败: {e}")
            return "知识图谱构建过程中出现错误。"

    def _enhanced_webpage_to_reasonchain_batch(self, original_questions: List[str], prev_reasonings: List[str],
                                              search_queries: List[str], documents: List[str], dataset_name: str,
                                              batch_output_records: List[Dict], max_tokens: int = 32768) -> List[str]:
        """增强的网页到推理链批处理函数，集成知识图谱推理"""

        # 首先调用原始推理
        user_prompts = [
            get_webpage_to_reasonchain_instruction(r, sq, doc)
            for r, sq, doc in zip(prev_reasonings, search_queries, documents)
        ]

        prompts = [{"role": "user", "content": up} for up in user_prompts]
        prompts = [self.tokenizer.apply_chat_template([p], tokenize=False, add_generation_prompt=True) for p in prompts]

        # 添加知识图谱增强指令
        enhanced_prompts = []
        for prompt in prompts:
            enhanced_prompt = prompt.replace("Once you have all the information you need, continue your reasoning.",
                                            "Once you have all the information you need, continue your reasoning. " +
                                            f"You can also use {BEGIN_KG_REASONING} knowledge graph reasoning {END_KG_REASONING} if helpful.")
            enhanced_prompts.append(enhanced_prompt)

        # 生成基础推理结果
        output = self.llm.generate(
            enhanced_prompts,
            sampling_params=SamplingParams(
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.05,
            )
        )

        raw_outputs = [out.outputs[0].text for out in output]
        extracted_infos = [extract_answer(raw, mode='infogen') for raw in raw_outputs]

        # 为每个样本生成知识图谱增强
        enhanced_infos = []
        for i, (original_question, search_query, document, extracted_info) in enumerate(
            zip(original_questions, search_queries, documents, extracted_infos)):

            # 提取文档内容用于知识图谱构建
            doc_content = document.replace('**Web Page', '').replace('**:','').replace('\\n',' ').strip()
            doc_content = re.sub(r'\s+', ' ', doc_content)

            # 分割文档为段落
            doc_paragraphs = [p.strip() for p in doc_content.split('.') if p.strip() and len(p.strip()) > 20]

            # 构建知识图谱
            if len(doc_paragraphs) >= 2:
                kg_reasoning = self._build_knowledge_graph_from_documents(doc_paragraphs, [search_query])

                # 将知识图谱推理结果整合到原始信息中
                enhanced_info = extracted_info + f"\n\n{BEGIN_KG_REASONING}\n{kg_reasoning}\n{END_KG_REASONING}"
            else:
                enhanced_info = extracted_info

            enhanced_infos.append(enhanced_info)

        # 记录输出
        for i, (p, r, e) in enumerate(zip(prompts, raw_outputs, enhanced_infos)):
            batch_output_records.append({
                'prompt': p,
                'raw_output': r,
                'extracted_info': e,
                'kg_enhanced': True
            })

        return enhanced_infos

    def run_inference(self):
        """运行推理"""
        print(f"开始Search-o1-KG推理，数据集: {self.args.dataset_name}")

        # 调整参数
        MAX_SEARCH_LIMIT = self.args.max_search_limit
        MAX_TURN = self.args.max_turn
        top_k = self.args.top_k
        max_doc_len = self.args.max_doc_len

        if self.args.dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
            MAX_SEARCH_LIMIT = 5
            if self.args.dataset_name in ['hotpotqa', 'musique', 'bamboogle', '2wiki']:
                MAX_SEARCH_LIMIT = 10
                MAX_TURN = 15
            top_k = 10
            max_doc_len = 3000

        # 准备输入
        input_list = []
        for item in self.filtered_data:
            question = item['Question']

            # 生成任务指令
            if self.args.dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
                from prompts import get_singleqa_search_o1_instruction
                if self.args.dataset_name in ['nq', 'triviaqa']:
                    instruction = get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT)
                else:
                    from prompts import get_multiqa_search_o1_instruction
                    instruction = get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT)

                from prompts import get_task_instruction_openqa
                if 'qwq' in self.args.model_path.lower():
                    user_prompt = get_task_instruction_openqa(question, model_name='qwq')
                else:
                    user_prompt = get_task_instruction_openqa(question)

            elif self.args.dataset_name in ['math500', 'aime', 'amc']:
                from prompts import get_math_search_o1_instruction
                instruction = get_math_search_o1_instruction(MAX_SEARCH_LIMIT)
                from prompts import get_task_instruction_math
                if 'qwq' in self.args.model_path.lower():
                    user_prompt = get_task_instruction_math(question, model_name='qwq')
                else:
                    user_prompt = get_task_instruction_math(question)

            elif self.args.dataset_name == 'gpqa':
                from prompts import get_gpqa_search_o1_instruction
                instruction = get_gpqa_search_o1_instruction(MAX_SEARCH_LIMIT)
                from prompts import get_task_instruction_multi_choice
                if 'qwq' in self.args.model_path.lower():
                    user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
                elif 'llama' in self.args.model_path.lower():
                    user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
                else:
                    user_prompt = get_task_instruction_multi_choice(question)

            elif self.args.dataset_name == 'livecode':
                from prompts import get_code_search_o1_instruction
                instruction = get_code_search_o1_instruction(MAX_SEARCH_LIMIT)
                question_title = item.get('question_title', '')
                from prompts import get_task_instruction_code
                if 'qwq' in self.args.model_path.lower():
                    user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
                else:
                    user_prompt = get_task_instruction_code(question)
            else:
                user_prompt = ""

            # 添加知识图谱增强指令
            enhanced_instruction = self._get_kg_enhanced_instruction(instruction)

            prompt = [{"role": "user", "content": enhanced_instruction + user_prompt}]
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            input_list.append(prompt)

        # 限制样本数量
        if self.args.subset_num != -1:
            input_list = input_list[:self.args.subset_num]
            self.filtered_data = self.filtered_data[:self.args.subset_num]

        # 初始化活动序列
        active_sequences = [{
            'item': item,
            'prompt': prompt,
            'output': '',
            'finished': False,
            'history': [],
            'search_count': 0,
            'executed_search_queries': set(),
            'kg_built': False
        } for item, prompt in zip(self.filtered_data, input_list)]

        # 设置最大token数
        if 'qwq' in self.args.model_path.lower():
            if self.args.dataset_name in ['aime', 'amc', 'livecode']:
                max_tokens = 32768
            else:
                max_tokens = 20480
        else:
            max_tokens = 8192

        # 主推理循环
        start_time = time.time()
        turn = 0
        batch_output_records = []

        while True:
            sequences_needing_generation = [seq for seq in active_sequences if not seq['finished']]

            if sequences_needing_generation:
                turn += 1
                print(f'\n-------------- Turn {turn} --------------')
                print(f"We have {len(sequences_needing_generation)} sequences needing generation...")

                # 批量生成
                prompts = [s['prompt'] for s in sequences_needing_generation]

                sampling_params = SamplingParams(
                    max_tokens=max_tokens,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    top_k=self.args.top_k_sampling,
                    repetition_penalty=self.args.repetition_penalty,
                    stop=[END_SEARCH_QUERY, self.tokenizer.eos_token],
                    include_stop_str_in_output=True,
                )

                output_list = self.llm.generate(prompts, sampling_params=sampling_params)
                print("Generation completed, processing outputs...")

                # 处理输出和搜索
                batch_original_questions = []
                batch_prev_reasonings = []
                batch_search_queries = []
                batch_documents = []
                batch_sequences = []
                all_urls_to_fetch = set()
                url_snippets = {}

                for seq, out in zip(sequences_needing_generation, output_list):
                    text = out.outputs[0].text
                    seq['history'].append(text)
                    seq['prompt'] += text
                    seq['output'] += text

                    # 提取搜索查询
                    search_query = self._extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

                    if search_query and seq['output'].rstrip().endswith(END_SEARCH_QUERY):
                        if seq['search_count'] < MAX_SEARCH_LIMIT and search_query not in seq['executed_search_queries']:
                            # 执行搜索
                            if search_query in self.search_cache:
                                results = self.search_cache[search_query]
                                print(f"Using cached search results for query: \"{search_query}\"")
                            else:
                                try:
                                    results = bing_web_search(search_query, self.args.bing_subscription_key,
                                                           self.args.bing_endpoint, market='en-US', language='en')
                                    self.search_cache[search_query] = results
                                    print(f"Executed and cached search for query: \"{search_query}\"")
                                except Exception as e:
                                    print(f"Error during search query '{search_query}': {e}")
                                    self.search_cache[search_query] = {}
                                    results = {}

                            # 提取相关信息
                            relevant_info = extract_relevant_info(results)[:top_k]
                            seq['relevant_info'] = relevant_info

                            # 准备URL获取
                            urls_to_fetch = [it['url'] for it in relevant_info]
                            snippets = {info['url']: info['snippet'] for info in relevant_info if 'snippet' in info}

                            urls_to_fetch_filtered = [u for u in urls_to_fetch if u not in self.url_cache]

                            for url in urls_to_fetch_filtered:
                                all_urls_to_fetch.add(url)
                                url_snippets[url] = snippets.get(url, "")

                            # 准备推理文本
                            all_reasoning_steps = seq['output'].replace('\n\n', '\n').split("\n")
                            truncated_prev_reasoning = ""
                            for i, step in enumerate(all_reasoning_steps):
                                truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

                            prev_steps = truncated_prev_reasoning.split('\n\n')
                            if len(prev_steps) <= 5:
                                truncated_prev_reasoning = '\n\n'.join(prev_steps)
                            else:
                                truncated_prev_reasoning = ''
                                for i, step in enumerate(prev_steps):
                                    if i == 0 or i >= len(prev_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                                        truncated_prev_reasoning += step + '\n\n'
                                    else:
                                        if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                                            truncated_prev_reasoning += '...\n\n'
                            truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')

                            batch_original_questions.append(seq['item']['Question'])
                            batch_prev_reasonings.append(truncated_prev_reasoning)
                            batch_search_queries.append(search_query)
                            batch_sequences.append(seq)

                            seq['search_count'] += 1
                            seq['executed_search_queries'].add(search_query)

                        elif seq['search_count'] >= MAX_SEARCH_LIMIT:
                            limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                            seq['prompt'] += limit_message
                            seq['output'] += limit_message
                            seq['history'].append(limit_message)
                            print(f"Search limit reached for query: \"{search_query}\"")

                        else:
                            limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                            seq['prompt'] += limit_message
                            seq['output'] += limit_message
                            seq['history'].append(limit_message)
                            print(f"Repeated search for query: \"{search_query}\"")
                    else:
                        seq['finished'] = True
                        print("Sequence marked as complete.")

                # 批量获取URL内容
                if all_urls_to_fetch:
                    print(f"Fetching {len(all_urls_to_fetch)} URLs...")
                    try:
                        fetched_contents = fetch_page_content(
                            list(all_urls_to_fetch),
                            use_jina=self.args.use_jina,
                            jina_api_key=self.args.jina_api_key,
                        )
                        print(f"Fetched {len(fetched_contents)} URLs successfully.")
                    except Exception as e:
                        print(f"Error during batch URL fetching: {e}")
                        fetched_contents = {url: f"Error fetching URL: {e}" for url in all_urls_to_fetch}

                    for url, content in fetched_contents.items():
                        self.url_cache[url] = content

                # 准备格式化文档
                for relevant_info in [seq['relevant_info'] for seq in batch_sequences]:
                    formatted_documents = ""
                    for i, doc_info in enumerate(relevant_info):
                        url = doc_info['url']
                        raw_context = self.url_cache.get(url, "")
                        doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','')

                        # 简化的上下文提取
                        success, filtered_context = True, raw_context[:max_doc_len*2]

                        doc_info['context'] = filtered_context
                        formatted_documents += f"**Web Page {i + 1}:**\n"
                        formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"

                    batch_documents.append(formatted_documents)

                # 使用增强的推理链生成
                if batch_sequences:
                    print(f"Batch processing {len(batch_sequences)} sequences with enhanced KG reasoning...")
                    webpage_analyses = self._enhanced_webpage_to_reasonchain_batch(
                        original_questions=batch_original_questions,
                        prev_reasonings=batch_prev_reasonings,
                        search_queries=batch_search_queries,
                        documents=batch_documents,
                        dataset_name=self.args.dataset_name,
                        batch_output_records=batch_output_records,
                        max_tokens=max_tokens,
                    )
                    print("Batch generation completed, assigning outputs to sequences...")

                    for seq, analysis in zip(batch_sequences, webpage_analyses):
                        append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)

            unfinished = [seq for seq in active_sequences if not seq['finished']]
            if not unfinished:
                break
            else:
                if turn >= MAX_TURN:
                    print(f"Maximum number of turns ({MAX_TURN}) reached, stopping.")
                    break

        total_time = time.time() - start_time

        # 保存批量输出记录
        t = time.localtime()
        batch_output_file = os.path.join(self.output_dir, f'{self.args.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.info_extract.json')

        with open(batch_output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_output_records, f, ensure_ascii=False, indent=2)

        print(f"Batch outputs saved to {batch_output_file}")

        # 准备评估
        output_list = [seq['output'] for seq in active_sequences]
        run_evaluation(self.filtered_data, input_list, output_list, self.args.dataset_name, self.output_dir, total_time, self.args.split)

        print("Search-o1-KG inference completed!")

    def _extract_between(self, text: str, start_tag: str, end_tag: str) -> Optional[str]:
        """从文本中提取两个标记之间的内容"""
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Run Search O1 with Knowledge Graph enhancement")

    # 数据集参数
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['gpqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 'triviaqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle'],
                        help="Name of the dataset to use")
    parser.add_argument('--split', type=str, required=True,
                        choices=['test', 'diamond', 'main', 'extended'],
                        help="Dataset split to use")
    parser.add_argument('--subset_num', type=int, default=-1,
                        help="Number of examples to process")

    # 搜索参数
    parser.add_argument('--max_search_limit', type=int, default=10,
                        help="Maximum number of searches per question")
    parser.add_argument('--max_turn', type=int, default=15,
                        help="Maximum number of turns")
    parser.add_argument('--top_k', type=int, default=10,
                        help="Maximum number of search documents to return")
    parser.add_argument('--max_doc_len', type=int, default=3000,
                        help="Maximum length of each searched document")
    parser.add_argument('--use_jina', type=bool, default=True,
                        help="Whether to use Jina API for document fetching")
    parser.add_argument('--jina_api_key', type=str, default='None',
                        help="Your Jina API Key")

    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the pre-trained model")
    parser.add_argument('--temperature', type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument('--top_p', type=float, default=0.8,
                        help="Top-p sampling parameter")
    parser.add_argument('--top_k_sampling', type=int, default=20,
                        help="Top-k sampling parameter")
    parser.add_argument('--repetition_penalty', type=float, default=None,
                        help="Repetition penalty")
    parser.add_argument('--max_tokens', type=int, default=32768,
                        help="Maximum number of tokens to generate")

    # GNN参数
    parser.add_argument('--gnn_hidden_dim', type=int, default=128,
                        help="GNN hidden dimension")
    parser.add_argument('--gnn_output_dim', type=int, default=64,
                        help="GNN output dimension")
    parser.add_argument('--gnn_num_layers', type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument('--gnn_dropout', type=float, default=0.1,
                        help="GNN dropout rate")

    # API参数
    parser.add_argument('--bing_subscription_key', type=str, required=True,
                        help="Bing Search API subscription key")
    parser.add_argument('--bing_endpoint', type=str, default="https://api.bing.microsoft.com/v7.0/search",
                        help="Bing Search API endpoint")

    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()

    # 设置重复惩罚
    if args.repetition_penalty is None:
        args.repetition_penalty = 1.05 if 'qwq' in args.model_path.lower() else 1.0

    # 处理Jina API密钥
    if args.jina_api_key == 'None':
        args.jina_api_key = None

    # 运行Search-o1-KG
    search_o1_kg = SearchO1KG(args)
    search_o1_kg.run_inference()

if __name__ == "__main__":
    main()