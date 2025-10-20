"""
图神经网络模块

实现多种图神经网络架构：
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- GraphSAGE
- 自定义知识图谱推理网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import networkx as nx

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphEmbeddingConfig:
    """图嵌入配置"""
    hidden_dim: int = 128
    output_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    activation: str = 'relu'
    use_batch_norm: bool = True
    attention_heads: int = 4  # for GAT
    aggregation_type: str = 'mean'  # for GraphSAGE

class GraphConvolutionLayer(nn.Module):
    """图卷积层"""

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True, activation: str = 'relu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)

        # 初始化权重
        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征矩阵 [num_nodes, input_dim]
            adj: 邻接矩阵 [num_nodes, num_nodes]

        Returns:
            输出特征矩阵 [num_nodes, output_dim]
        """
        # 消息传递: A * X * W
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.activation == 'relu':
            output = F.relu(output)
        elif self.activation == 'tanh':
            output = torch.tanh(output)
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(output)

        return output

class GraphAttentionLayer(nn.Module):
    """图注意力层"""

    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # 注意力参数
        self.W = nn.Parameter(torch.FloatTensor(num_heads, input_dim, output_dim // num_heads))
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * (output_dim // num_heads), 1))

        self.dropout_layer = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        nn.init.kaiming_uniform_(self.W)
        nn.init.kaiming_uniform_(self.a)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征矩阵 [num_nodes, input_dim]
            adj: 邻接矩阵 [num_nodes, num_nodes]

        Returns:
            输出特征矩阵 [num_nodes, output_dim]
        """
        num_nodes = x.size(0)

        # 线性变换
        h = torch.einsum('hio,bj->bhji', self.W, x)  # [num_heads, num_nodes, head_dim]

        # 计算注意力分数
        h_repeat = h.repeat(1, num_nodes, 1)  # [num_heads, num_nodes * num_nodes, head_dim]
        h_tiled = h.repeat(1, 1, num_nodes).view(num_heads, num_nodes * num_nodes, -1)  # [num_heads, num_nodes * num_nodes, head_dim]

        cat_features = torch.cat([h_repeat, h_tiled], dim=-1)  # [num_heads, num_nodes * num_nodes, 2 * head_dim]

        # 注意力计算
        energy = torch.einsum('hki,hkj->hkj', self.a, cat_features.transpose(-1, -2)).squeeze(-1)  # [num_heads, num_nodes * num_nodes]
        energy = self.leaky_relu(energy)
        energy = energy.view(num_heads, num_nodes, num_nodes)

        # 应用邻接矩阵掩码
        mask = -9e15 * torch.ones_like(energy)
        energy = torch.where(adj.unsqueeze(0) > 0, energy, mask)

        # 注意力权重
        attention = F.softmax(energy, dim=-1)
        attention = self.dropout_layer(attention)

        # 加权聚合
        h_prime = torch.einsum('hij,hjk->hik', attention, h)  # [num_heads, num_nodes, head_dim]

        # 拼接多头注意力
        output = h_prime.view(num_nodes, -1)  # [num_nodes, output_dim]

        return output

class GraphSAGELayer(nn.Module):
    """GraphSAGE层"""

    def __init__(self, input_dim: int, output_dim: int, aggregation_type: str = 'mean'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregation_type = aggregation_type

        # 自身变换
        self.self_linear = nn.Linear(input_dim, output_dim)

        # 邻居聚合
        self.neighbor_linear = nn.Linear(input_dim, output_dim)

        # 聚合后变换
        self.combine_linear = nn.Linear(output_dim * 2, output_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征矩阵 [num_nodes, input_dim]
            adj: 邻接矩阵 [num_nodes, num_nodes]

        Returns:
            输出特征矩阵 [num_nodes, output_dim]
        """
        num_nodes = x.size(0)

        # 自身特征变换
        self_features = self.self_linear(x)

        # 邻居聚合
        neighbor_features = []
        for i in range(num_nodes):
            # 获取邻居索引
            neighbors = torch.where(adj[i] > 0)[0]
            if len(neighbors) > 0:
                neighbor_x = x[neighbors]

                if self.aggregation_type == 'mean':
                    agg_feature = torch.mean(neighbor_x, dim=0)
                elif self.aggregation_type == 'max':
                    agg_feature = torch.max(neighbor_x, dim=0)[0]
                elif self.aggregation_type == 'sum':
                    agg_feature = torch.sum(neighbor_x, dim=0)
                else:
                    agg_feature = torch.mean(neighbor_x, dim=0)

                neighbor_features.append(agg_feature)
            else:
                neighbor_features.append(torch.zeros(x.size(1)))

        neighbor_features = torch.stack(neighbor_features)
        neighbor_features = self.neighbor_linear(neighbor_features)

        # 组合自身和邻居特征
        combined = torch.cat([self_features, neighbor_features], dim=-1)
        output = self.combine_linear(combined)

        return F.relu(output)

class KnowledgeGraphGNN(nn.Module):
    """知识图谱图神经网络"""

    def __init__(self, config: GraphEmbeddingConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()

        # 构建多层网络
        current_dim = config.input_dim if hasattr(config, 'input_dim') else 128

        for i in range(config.num_layers):
            if config.gnn_type == 'gcn':
                layer = GraphConvolutionLayer(current_dim, config.hidden_dim, activation=config.activation)
            elif config.gnn_type == 'gat':
                layer = GraphAttentionLayer(current_dim, config.hidden_dim, config.attention_heads)
            elif config.gnn_type == 'sage':
                layer = GraphSAGELayer(current_dim, config.hidden_dim, config.aggregation_type)
            else:
                raise ValueError(f"不支持的GNN类型: {config.gnn_type}")

            self.layers.append(layer)
            current_dim = config.hidden_dim

        # 输出层
        self.output_layer = nn.Linear(current_dim, config.output_dim)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # BatchNorm
        if config.use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(config.hidden_dim) for _ in range(config.num_layers)
            ])
        else:
            self.batch_norms = None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征矩阵 [num_nodes, input_dim]
            adj: 邻接矩阵 [num_nodes, num_nodes]

        Returns:
            节点嵌入 [num_nodes, output_dim]
        """
        h = x

        for i, layer in enumerate(self.layers):
            h = layer(h, adj)

            # BatchNorm
            if self.batch_norms:
                h = self.batch_norms[i](h)

            # Dropout
            h = self.dropout(h)

        # 输出层
        output = self.output_layer(h)

        return output

class GraphEmbeddingManager:
    """图嵌入管理器"""

    def __init__(self, config: GraphEmbeddingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化GNN模型
        self.models = {}
        self._init_models()

        # 损失函数
        self.criterion = nn.MSELoss()

        # 优化器
        self.optimizers = {}

    def _init_models(self):
        """初始化GNN模型"""
        gnn_types = ['gcn', 'gat', 'sage']

        for gnn_type in gnn_types:
            config = self.config
            config.gnn_type = gnn_type
            model = KnowledgeGraphGNN(config)
            model.to(self.device)
            self.models[gnn_type] = model
            self.optimizers[gnn_type] = torch.optim.Adam(model.parameters(), lr=0.001)

    def prepare_graph_data(self, nx_graph: nx.Graph, node_features: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        """
        准备图数据

        Args:
            nx_graph: NetworkX图
            node_features: 节点特征字典

        Returns:
            (特征矩阵, 邻接矩阵, 节点索引映射)
        """
        # 节点索引映射
        node_list = list(nx_graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        idx_to_node = {idx: node for idx, node in enumerate(node_list)}

        num_nodes = len(node_list)
        feature_dim = len(next(iter(node_features.values()))) if node_features else self.config.hidden_dim

        # 特征矩阵
        if node_features:
            x = np.zeros((num_nodes, feature_dim))
            for node, features in node_features.items():
                if node in node_to_idx:
                    x[node_to_idx[node]] = features
        else:
            # 随机初始化特征
            x = np.random.randn(num_nodes, feature_dim)

        x = torch.FloatTensor(x).to(self.device)

        # 邻接矩阵
        adj = np.zeros((num_nodes, num_nodes))
        for edge in nx_graph.edges():
            i, j = node_to_idx[edge[0]], node_to_idx[edge[1]]
            adj[i, j] = 1
            adj[j, i] = 1  # 无向图

        # 添加自环
        np.fill_diagonal(adj, 1)

        adj = torch.FloatTensor(adj).to(self.device)

        return x, adj, node_to_idx

    def generate_node_embeddings(self, nx_graph: nx.Graph, node_features: Dict[str, np.ndarray] = None, gnn_type: str = 'gcn') -> Dict[str, np.ndarray]:
        """
        生成节点嵌入

        Args:
            nx_graph: NetworkX图
            node_features: 节点特征字典
            gnn_type: GNN类型

        Returns:
            节点嵌入字典
        """
        model = self.models[gnn_type]
        model.eval()

        # 准备数据
        x, adj, node_to_idx = self.prepare_graph_data(nx_graph, node_features)

        with torch.no_grad():
            embeddings = model(x, adj)

        # 转换为numpy字典
        embedding_dict = {}
        for node, idx in node_to_idx.items():
            embedding_dict[node] = embeddings[idx].cpu().numpy()

        return embedding_dict

    def compute_node_similarity(self, embeddings: Dict[str, np.ndarray], node1: str, node2: str) -> float:
        """
        计算节点相似度

        Args:
            embeddings: 节点嵌入字典
            node1: 节点1
            node2: 节点2

        Returns:
            余弦相似度
        """
        if node1 not in embeddings or node2 not in embeddings:
            return 0.0

        emb1 = embeddings[node1]
        emb2 = embeddings[node2]

        # 余弦相似度
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def find_similar_nodes(self, target_node: str, embeddings: Dict[str, np.ndarray], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        查找相似节点

        Args:
            target_node: 目标节点
            embeddings: 节点嵌入字典
            top_k: 返回top-k相似节点

        Returns:
            (节点, 相似度) 列表
        """
        if target_node not in embeddings:
            return []

        similarities = []
        target_emb = embeddings[target_node]

        for node, emb in embeddings.items():
            if node != target_node:
                similarity = self.compute_node_similarity(embeddings, target_node, node)
                similarities.append((node, similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def train_embeddings(self, nx_graph: nx.Graph, node_features: Dict[str, np.ndarray] = None, epochs: int = 100, gnn_type: str = 'gcn'):
        """
        训练图嵌入

        Args:
            nx_graph: NetworkX图
            node_features: 节点特征字典
            epochs: 训练轮数
            gnn_type: GNN类型
        """
        model = self.models[gnn_type]
        optimizer = self.optimizers[gnn_type]
        model.train()

        # 准备数据
        x, adj, _ = self.prepare_graph_data(nx_graph, node_features)

        # 创建训练目标（简化：使用相邻节点作为正样本）
        adj_labels = adj.clone()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # 前向传播
            embeddings = model(x, adj)

            # 计算损失（简化：重建损失）
            reconstructed = torch.mm(embeddings, embeddings.t())
            loss = self.criterion(reconstructed, adj_labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    # 测试代码
    config = GraphEmbeddingConfig(
        hidden_dim=64,
        output_dim=32,
        num_layers=2,
        dropout=0.1
    )

    # 创建测试图
    graph = nx.Graph()
    graph.add_edges_from([
        ('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'),
        ('A', 'C'), ('B', 'D'), ('C', 'E')
    ])

    # 随机初始化节点特征
    node_features = {node: np.random.randn(10) for node in graph.nodes()}

    # 创建嵌入管理器
    manager = GraphEmbeddingManager(config)

    # 生成嵌入
    print("生成节点嵌入...")
    embeddings = manager.generate_node_embeddings(graph, node_features, 'gcn')

    # 打印嵌入结果
    for node, emb in embeddings.items():
        print(f"节点 {node}: 嵌入维度 {emb.shape}")

    # 计算相似度
    print("\n节点相似度:")
    node1, node2 = 'A', 'C'
    similarity = manager.compute_node_similarity(embeddings, node1, node2)
    print(f"{node1} 和 {node2} 的相似度: {similarity:.4f}")

    # 查找相似节点
    print(f"\n与节点 {node1} 最相似的节点:")
    similar_nodes = manager.find_similar_nodes(node1, embeddings, top_k=3)
    for node, sim in similar_nodes:
        print(f"  {node}: {sim:.4f}")