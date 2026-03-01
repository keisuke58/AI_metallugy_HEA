"""
Crystal Graph Convolutional Neural Networks (CGCNN) 実装
Xie & Grossman (2018) "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
import numpy as np


class CGCNNConv(MessagePassing):
    """
    CGCNN Graph Convolution Layer
    """
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        super(CGCNNConv, self).__init__(aggr='add', flow='source_to_target')
        
        # Don't use self.node_dim as it conflicts with MessagePassing's internal node_dim
        # MessagePassing uses node_dim internally to specify the dimension index (usually 0)
        # We store the feature dimension size separately
        self.node_feat_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        
        # エッジ特徴量の変換
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, out_dim),
            nn.Sigmoid()
        )
        
        # ノード特徴量の変換
        self.node_mlp = nn.Sequential(
            nn.Linear(self.node_feat_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        
        # 出力変換
        self.output_mlp = nn.Sequential(
            nn.Linear(self.node_feat_dim + out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: [num_nodes, node_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_dim]
        """
        # エッジ特徴量を変換
        edge_attr_transformed = self.edge_mlp(edge_attr)
        
        # メッセージ伝播
        # batched graph では MessagePassing が自動でサイズを推論する
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr_transformed)
        
        # ノード特徴量を変換
        x_transformed = self.node_mlp(x)
        
        # 結合して出力
        output = self.output_mlp(torch.cat([x, x_transformed], dim=1))
        
        return output
    
    def message(self, x_j, edge_attr):
        """
        メッセージ計算: エッジ特徴量でゲーティング
        """
        return edge_attr * x_j


class CGCNN(nn.Module):
    """
    CGCNN Model for Crystal Property Prediction
    """
    def __init__(
        self,
        node_dim: int = 5,
        edge_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
        additional_feat_dim: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            node_dim: ノード特徴量次元
            edge_dim: エッジ特徴量次元
            hidden_dim: 隠れ層次元
            num_layers: CGCNN層数
            additional_feat_dim: 追加特徴量次元
            dropout: ドロップアウト率
        """
        super(CGCNN, self).__init__()
        
        self.num_layers = num_layers
        
        # 入力埋め込み
        self.node_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # CGCNN層
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(
                CGCNNConv(hidden_dim, hidden_dim, hidden_dim)
            )
        
        self.dropout = nn.Dropout(dropout)
        
        # 追加特徴量の処理
        self.additional_mlp = nn.Sequential(
            nn.Linear(additional_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 出力層
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        """
        Args:
            data: PyTorch Geometric Dataオブジェクト
        Returns:
            [batch_size, 1]
        """
        x, edge_index, edge_attr, batch, additional_features = (
            data.x, data.edge_index, data.edge_attr,
            data.batch, data.additional_features
        )
        
        # 埋め込み
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # CGCNN層
        for conv_layer in self.conv_layers:
            # For batched graphs, size is automatically inferred by MessagePassing
            x_new = conv_layer(x, edge_index, edge_attr)
            x = x + self.dropout(x_new)  # 残差接続
        
        # グラフレベルの特徴量
        x_pooled = global_mean_pool(x, batch)
        
        # 追加特徴量の処理
        batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1
        
        # Handle additional_features: PyTorch Geometric batches them by concatenating
        # If 1D, reshape to [batch_size, feature_dim]
        if additional_features.dim() == 1:
            # Assume it's concatenated: [batch_size * feature_dim]
            # Get feature_dim from the first layer of additional_mlp
            if isinstance(self.additional_mlp, nn.Sequential) and len(self.additional_mlp) > 0:
                feature_dim = self.additional_mlp[0].in_features
            else:
                feature_dim = 8  # Default
            if additional_features.size(0) == batch_size * feature_dim:
                additional_features = additional_features.view(batch_size, feature_dim)
            elif additional_features.size(0) == feature_dim:
                # Single sample case
                additional_features = additional_features.unsqueeze(0).expand(batch_size, -1)
            else:
                # Fallback: pad or truncate
                feature_dim = 8  # Default
                if additional_features.size(0) >= batch_size * feature_dim:
                    additional_features = additional_features[:batch_size * feature_dim].view(batch_size, feature_dim)
                else:
                    # Pad with zeros
                    padding_size = batch_size * feature_dim - additional_features.size(0)
                    padding = torch.zeros(padding_size, device=additional_features.device, dtype=additional_features.dtype)
                    additional_features = torch.cat([additional_features, padding]).view(batch_size, feature_dim)
        elif additional_features.dim() == 2:
            # Already 2D, but check batch size
            if additional_features.size(0) != batch_size:
                if additional_features.size(0) > batch_size:
                    additional_features = additional_features[:batch_size]
                else:
                    # Pad with zeros
                    padding = torch.zeros(
                        batch_size - additional_features.size(0),
                        additional_features.size(1),
                        device=additional_features.device,
                        dtype=additional_features.dtype
                    )
                    additional_features = torch.cat([additional_features, padding], dim=0)
        
        additional_out = self.additional_mlp(additional_features)
        
        # 結合
        combined = torch.cat([x_pooled, additional_out], dim=1)
        
        # 出力
        output = self.output_layers(combined)
        
        return output
