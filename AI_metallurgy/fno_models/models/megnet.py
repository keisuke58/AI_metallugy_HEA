"""
MatErials Graph Networks (MEGNet) 実装
Chen et al. (2019) "Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops


class MEGNetLayer(MessagePassing):
    """
    MEGNet Graph Convolution Layer
    """
    def __init__(self, node_dim: int, edge_dim: int, state_dim: int, out_dim: int):
        # MessagePassing の node_dim パラメータを明示的に 0 に設定（ノード次元のインデックス）
        super(MEGNetLayer, self).__init__(aggr='add', flow='source_to_target', node_dim=0)
        
        # 注意: MessagePassing は内部で self.node_dim を次元インデックスとして使用する
        # ここで self.node_dim を上書きすると次元エラーの原因になるため、別名で保持する
        self.node_feat_dim = node_dim
        self.edge_dim = edge_dim
        self.state_dim = state_dim
        self.out_dim = out_dim
        
        # エッジ更新関数
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + self.node_feat_dim * 2 + state_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # ノード更新関数
        self.node_mlp = nn.Sequential(
            nn.Linear(self.node_feat_dim + out_dim + state_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # 状態更新関数
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim + out_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim)
        )
    
    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        Args:
            x: [num_nodes, node_dim] ノード特徴量
            edge_index: [2, num_edges] エッジインデックス
            edge_attr: [num_edges, edge_dim] エッジ特徴量
            u: [batch_size, state_dim] グローバル状態
            batch: [num_nodes] バッチインデックス
        """
        # エッジ更新: messageメソッドを直接呼び出してエッジ特徴量を更新
        # propagateはメッセージを集約するため、エッジ特徴量を直接返さない
        # そのため、messageメソッドを直接呼び出してエッジ特徴量を更新
        row, col = edge_index
        x_i = x[row]  # エッジのソースノード
        x_j = x[col]  # エッジのターゲットノード
        batch_i = batch[row] if batch.numel() > 0 else torch.zeros(row.size(0), dtype=torch.long, device=row.device)
        
        # エッジ特徴量を更新
        edge_attr = self.message(x_i, x_j, edge_attr, u, batch_i)
        
        # ノード更新: 各ノードに接続されているエッジ特徴量を集約
        # エッジ特徴量をノードに集約（各ノードに接続されているエッジの平均）
        num_nodes = x.size(0)
        node_edge_attr = torch.zeros(num_nodes, edge_attr.size(1), device=edge_attr.device, dtype=edge_attr.dtype)
        node_edge_count = torch.zeros(num_nodes, device=edge_attr.device, dtype=edge_attr.dtype)
        node_edge_attr.index_add_(0, row, edge_attr)
        node_edge_count.index_add_(0, row, torch.ones_like(row, dtype=edge_attr.dtype))
        node_edge_attr = node_edge_attr / node_edge_count.unsqueeze(1).clamp(min=1)
        
        x = self.node_mlp(torch.cat([x, node_edge_attr, u[batch]], dim=1))
        
        # 状態更新
        batch_size = u.size(0)
        if batch.numel() > 0 and edge_index.size(1) > 0:
            # エッジのバッチインデックスを取得
            max_node_idx = edge_index[0].max().item()
            if max_node_idx < batch.size(0):
                edge_batch = batch[edge_index[0]]
            else:
                edge_batch = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        else:
            edge_batch = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        
        # global_mean_poolにsizeを明示的に指定
        edge_pooled = global_mean_pool(edge_attr, edge_batch, size=batch_size)
        u = self.state_mlp(torch.cat([u, edge_pooled], dim=1))
        
        return x, edge_attr, u
    
    def message(self, x_i, x_j, edge_attr, u, batch_i):
        """
        メッセージ計算
        """
        # グローバル状態を取得
        u_i = u[batch_i]
        
        # エッジ特徴量を更新
        edge_input = torch.cat([edge_attr, x_i, x_j, u_i], dim=1)
        edge_out = self.edge_mlp(edge_input)
        
        return edge_out


class MEGNet(nn.Module):
    """
    MEGNet Model for Material Property Prediction
    """
    def __init__(
        self,
        node_dim: int = 5,
        edge_dim: int = 3,
        state_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        additional_feat_dim: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            node_dim: ノード特徴量次元
            edge_dim: エッジ特徴量次元
            state_dim: グローバル状態次元
            hidden_dim: 隠れ層次元
            num_layers: MEGNet層数
            additional_feat_dim: 追加特徴量次元
            dropout: ドロップアウト率
        """
        super(MEGNet, self).__init__()
        
        self.num_layers = num_layers
        self.state_dim = state_dim
        
        # 入力埋め込み
        self.node_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        self.state_embedding = nn.Sequential(
            nn.Linear(additional_feat_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.SiLU()
        )
        
        # MEGNet層
        self.megnet_layers = nn.ModuleList()
        for i in range(num_layers):
            self.megnet_layers.append(
                MEGNetLayer(hidden_dim, hidden_dim, state_dim, hidden_dim)
            )
        
        self.dropout = nn.Dropout(dropout)
        
        # 出力層
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim * 3 + state_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        """
        Args:
            data: PyTorch Geometric Dataオブジェクト
                - x: [num_nodes, node_dim]
                - edge_index: [2, num_edges]
                - edge_attr: [num_edges, edge_dim]
                - batch: [num_nodes]
                - additional_features: [batch_size, additional_feat_dim]
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
        
        # グローバル状態の初期化
        batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1
        
        # Handle additional_features: PyTorch Geometric batches them by concatenating
        # If 1D, reshape to [batch_size, feature_dim]
        if additional_features.dim() == 1:
            # Get feature_dim from the first layer of state_embedding
            if isinstance(self.state_embedding, nn.Sequential) and len(self.state_embedding) > 0:
                feature_dim = self.state_embedding[0].in_features
            else:
                feature_dim = 29  # Default (updated to 29)
            
            # PyTorch Geometricはadditional_featuresを連結しない（各グラフごとに保持）
            # したがって、1次元の場合は単一サンプルとして扱う
            if additional_features.size(0) == feature_dim:
                # Single sample case: [feature_dim] -> [1, feature_dim]
                additional_features = additional_features.unsqueeze(0)
            elif additional_features.size(0) == batch_size * feature_dim:
                # Concatenated case: [batch_size * feature_dim] -> [batch_size, feature_dim]
                additional_features = additional_features.view(batch_size, feature_dim)
            else:
                # Fallback: 単一サンプルとして扱い、バッチサイズに合わせて拡張
                if additional_features.size(0) < feature_dim:
                    # パディング
                    padding = torch.zeros(feature_dim - additional_features.size(0), 
                                        device=additional_features.device, 
                                        dtype=additional_features.dtype)
                    additional_features = torch.cat([additional_features, padding])
                additional_features = additional_features[:feature_dim].unsqueeze(0).expand(batch_size, -1)
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
        
        u = self.state_embedding(additional_features)  # [batch_size, state_dim]
        
        # MEGNet層
        for megnet_layer in self.megnet_layers:
            x_new, edge_attr_new, u_new = megnet_layer(x, edge_index, edge_attr, u, batch)
            x = x + self.dropout(x_new)  # 残差接続
            edge_attr = edge_attr + self.dropout(edge_attr_new)
            u = u + self.dropout(u_new)
        
        # グラフレベルの特徴量
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_mean_pool(x, batch)  # 代替として使用
        
        # 結合
        graph_features = torch.cat([x_mean, x_max, x_sum, u], dim=1)
        
        # 出力
        output = self.output_layers(graph_features)
        
        return output
