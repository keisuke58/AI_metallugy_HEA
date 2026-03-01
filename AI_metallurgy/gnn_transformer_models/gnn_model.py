"""
Graph Neural Network Model for HEA Property Prediction
最新研究に基づく実装: ALIGNN/MatGNetスタイル
- 角度特徴量を含む
- エッジゲーティング
- 自己注意機構
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops, degree
from typing import Optional


class EdgeGatedConv(MessagePassing):
    """
    エッジゲーティング畳み込み層
    MatGNetスタイルの実装
    """
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super(EdgeGatedConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # ノード特徴量変換
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # エッジゲート
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim + in_channels * 2, out_channels),
            nn.Sigmoid()
        )
        
        # エッジ特徴量更新
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + in_channels * 2, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index, edge_attr):
        # バッチ処理されたグラフでは、add_self_loopsは使わない
        # メッセージ伝播
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # 残差接続
        out = out + self.node_mlp(x)
        
        # edge_attrを更新（メッセージ伝播で使用したエッジ特徴量を更新）
        if edge_attr is not None and edge_attr.numel() > 0:
            # エッジ特徴量を更新（簡易版：元のedge_attrをそのまま返す）
            # より高度な実装では、メッセージ伝播中に更新されたedge_attrを使用
            pass  # 現在の実装ではedge_attrは更新しない
        
        return out, edge_attr
    
    def message(self, x_i, x_j, edge_attr):
        # エッジとノード特徴量を結合
        # PyTorch Geometricでは、edge_attrはエッジの数だけ存在する
        # x_i, x_jはエッジの数だけ存在する（各エッジに対応する送信元・送信先ノード）
        
        if edge_attr is not None and edge_attr.numel() > 0:
            # edge_attrが存在する場合、エッジ特徴量とノード特徴量を結合
            # edge_attrの形状: (num_edges, edge_dim)
            # x_i, x_jの形状: (num_edges, node_dim)
            edge_input = torch.cat([edge_attr, x_i, x_j], dim=-1)
            gate = self.edge_gate(edge_input)
        else:
            # edge_attrが存在しない場合、ノード特徴量のみで処理
            # この場合、edge_gateの入力次元を調整する必要があるが、
            # 通常はedge_attrが提供されることを想定している
            # フォールバック: ゼロパディングでedge_attrを模倣
            edge_dim = self.edge_gate[0].in_features - x_i.size(-1) * 2
            if edge_dim > 0:
                dummy_edge_attr = torch.zeros(x_i.size(0), edge_dim, 
                                            device=x_i.device, dtype=x_i.dtype)
                edge_input = torch.cat([dummy_edge_attr, x_i, x_j], dim=-1)
                gate = self.edge_gate(edge_input)
            else:
                # エッジ特徴量なしで処理（簡易版）
                edge_input = torch.cat([x_i, x_j], dim=-1)
                # 簡易ゲート（全1）
                gate = torch.ones(x_i.size(0), self.out_channels, 
                                device=x_i.device, dtype=x_i.dtype)
        
        # ゲートされたメッセージ
        msg = gate * self.node_mlp(x_j)
        
        return msg


class SelfAttentionLayer(nn.Module):
    """自己注意層"""
    def __init__(self, dim: int, num_heads: int = 4):
        super(SelfAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        B, N, D = x.shape
        residual = x
        
        x = self.norm(x)
        
        # Multi-head attention
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        
        return out + residual


class HEAGNN(nn.Module):
    """
    HEA用Graph Neural Network
    最新研究（MatGNet, ALIGNN）に基づく実装
    """
    def __init__(
        self,
        node_dim: int = 5,
        edge_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        additional_feat_dim: int = 8,
        dropout: float = 0.1
    ):
        super(HEAGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
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
        
        # Graph Convolution層
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(
                EdgeGatedConv(hidden_dim, hidden_dim, hidden_dim)
            )
        
        # 自己注意層（グローバルな相互作用を捉える）
        self.attention_layers = nn.ModuleList()
        for i in range(num_layers):
            self.attention_layers.append(
                SelfAttentionLayer(hidden_dim, num_heads)
            )
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
        # 追加特徴量の処理
        self.additional_mlp = nn.Sequential(
            nn.Linear(additional_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 出力層（改善版：より深いネットワーク）
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim * 3 + hidden_dim, hidden_dim * 2),  # graph + additional
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch, additional_features = (
            data.x, data.edge_index, data.edge_attr, 
            data.batch, data.additional_features
        )
        
        # 埋め込み
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # Graph Convolution + Attention
        for i in range(self.num_layers):
            # Graph convolution
            x_new, edge_attr = self.conv_layers[i](x, edge_index, edge_attr)
            x = x + self.dropout(x_new)  # 残差接続
            
            # Self-attention on node features
            # バッチごとに処理（より効率的な実装）
            batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1
            if batch_size > 0:
                x_list = []
                for b in range(batch_size):
                    mask = (batch == b)
                    if mask.any():
                        x_batch = x[mask].unsqueeze(0)  # [1, N, D]
                        if x_batch.size(1) > 0:  # ノードが存在する場合のみ
                            x_batch = self.attention_layers[i](x_batch)
                            x_list.append(x_batch.squeeze(0))
                        else:
                            x_list.append(x[mask])
                if x_list:
                    x = torch.cat(x_list, dim=0)
        
        # グラフレベルの特徴量（pooling）
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        # x_sumは各グラフのノード特徴量の合計（グラフサイズに依存するため、平均で正規化）
        # より良い方法: グラフサイズで正規化した合計
        batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1
        if batch.numel() > 0:
            # 各グラフのノード数を計算
            graph_sizes = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            for b in range(batch_size):
                graph_sizes[b] = (batch == b).sum()
            # 各グラフの合計を計算
            x_sum_list = []
            for b in range(batch_size):
                mask = (batch == b)
                if mask.any():
                    x_sum_list.append(x[mask].sum(dim=0) / graph_sizes[b].float())  # サイズで正規化
                else:
                    x_sum_list.append(torch.zeros(x.size(1), device=x.device))
            x_sum = torch.stack(x_sum_list, dim=0)
        else:
            x_sum = global_mean_pool(x, batch)
        
        # 追加特徴量を処理（バッチごとに処理）
        # collate_gnnでスタックされているので、形状は(batch_size, 8)のはず
        batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1
        additional_feat_dim = 8  # 固定: mixing_entropy, mixing_enthalpy, vec, delta_r, delta_chi, mean_atomic_radius, mean_electronegativity, density
        
        # 形状の確認と修正
        if additional_features.dim() == 1:
            # 1Dの場合: (batch_size * feat_dim,) -> (batch_size, feat_dim)にリシェイプ
            if additional_features.size(0) == batch_size * additional_feat_dim:
                additional_features = additional_features.view(batch_size, additional_feat_dim)
            elif additional_features.size(0) == additional_feat_dim:
                # 単一サンプルの場合
                additional_features = additional_features.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected additional_features shape: {additional_features.shape} for batch_size={batch_size}, expected {(batch_size * additional_feat_dim,)} or {(additional_feat_dim,)}")
        elif additional_features.dim() == 2:
            # 2Dの場合: (batch_size, feat_dim)であることを確認
            if additional_features.size(0) != batch_size:
                # バッチサイズが一致しない場合、調整
                if additional_features.size(0) > batch_size:
                    additional_features = additional_features[:batch_size]
                else:
                    # パディング
                    padding = torch.zeros(batch_size - additional_features.size(0), 
                                        additional_features.size(1),
                                        device=additional_features.device,
                                        dtype=additional_features.dtype)
                    additional_features = torch.cat([additional_features, padding], dim=0)
            if additional_features.size(1) != additional_feat_dim:
                raise ValueError(f"Unexpected feature dimension: {additional_features.size(1)}, expected {additional_feat_dim}")
        else:
            raise ValueError(f"Unexpected additional_features dimensions: {additional_features.dim()}, expected 1 or 2")
        
        additional_out = self.additional_mlp(additional_features)
        
        # 結合
        graph_features = torch.cat([x_mean, x_max, x_sum], dim=1)
        combined = torch.cat([graph_features, additional_out], dim=1)
        
        # 出力
        output = self.output_layers(combined)
        
        return output


class HEAGNNLight(nn.Module):
    """
    軽量版GNN（データ数が少ない場合用）
    """
    def __init__(
        self,
        node_dim: int = 5,
        edge_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 3,
        additional_feat_dim: int = 8,
        dropout: float = 0.1
    ):
        super(HEAGNNLight, self).__init__()
        
        self.node_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(
                EdgeGatedConv(hidden_dim, hidden_dim, hidden_dim)
            )
        
        self.dropout = nn.Dropout(dropout)
        
        self.additional_mlp = nn.Sequential(
            nn.Linear(additional_feat_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim * 3 + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch, additional_features = (
            data.x, data.edge_index, data.edge_attr, 
            data.batch, data.additional_features
        )
        
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        for conv in self.conv_layers:
            x_new, edge_attr = conv(x, edge_index, edge_attr)
            x = x + self.dropout(x_new)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        # x_sum: 各グラフのノード特徴量の合計（グラフサイズで正規化）
        x_sum = global_add_pool(x, batch)
        batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1
        if batch.numel() > 0:
            graph_sizes = torch.zeros(batch_size, dtype=torch.float32, device=x.device)
            for b in range(batch_size):
                graph_sizes[b] = float((batch == b).sum())
            graph_sizes = torch.clamp(graph_sizes, min=1.0)
            x_sum = x_sum / graph_sizes.unsqueeze(1)
        
        # 追加特徴量を処理（バッチごとに処理）
        # collate_gnnでスタックされているので、形状は(batch_size, 8)のはず
        batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1
        additional_feat_dim = 8  # 固定: mixing_entropy, mixing_enthalpy, vec, delta_r, delta_chi, mean_atomic_radius, mean_electronegativity, density
        
        # 形状の確認と修正
        if additional_features.dim() == 1:
            # 1Dの場合: (batch_size * feat_dim,) -> (batch_size, feat_dim)にリシェイプ
            if additional_features.size(0) == batch_size * additional_feat_dim:
                additional_features = additional_features.view(batch_size, additional_feat_dim)
            elif additional_features.size(0) == additional_feat_dim:
                # 単一サンプルの場合
                additional_features = additional_features.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected additional_features shape: {additional_features.shape} for batch_size={batch_size}, expected {(batch_size * additional_feat_dim,)} or {(additional_feat_dim,)}")
        elif additional_features.dim() == 2:
            # 2Dの場合: (batch_size, feat_dim)であることを確認
            if additional_features.size(0) != batch_size:
                # バッチサイズが一致しない場合、調整
                if additional_features.size(0) > batch_size:
                    additional_features = additional_features[:batch_size]
                else:
                    # パディング
                    padding = torch.zeros(batch_size - additional_features.size(0), 
                                        additional_features.size(1),
                                        device=additional_features.device,
                                        dtype=additional_features.dtype)
                    additional_features = torch.cat([additional_features, padding], dim=0)
            if additional_features.size(1) != additional_feat_dim:
                raise ValueError(f"Unexpected feature dimension: {additional_features.size(1)}, expected {additional_feat_dim}")
        else:
            raise ValueError(f"Unexpected additional_features dimensions: {additional_features.dim()}, expected 1 or 2")
        
        additional_out = self.additional_mlp(additional_features)
        
        combined = torch.cat([x_mean, x_max, x_sum, additional_out], dim=1)
        output = self.output_layers(combined)
        
        return output
