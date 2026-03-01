"""
Transformer Model for HEA Property Prediction
最新研究に基づく実装: Crystalformer/AlloyBERTスタイル
- 位置エンコーディング
- Multi-head attention
- 組成情報の統合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """位置エンコーディング（Transformer用）"""
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer"""
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Activation
        self.activation = F.gelu if activation == 'gelu' else F.relu
    
    def forward(self, src, src_key_padding_mask: Optional[torch.Tensor] = None, src_mask: Optional[torch.Tensor] = None, is_causal: bool = False):
        # Self-attention with residual
        src2 = self.norm1(src)
        src2, _ = self.self_attn(
            src2, src2, src2, 
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask,
            is_causal=is_causal
        )
        src = src + self.dropout(src2)
        
        # Feed-forward with residual
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout(src2)
        
        return src


class CompositionEmbedding(nn.Module):
    """組成情報の埋め込み（元素トークン + 組成比）"""
    def __init__(self, vocab_size: int, d_model: int):
        super(CompositionEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.comp_projection = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, token_ids: torch.Tensor, comp_values: torch.Tensor):
        """
        Args:
            token_ids: [batch_size, seq_len]
            comp_values: [batch_size, seq_len]
        """
        # トークン埋め込み
        token_emb = self.token_embedding(token_ids)
        
        # 組成比の埋め込み
        comp_emb = self.comp_projection(comp_values.unsqueeze(-1))
        
        # 結合
        embeddings = token_emb + comp_emb
        embeddings = self.norm(embeddings)
        
        return embeddings


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence features"""
    def __init__(self, d_model: int):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, encoded: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args:
            encoded: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len]
        """
        # Attention scores
        attn_scores = self.attention(encoded).squeeze(-1)  # [batch_size, seq_len]
        
        # Mask out padding positions
        attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Weighted sum
        pooled = (encoded * attn_weights).sum(dim=1)  # [batch_size, d_model]
        return pooled


class HEATransformer(nn.Module):
    """
    HEA用Transformerモデル（改善版）
    最新研究（Crystalformer, AlloyBERT）に基づく実装
    精度向上のための改善:
    - Attention-based pooling
    - Cross-attention between sequence and additional features
    - Deeper output layers with residual connections
    - Better feature integration
    """
    def __init__(
        self,
        vocab_size: int = 20,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_seq_len: int = 20,
        additional_feat_dim: int = 8,
        dropout: float = 0.1
    ):
        super(HEATransformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 埋め込み層
        self.composition_embedding = CompositionEmbedding(vocab_size, d_model)
        
        # 位置エンコーディング
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer Encoder層
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Attention-based pooling
        self.attention_pooling = AttentionPooling(d_model)
        
        # 追加特徴量の処理（より深いネットワーク）
        self.additional_mlp = nn.Sequential(
            nn.Linear(additional_feat_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Cross-attention between sequence and additional features
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead // 2, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)
        
        # 出力層（より深く、残差接続あり）
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),  # seq_attn + max_pool + additional
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.output_layers = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # 初期化
        self._init_weights()
    
    def _init_weights(self):
        """重みの初期化（改善版）"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization with gain for GELU
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        comp_values: torch.Tensor,
        attention_mask: torch.Tensor,
        additional_features: torch.Tensor
    ):
        """
        Args:
            token_ids: [batch_size, seq_len]
            comp_values: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (1 for valid, 0 for padding)
            additional_features: [batch_size, additional_feat_dim]
        """
        batch_size, seq_len = token_ids.shape
        
        # 埋め込み
        src = self.composition_embedding(token_ids, comp_values)
        
        # 位置エンコーディング
        src = self.pos_encoder(src)
        
        # Attention maskをkey_padding_maskに変換
        key_padding_mask = (attention_mask == 0)  # True for padding
        
        # Transformer Encoder
        encoded = self.transformer_encoder(src, src_key_padding_mask=key_padding_mask)
        
        # Attention-based pooling（改善された特徴量抽出）
        seq_attn_pooled = self.attention_pooling(encoded, attention_mask)
        
        # 最大プーリング（多様な情報を捉える）
        max_pooled = encoded.max(dim=1)[0]
        
        # 平均プーリング（マスクを考慮）
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded)
        masked_encoded = encoded * mask_expanded
        mean_pooled = masked_encoded.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        # 追加特徴量を処理
        additional_out = self.additional_mlp(additional_features)
        
        # Cross-attention: sequence features attend to additional features
        additional_query = additional_out.unsqueeze(1)  # [batch_size, 1, d_model]
        cross_attn_out, _ = self.cross_attention(
            additional_query, encoded, encoded,
            key_padding_mask=key_padding_mask
        )
        cross_attn_out = self.cross_attn_norm(cross_attn_out.squeeze(1) + additional_out)
        
        # 複数の特徴量を結合（より豊富な情報）
        combined = torch.cat([seq_attn_pooled, max_pooled, cross_attn_out], dim=1)
        
        # 出力プロジェクション
        projected = self.output_proj(combined)
        
        # 最終出力
        output = self.output_layers(projected)
        
        return output


class HEATransformerLight(nn.Module):
    """
    軽量版Transformer（改善版）
    データ数が少ない場合用だが、精度向上のための改善を適用
    """
    def __init__(
        self,
        vocab_size: int = 20,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        max_seq_len: int = 20,
        additional_feat_dim: int = 8,
        dropout: float = 0.1
    ):
        super(HEATransformerLight, self).__init__()
        
        self.composition_embedding = CompositionEmbedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Attention pooling for better feature extraction
        self.attention_pooling = AttentionPooling(d_model)
        
        # より深い追加特徴量処理
        self.additional_mlp = nn.Sequential(
            nn.Linear(additional_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # 改善された出力層
        self.output_layers = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),  # attn_pool + max_pool + additional
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # 初期化
        self._init_weights()
    
    def _init_weights(self):
        """重みの初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        comp_values: torch.Tensor,
        attention_mask: torch.Tensor,
        additional_features: torch.Tensor
    ):
        src = self.composition_embedding(token_ids, comp_values)
        src = self.pos_encoder(src)
        
        key_padding_mask = (attention_mask == 0)
        encoded = self.transformer_encoder(src, src_key_padding_mask=key_padding_mask)
        
        # Attention-based pooling
        seq_attn_pooled = self.attention_pooling(encoded, attention_mask)
        
        # 最大プーリング
        max_pooled = encoded.max(dim=1)[0]
        
        # 追加特徴量
        additional_out = self.additional_mlp(additional_features)
        
        combined = torch.cat([seq_attn_pooled, max_pooled, additional_out], dim=1)
        output = self.output_layers(combined)
        
        return output
