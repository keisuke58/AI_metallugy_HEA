"""
Physics-Informed Neural Networks (PINNs) 実装
Raissi et al. (2019) "Physics-informed neural networks"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PINN(nn.Module):
    """
    Physics-Informed Neural Network
    物理法則を損失関数に組み込む
    """
    def __init__(
        self,
        input_dim: int = 72,  # 組成 + 材料記述子
        hidden_dims: list = [128, 128, 128, 128],
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 入力次元（組成 + 材料記述子）
            hidden_dims: 隠れ層サイズのリスト
            output_dim: 出力次元（弾性率）
            dropout: ドロップアウト率
        """
        super(PINN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),  # PINNではTanhがよく使われる
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
        # 重みの初期化（Xavier初期化）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim] 組成 + 材料記述子
        Returns:
            [batch, output_dim] 弾性率予測
        """
        return self.net(x)
    
    def physics_loss(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        物理法則に基づく損失関数
        
        材料科学における物理的制約:
        1. 弾性率は正の値である必要がある
        2. 組成の合計は1である必要がある（入力で保証）
        3. 弾性率と組成の間の関係（線形弾性理論など）
        
        Args:
            x: [batch, input_dim] 入力
            y_pred: [batch, 1] 予測値
        Returns:
            物理的損失
        """
        # 1. 弾性率は正の値である必要がある
        positivity_loss = F.relu(-y_pred).mean()
        
        # 2. 弾性率の範囲制約（30-500 GPa程度）
        # 範囲外の値にペナルティ
        lower_bound = 0.0
        upper_bound = 500.0
        range_loss = (
            F.relu(lower_bound - y_pred).mean() +
            F.relu(y_pred - upper_bound).mean()
        )
        
        # 3. 組成と弾性率の関係（簡易版）
        # 組成の重み付き平均が弾性率に影響を与える
        # ここでは簡易的な制約を追加
        
        return positivity_loss + 0.1 * range_loss
    
    def compute_physics_residual(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        物理的残差の計算（将来の拡張用）
        """
        # ここでは簡易的な実装
        # 実際のPINNでは、偏微分方程式の残差を計算
        return torch.zeros_like(y_pred)
