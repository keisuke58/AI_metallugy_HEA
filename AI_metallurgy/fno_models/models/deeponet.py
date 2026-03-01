"""
Deep Operator Network (DeepONet) 実装
Lu et al. (2021) "Learning nonlinear operators via DeepONet"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchNet(nn.Module):
    """
    Branch Network: 入力関数（組成）を処理
    """
    def __init__(self, input_dim: int = 64, hidden_dims: list = [128, 128, 128], output_dim: int = 128):
        super(BranchNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim] (組成の空間化されたデータ)
        Returns:
            [batch, output_dim]
        """
        return self.net(x)


class TrunkNet(nn.Module):
    """
    Trunk Network: 出力関数の座標を処理
    """
    def __init__(self, input_dim: int = 1, hidden_dims: list = [128, 128, 128], output_dim: int = 128):
        super(TrunkNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: [batch, input_dim] (座標、この場合は材料記述子)
        Returns:
            [batch, output_dim]
        """
        return self.net(y)


class DeepONet(nn.Module):
    """
    Deep Operator Network
    組成から弾性率へのオペレーターを学習
    """
    def __init__(
        self,
        branch_input_dim: int = 64,
        trunk_input_dim: int = 8,
        branch_hidden_dims: list = [128, 128, 128],
        trunk_hidden_dims: list = [128, 128, 128],
        branch_output_dim: int = 128,
        trunk_output_dim: int = 128,
        additional_feat_dim: int = 8
    ):
        """
        Args:
            branch_input_dim: Branchネットワークの入力次元（組成グリッドのサイズ）
            trunk_input_dim: Trunkネットワークの入力次元（材料記述子）
            branch_hidden_dims: Branchネットワークの隠れ層サイズ
            trunk_hidden_dims: Trunkネットワークの隠れ層サイズ
            branch_output_dim: Branchネットワークの出力次元
            trunk_output_dim: Trunkネットワークの出力次元
            additional_feat_dim: 追加特徴量次元
        """
        super(DeepONet, self).__init__()
        
        assert branch_output_dim == trunk_output_dim, "Branch and Trunk output dimensions must match"
        
        self.branch_net = BranchNet(branch_input_dim, branch_hidden_dims, branch_output_dim)
        self.trunk_net = TrunkNet(trunk_input_dim, trunk_hidden_dims, trunk_output_dim)
        
        # 追加特徴量の処理
        self.additional_mlp = nn.Sequential(
            nn.Linear(additional_feat_dim, branch_output_dim),
            nn.LayerNorm(branch_output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 出力層
        # When additional_features is None: [operator_output, branch_out] = [branch_output_dim * 2]
        # When additional_features is provided: [operator_output, branch_out, additional_out] = [branch_output_dim * 3]
        # We'll handle both cases dynamically, but define for the case with additional_features
        self.output_layer = nn.Sequential(
            nn.Linear(branch_output_dim * 3, branch_output_dim * 2),
            nn.LayerNorm(branch_output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(branch_output_dim * 2, branch_output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(branch_output_dim, 1)
        )
    
    def forward(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
        additional_features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            branch_input: [batch, branch_input_dim] (組成の空間化データ)
            trunk_input: [batch, trunk_input_dim] (材料記述子)
            additional_features: [batch, additional_feat_dim]
        Returns:
            [batch, 1]
        """
        # Branchネットワーク
        branch_out = self.branch_net(branch_input)  # [batch, branch_output_dim]
        
        # Trunkネットワーク
        trunk_out = self.trunk_net(trunk_input)  # [batch, trunk_output_dim]
        
        # DeepONetの出力: branch_out * trunk_out (要素積)
        operator_output = branch_out * trunk_out  # [batch, output_dim]
        
        # 追加特徴量の処理
        if additional_features is not None:
            additional_out = self.additional_mlp(additional_features)
            combined = torch.cat([operator_output, branch_out, additional_out], dim=1)
        else:
            # When no additional features, pad with zeros to match expected input size
            additional_out = torch.zeros_like(branch_out)
            combined = torch.cat([operator_output, branch_out, additional_out], dim=1)
        
        # 最終出力
        output = self.output_layer(combined)
        
        return output
