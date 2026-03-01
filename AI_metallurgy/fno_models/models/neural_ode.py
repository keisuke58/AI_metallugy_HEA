"""
Neural Ordinary Differential Equations (Neural ODE) 実装
Chen et al. (2018) "Neural Ordinary Differential Equations"
"""
import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint
except ImportError:
    print("Warning: torchdiffeq not installed. Neural ODE will use a simple Euler method.")
    def odeint(func, y0, t, method='euler'):
        """簡易的なEuler法によるODE積分（torchdiffeqがない場合のフォールバック）"""
        y = [y0]
        for i in range(len(t) - 1):
            dt = t[i+1] - t[i]
            y_new = y[-1] + dt * func(t[i], y[-1])
            y.append(y_new)
        return torch.stack(y)


class ODEFunc(nn.Module):
    """
    ODE関数（連続的な動的システム）
    """
    def __init__(self, dim: int, hidden_dim: int = 128):
        """
        Args:
            dim: 状態次元
            hidden_dim: 隠れ層次元
        """
        super(ODEFunc, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        
        # 重みの初期化
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, t, y):
        """
        Args:
            t: 時間（この場合は使用しない）
            y: [batch, dim] 状態
        Returns:
            [batch, dim] 状態の時間微分
        """
        return self.net(y)


class NeuralODE(nn.Module):
    """
    Neural ODE Model
    組成や処理条件の連続的な変化をモデル化
    """
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        ode_dim: int = 64,
        additional_feat_dim: int = 8,
        num_ode_layers: int = 1,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 入力次元（組成の空間化データ）
            hidden_dim: 隠れ層次元
            ode_dim: ODE状態次元
            additional_feat_dim: 追加特徴量次元
            num_ode_layers: ODE層数（時間ステップ数）
            dropout: ドロップアウト率
        """
        super(NeuralODE, self).__init__()
        
        self.ode_dim = ode_dim
        self.num_ode_layers = num_ode_layers
        
        # 入力埋め込み
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ode_dim)
        )
        
        # ODE関数
        self.ode_func = ODEFunc(ode_dim, hidden_dim)
        
        # 追加特徴量の処理
        self.additional_mlp = nn.Sequential(
            nn.Linear(additional_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ode_dim)
        )
        
        # 出力層
        self.output_layers = nn.Sequential(
            nn.Linear(ode_dim + ode_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, additional_features: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim] 組成の空間化データ
            additional_features: [batch, additional_feat_dim]
        Returns:
            [batch, 1]
        """
        # 入力埋め込み
        y0 = self.input_embedding(x)  # [batch, ode_dim]
        
        # ODE積分
        # 時間ステップ: [0, 1, 2, ..., num_ode_layers]
        t = torch.linspace(0, self.num_ode_layers, self.num_ode_layers + 1, device=x.device)
        
        # ODEを解く
        y = odeint(self.ode_func, y0, t, method='euler')  # [num_steps, batch, ode_dim]
        
        # 最終状態を取得
        y_final = y[-1]  # [batch, ode_dim]
        
        # 追加特徴量の処理
        if additional_features is not None:
            additional_out = self.additional_mlp(additional_features)
            combined = torch.cat([y_final, additional_out], dim=1)
        else:
            combined = torch.cat([y_final, torch.zeros_like(y_final)], dim=1)
        
        # 出力
        output = self.output_layers(combined)
        
        return output
