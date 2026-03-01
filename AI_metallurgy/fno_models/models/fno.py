"""
Fourier Neural Operator (FNO) 実装
Li et al. (2020) "Fourier Neural Operator for Parametric Partial Differential Equations"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """
    1次元スペクトル畳み込み層（FNOの核心）
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """
        Args:
            in_channels: 入力チャンネル数
            out_channels: 出力チャンネル数
            modes: Fourier modes数（低周波成分のみを保持）
        """
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # 複素数の重み行列（実部と虚部を別々に学習）
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.float32)
        )
        self.weights_imag = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.float32)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, in_channels, grid_size]
        Returns:
            [batch, out_channels, grid_size]
        """
        batch_size = x.size(0)
        grid_size = x.size(2)
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=2)  # [batch, in_channels, grid_size//2+1]
        
        # 低周波成分のみを処理（modes個）
        out_ft = torch.zeros(
            batch_size, self.out_channels, grid_size // 2 + 1,
            dtype=torch.complex64, device=x.device
        )
        
        # 低周波成分の処理
        # x_ft[:, :, :self.modes] has shape [batch, in_channels, modes]
        # weights have shape [in_channels, out_channels, modes]
        x_ft_modes = x_ft[:, :, :self.modes]  # [batch, in_channels, modes]
        out_ft[:, :, :self.modes] = torch.complex(
            torch.einsum('bim,iom->bom', x_ft_modes.real, self.weights_real) -
            torch.einsum('bim,iom->bom', x_ft_modes.imag, self.weights_imag),
            torch.einsum('bim,iom->bom', x_ft_modes.real, self.weights_imag) +
            torch.einsum('bim,iom->bom', x_ft_modes.imag, self.weights_real)
        )
        
        # IFFT
        x = torch.fft.irfft(out_ft, n=grid_size, dim=2)
        
        return x


class FNOBlock1d(nn.Module):
    """
    1次元FNOブロック
    """
    def __init__(self, modes: int, width: int):
        super(FNOBlock1d, self).__init__()
        self.modes = modes
        self.width = width
        
        # スペクトル畳み込み
        self.conv = SpectralConv1d(width, width, modes)
        
        # 線形変換
        self.w = nn.Linear(width, width)
        
        # バッチ正規化
        self.bn = nn.BatchNorm1d(width)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, width, grid_size]
        Returns:
            [batch, width, grid_size]
        """
        # スペクトル畳み込み
        x1 = self.conv(x)
        
        # 線形変換
        x2 = self.w(x.transpose(1, 2)).transpose(1, 2)
        
        # 残差接続
        x = x1 + x2
        x = self.bn(x)
        x = F.gelu(x)
        
        return x


class FNO1d(nn.Module):
    """
    1次元Fourier Neural Operator
    組成データの弾性率予測用
    """
    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        layers: int = 4,
        input_channels: int = 2,
        additional_feat_dim: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            modes: Fourier modes数
            width: チャンネル数
            layers: FNO層数
            input_channels: 入力チャンネル数（組成 + 記述子）
            additional_feat_dim: 追加特徴量次元
            dropout: ドロップアウト率
        """
        super(FNO1d, self).__init__()
        
        self.modes = modes
        self.width = width
        self.layers = layers
        
        # 入力埋め込み
        self.input_projection = nn.Linear(input_channels, width)
        
        # FNOブロック
        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(modes, width) for _ in range(layers)
        ])
        
        # 追加特徴量の処理
        self.additional_mlp = nn.Sequential(
            nn.Linear(additional_feat_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width)
        )
        
        # 出力層
        self.output_projection = nn.Sequential(
            nn.Linear(width + width, width * 2),  # FNO出力 + 追加特徴量
            nn.LayerNorm(width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1)
        )
    
    def forward(self, x: torch.Tensor, additional_features: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch, input_channels, grid_size]
            additional_features: [batch, additional_feat_dim]
        Returns:
            [batch, 1]
        """
        # 入力埋め込み
        x = x.transpose(1, 2)  # [batch, grid_size, input_channels]
        x = self.input_projection(x)  # [batch, grid_size, width]
        x = x.transpose(1, 2)  # [batch, width, grid_size]
        
        # FNOブロック
        for fno_block in self.fno_blocks:
            x = fno_block(x)
        
        # グローバル平均プーリング
        x = x.mean(dim=2)  # [batch, width]
        
        # 追加特徴量の処理
        if additional_features is not None:
            additional_out = self.additional_mlp(additional_features)
            x = torch.cat([x, additional_out], dim=1)  # [batch, width * 2]
        else:
            x = torch.cat([x, torch.zeros_like(x)], dim=1)
        
        # 出力
        output = self.output_projection(x)
        
        return output


class FNO2d(nn.Module):
    """
    2次元FNO（将来の拡張用）
    """
    def __init__(self, modes: int = 16, width: int = 64, layers: int = 4):
        super(FNO2d, self).__init__()
        # 2次元実装は必要に応じて追加
        raise NotImplementedError("2D FNO is not implemented yet")
