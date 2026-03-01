"""
モデル実装
"""

from .fno import FNO1d, FNO2d
from .deeponet import DeepONet
from .megnet import MEGNet
from .cgcnn import CGCNN
from .neural_ode import NeuralODE
from .pinns import PINN

__all__ = [
    'FNO1d',
    'FNO2d',
    'DeepONet',
    'MEGNet',
    'CGCNN',
    'NeuralODE',
    'PINN',
]
