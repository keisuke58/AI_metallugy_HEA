"""
統合訓練スクリプト
すべてのモデルを訓練可能
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# パスを追加
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

# モデル
from models import FNO1d, DeepONet, MEGNet, CGCNN, NeuralODE, PINN

# データローダー
from data_loaders import (
    FNODataset, collate_fno,
    DeepONetDataset, collate_deeponet,
    GraphDataset, collate_graph,
    NeuralODEDataset, collate_neural_ode,
    PINNsDataset, collate_pinns
)

# デフォルト設定
# 統一・クリーンアップ済みの最新データセットを使用
DEFAULT_DATA_PATH = _script_dir.parent / "data_collection" / "final_data" / "unified_dataset_latest.csv"
DEFAULT_OUTPUT_DIR = _script_dir / "results"
DEFAULT_MODEL_DIR = _script_dir / "checkpoints"

DEFAULT_CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-3,
    'num_epochs': 200,
    'early_stopping_patience': 30,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'device': 'cuda',  # デフォルトはGPU
    'use_huber_loss': True,
    'huber_delta': 1.0,
    'weight_decay': 1e-4,
}


def train_epoch(model, dataloader, criterion, optimizer, device, model_type):
    """1エポックの訓練"""
    model.train()
    total_loss = 0.0
    predictions = []
    targets = []
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        if model_type in ['fno', 'neural_ode']:
            input_data = batch['input'].to(device)
            additional_features = batch.get('additional_features', None)
            if additional_features is not None:
                additional_features = additional_features.to(device)
            target = batch['target'].to(device)
            
            if model_type == 'fno':
                output = model(input_data, additional_features)
            else:  # neural_ode
                output = model(input_data, additional_features)
        
        elif model_type == 'deeponet':
            branch_input = batch['branch_input'].to(device)
            trunk_input = batch['trunk_input'].to(device)
            target = batch['target'].to(device)
            output = model(branch_input, trunk_input)
        
        elif model_type in ['megnet', 'cgcnn']:
            batch_data = batch.to(device)
            target = batch_data.y
            output = model(batch_data)
        
        elif model_type == 'pinns':
            input_data = batch['input'].to(device)
            target = batch['target'].to(device)
            output = model(input_data)
        
        # 損失計算
        if output.dim() > 1 and output.size(1) == 1:
            output = output.squeeze(1)
        if target.dim() > 1 and target.size(1) == 1:
            target = target.squeeze(1)
        
        loss = criterion(output, target)
        
        # PINNsの場合は物理的損失も追加
        if model_type == 'pinns':
            physics_loss = model.physics_loss(input_data, output.unsqueeze(1))
            loss = loss + 0.1 * physics_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(output.detach().cpu().numpy().flatten())
        targets.extend(target.detach().cpu().numpy().flatten())
    
    avg_loss = total_loss / len(dataloader)
    r2 = r2_score(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    
    return avg_loss, r2, rmse, mae


def evaluate(model, dataloader, criterion, device, model_type):
    """評価"""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if model_type in ['fno', 'neural_ode']:
                input_data = batch['input'].to(device)
                additional_features = batch.get('additional_features', None)
                if additional_features is not None:
                    additional_features = additional_features.to(device)
                target = batch['target'].to(device)
                
                if model_type == 'fno':
                    output = model(input_data, additional_features)
                else:
                    output = model(input_data, additional_features)
            
            elif model_type == 'deeponet':
                branch_input = batch['branch_input'].to(device)
                trunk_input = batch['trunk_input'].to(device)
                target = batch['target'].to(device)
                output = model(branch_input, trunk_input)
            
            elif model_type in ['megnet', 'cgcnn']:
                batch_data = batch.to(device)
                target = batch_data.y
                output = model(batch_data)
            
            elif model_type == 'pinns':
                input_data = batch['input'].to(device)
                target = batch['target'].to(device)
                output = model(input_data)
            
            if output.dim() > 1 and output.size(1) == 1:
                output = output.squeeze(1)
            if target.dim() > 1 and target.size(1) == 1:
                target = target.squeeze(1)
            
            loss = criterion(output, target)
            total_loss += loss.item()
            predictions.extend(output.cpu().numpy().flatten())
            targets.extend(target.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(dataloader)
    r2 = r2_score(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    
    return avg_loss, r2, rmse, mae, predictions, targets


def train_model(model_type: str, config: dict, data_path: Path, output_dir: Path, model_dir: Path):
    """モデルを訓練"""
    print("=" * 80)
    print(f"{model_type.upper()} Model Training")
    print("=" * 80)
    
    # データローダーの準備
    if model_type == 'fno':
        dataset = FNODataset(str(data_path), grid_size=64)
        collate_fn = collate_fno
    elif model_type == 'deeponet':
        dataset = DeepONetDataset(str(data_path), grid_size=64)
        collate_fn = collate_deeponet
    elif model_type in ['megnet', 'cgcnn']:
        dataset = GraphDataset(str(data_path))
        collate_fn = collate_graph
    elif model_type == 'neural_ode':
        dataset = NeuralODEDataset(str(data_path), grid_size=64)
        collate_fn = collate_neural_ode
    elif model_type == 'pinns':
        dataset = PINNsDataset(str(data_path), max_elements=17)
        collate_fn = collate_pinns
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # データ分割
    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(
        indices, test_size=1 - config['train_ratio'], random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=config['test_ratio'] / (config['val_ratio'] + config['test_ratio']),
        random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    print(f"📊 データ分割: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # モデルの初期化
    device = torch.device(config['device'])
    
    if model_type == 'fno':
        model = FNO1d(modes=16, width=64, layers=4, input_channels=2, additional_feat_dim=8)
    elif model_type == 'deeponet':
        model = DeepONet(branch_input_dim=64, trunk_input_dim=8, branch_output_dim=128, trunk_output_dim=128)
    elif model_type == 'megnet':
        model = MEGNet(node_dim=5, edge_dim=3, state_dim=64, hidden_dim=128, num_layers=3, additional_feat_dim=29)
    elif model_type == 'cgcnn':
        model = CGCNN(node_dim=5, edge_dim=3, hidden_dim=128, num_layers=3, additional_feat_dim=29)
    elif model_type == 'neural_ode':
        model = NeuralODE(input_dim=64, hidden_dim=128, ode_dim=64, additional_feat_dim=8, num_ode_layers=3)
    elif model_type == 'pinns':
        model = PINN(input_dim=25, hidden_dims=[128, 128, 128, 128])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # 損失関数とオプティマイザ
    if config.get('use_huber_loss', False):
        criterion = nn.HuberLoss(delta=config.get('huber_delta', 1.0))
    else:
        criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 1e-4))
    # PyTorchバージョン互換のため verbose は使わない
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 訓練ループ
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_r2s = []
    val_r2s = []
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        train_loss, train_r2, train_rmse, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, device, model_type
        )
        
        val_loss, val_r2, val_rmse, val_mae, _, _ = evaluate(
            model, val_loader, criterion, device, model_type
        )
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2s.append(train_r2)
        val_r2s.append(val_r2)
        
        print(f"Train Loss: {train_loss:.4f}, R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
        print(f"Val Loss: {val_loss:.4f}, R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_dir / f"{model_type}_best_model.pth")
            print("✅ Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # テスト評価
    print("\n" + "=" * 80)
    print("Test Evaluation")
    print("=" * 80)
    model.load_state_dict(torch.load(model_dir / f"{model_type}_best_model.pth"))
    test_loss, test_r2, test_rmse, test_mae, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device, model_type
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f} GPa")
    print(f"Test MAE: {test_mae:.4f} GPa")
    
    # 結果を保存
    results = {
        'model': model_type,
        'test_r2': float(test_r2),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_r2s': train_r2s,
        'val_r2s': val_r2s
    }
    
    with open(output_dir / f"{model_type}_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # 可視化
    plot_results(test_targets, test_preds, train_losses, val_losses, train_r2s, val_r2s, model_type, output_dir)
    
    return results


def plot_results(targets, predictions, train_losses, val_losses, train_r2s, val_r2s, model_name, output_dir):
    """結果を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 予測 vs 実測
    axes[0, 0].scatter(targets, predictions, alpha=0.6)
    axes[0, 0].plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Elastic Modulus (GPa)')
    axes[0, 0].set_ylabel('Predicted Elastic Modulus (GPa)')
    axes[0, 0].set_title(f'{model_name}: Predicted vs Actual')
    r2 = r2_score(targets, predictions)
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes, verticalalignment='top')
    
    # 残差プロット
    residuals = np.array(predictions) - np.array(targets)
    axes[0, 1].scatter(targets, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Actual Elastic Modulus (GPa)')
    axes[0, 1].set_ylabel('Residuals (GPa)')
    axes[0, 1].set_title(f'{model_name}: Residuals')
    
    # 損失曲線
    axes[1, 0].plot(train_losses, label='Train Loss')
    axes[1, 0].plot(val_losses, label='Val Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title(f'{model_name}: Training Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # R²曲線
    axes[1, 1].plot(train_r2s, label='Train R²')
    axes[1, 1].plot(val_r2s, label='Val R²')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].set_title(f'{model_name}: R² Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_results.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Train Neural Operator Models')
    parser.add_argument('--model', type=str, choices=['fno', 'deeponet', 'megnet', 'cgcnn', 'neural_ode', 'pinns', 'all'],
                       default='all', help='Model to train')
    parser.add_argument('--data_path', type=str, default=str(DEFAULT_DATA_PATH), help='Path to data CSV')
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR), help='Output directory')
    parser.add_argument('--model_dir', type=str, default=str(DEFAULT_MODEL_DIR), help='Model directory')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_CONFIG['learning_rate'])
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_CONFIG['num_epochs'])
    # デフォルトはGPUを優先
    parser.add_argument('--device', type=str, default='cuda', choices=['auto', 'cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # 設定
    config = DEFAULT_CONFIG.copy()

    # デバイス設定（デフォルトGPU）
    if args.device == 'auto':
        # auto の場合のみ、自動判定
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = args.device

    # GPU指定だがCUDAが無い場合はエラーにする
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("❌ CUDA が利用できません。CPU で実行する場合は --device cpu を指定してください。")
        return

    config.update({
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'device': device_str,
    })
    
    # パス設定
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        print(f"❌ データファイルが見つかりません: {data_path}")
        return
    
    # モデルリスト
    models_to_train = ['fno', 'deeponet', 'megnet', 'cgcnn', 'neural_ode', 'pinns'] if args.model == 'all' else [args.model]
    
    # 訓練
    all_results = {}
    for model_type in models_to_train:
        try:
            results = train_model(model_type, config, data_path, output_dir, model_dir)
            all_results[model_type] = results
        except Exception as e:
            print(f"❌ {model_type}の訓練中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
    
    # 結果比較
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("Model Comparison")
        print("=" * 80)
        for model_type, results in all_results.items():
            print(f"{model_type.upper()}: R²={results['test_r2']:.4f}, RMSE={results['test_rmse']:.4f} GPa")
        
        with open(output_dir / "all_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
