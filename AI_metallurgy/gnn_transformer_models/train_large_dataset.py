"""
大規模データセット（5000+サンプル）用訓練スクリプト
大規模データに最適化されたハイパーパラメータと設定
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
import seaborn as sns
import math
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from data_loader import HEADataset, TransformerDataset, collate_gnn, collate_transformer
from gnn_model import HEAGNN, HEAGNNLight
from transformer_model import HEATransformer, HEATransformerLight

# デフォルト設定
_script_dir = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = _script_dir.parent / "data_collection" / "processed_data" / "data_with_features.csv"
DEFAULT_OUTPUT_DIR = _script_dir / "results_large"
DEFAULT_MODEL_DIR = _script_dir / "models_large"

# 大規模データセット用ハイパーパラメータ（5000+サンプル向け最適化）
LARGE_DATASET_CONFIG = {
    'batch_size': 64,  # 大規模データではより大きなバッチサイズが有効
    'learning_rate': 1e-3,  # より高い学習率で学習を促進（大規模データでは有効）
    'num_epochs': 500,  # より長い訓練
    'early_stopping_patience': 80,  # より長い忍耐（大規模データでは学習に時間がかかる）
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'device': 'auto',
    'use_light_model': False,  # フルモデルを使用（大規模データでは容量が重要）
    'use_huber_loss': True,  # Huber Lossを使用
    'huber_delta': 1.0,
    'weight_decay': 1e-4,  # 適度な正則化（学習を促進）
    'gradient_accumulation_steps': 2,  # 勾配累積で実質的なバッチサイズを増やす
    'warmup_epochs': 30,  # より長いウォームアップ
    'num_workers': 8,  # より多くのワーカー（大規模データ処理）
    'pin_memory': True,
    'persistent_workers': True,
    'normalize_target': True,
    'normalize_features': True,
    'transformer_dropout': 0.2,  # 適度な正則化（学習を促進）
    'transformer_num_layers': 4,  # 適度な深さ（学習を促進）
    'transformer_dim_feedforward': 512,  # 適度なサイズ（学習を促進）
    'gnn_hidden_dim': 256,  # より大きなGNN隠れ層
    'gnn_num_layers': 6,  # より深いGNN層
}


def train_epoch(model, dataloader, criterion, optimizer, device, gradient_accumulation_steps=1):
    """1エポックの訓練"""
    model.train()
    total_loss = 0.0
    predictions = []
    targets = []
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        if isinstance(batch, Batch):  # GNN
            batch = batch.to(device)
            output = model(batch)
            target = batch.y
        else:  # Transformer
            token_ids = batch['token_ids'].to(device)
            comp_values = batch['comp_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            additional_features = batch['additional_features'].to(device)
            target = batch['target'].to(device)
            
            output = model(token_ids, comp_values, attention_mask, additional_features)
        
        # 出力とターゲットの形状を一致させる
        if output.dim() > 1 and output.size(1) == 1:
            output = output.squeeze(1)
        if target.dim() > 1 and target.size(1) == 1:
            target = target.squeeze(1)
        
        loss = criterion(output, target)
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # 勾配累積の更新
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        predictions.extend(output.detach().cpu().numpy().flatten())
        targets.extend(target.detach().cpu().numpy().flatten())
    
    # 残りの勾配を更新
    if len(dataloader) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)
    r2 = r2_score(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    
    return avg_loss, r2, rmse, mae


def evaluate(model, dataloader, criterion, device):
    """評価"""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, Batch):  # GNN
                batch = batch.to(device)
                output = model(batch)
                target = batch.y
            else:  # Transformer
                token_ids = batch['token_ids'].to(device)
                comp_values = batch['comp_values'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                additional_features = batch['additional_features'].to(device)
                target = batch['target'].to(device)
                
                output = model(token_ids, comp_values, attention_mask, additional_features)
            
            if output.dim() == 2:
                output = output.squeeze(-1)
            if target.dim() == 2:
                target = target.squeeze(-1)
            
            assert output.shape == target.shape, f"Shape mismatch: output {output.shape} vs target {target.shape}"
            
            loss = criterion(output, target)
            total_loss += loss.item()
            predictions.extend(output.cpu().numpy().flatten())
            targets.extend(target.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(dataloader)
    r2 = r2_score(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    
    return avg_loss, r2, rmse, mae, predictions, targets


def train_gnn_large(config, data_path, output_dir, model_dir):
    """GNNモデルの訓練（大規模データセット用）"""
    print("=" * 80)
    print("GNN Model Training (Large Dataset Optimized)")
    print("=" * 80)
    
    normalize_target = config.get('normalize_target', True)
    normalize_features = config.get('normalize_features', True)
    dataset = HEADataset(str(data_path), normalize_target=normalize_target, normalize_features=normalize_features)
    
    if normalize_target:
        target_mean = dataset.target_mean
        target_std = dataset.target_std
        print(f"💾 ターゲット正規化: mean={target_mean:.4f}, std={target_std:.4f}")
    
    if normalize_features and hasattr(dataset, 'feature_scaler') and dataset.feature_scaler is not None:
        import pickle
        scaler_path = model_dir / "feature_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(dataset.feature_scaler, f)
        print(f"💾 特徴量スケーラーを保存: {scaler_path}")
    
    data_size = len(dataset)
    print(f"\n📊 総データ数: {data_size}")
    
    # 大規模データ用のバッチサイズ調整
    batch_size = config['batch_size']
    if data_size > 10000:
        batch_size = min(128, max(batch_size, data_size // 50))
    elif data_size > 5000:
        batch_size = min(96, max(batch_size, data_size // 60))
    
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
    
    # DataLoaderの設定（大規模データ最適化）
    num_workers = config.get('num_workers', 8) if data_size > 5000 else 4
    pin_memory = config.get('pin_memory', True) and torch.cuda.is_available()
    persistent_workers = config.get('persistent_workers', True) and num_workers > 0
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, collate_fn=collate_gnn,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, collate_fn=collate_gnn,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, collate_fn=collate_gnn,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    print(f"データ: {data_size}サンプル | 分割: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)} | バッチサイズ: {batch_size}")
    
    # 勾配累積
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 2)
    if data_size > 5000:
        gradient_accumulation_steps = max(2, min(4, batch_size // 32))
        print(f"💾 勾配累積ステップ数: {gradient_accumulation_steps}")
    
    # モデル（大規模データ用に大きく）
    model = HEAGNN(
        node_dim=5,
        edge_dim=3,
        hidden_dim=config.get('gnn_hidden_dim', 256),
        num_layers=config.get('gnn_num_layers', 6),
        num_heads=8,
        additional_feat_dim=8,
        dropout=0.2
    )
    
    device = torch.device(config['device'])
    model = model.to(device)
    
    # 出力層のバイアス初期化
    if normalize_target:
        target_mean_normalized = 0.0
    else:
        dataset_sample = HEADataset(str(data_path), normalize_target=False)
        if len(dataset_sample.df) > 0:
            target_mean_normalized = dataset_sample.df['elastic_modulus'].mean()
        else:
            target_mean_normalized = 150.0
    
    if hasattr(model, 'output_layers'):
        for layer in model.output_layers:
            if isinstance(layer, nn.Linear) and layer.out_features == 1:
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, float(target_mean_normalized))
                    break
    
    # 損失関数とオプティマイザ
    if config.get('use_huber_loss', False):
        criterion = nn.HuberLoss(delta=config.get('huber_delta', 1.0))
    else:
        criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config.get('weight_decay', 3e-4),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 学習率スケジューラー
    warmup_epochs = config.get('warmup_epochs', 30)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 1.0
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=10, verbose=True, min_lr=1e-7
    )
    
    # 訓練ループ
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_r2s = []
    val_r2s = []
    
    for epoch in range(config['num_epochs']):
        train_loss, train_r2, train_rmse, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, device, gradient_accumulation_steps
        )
        
        val_loss, val_r2, val_rmse, val_mae, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        warmup_scheduler.step()
        plateau_scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2s.append(train_r2)
        val_r2s.append(val_r2)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{config['num_epochs']} | LR: {current_lr:.2e} | "
              f"Train: Loss={train_loss:.2f}, R²={train_r2:.4f} | "
              f"Val: Loss={val_loss:.2f}, R²={val_r2:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_dir / "gnn_best_model.pth")
            print("  ✅ Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # テスト評価
    print("\n" + "=" * 80)
    print("Test Evaluation")
    print("=" * 80)
    model.load_state_dict(torch.load(model_dir / "gnn_best_model.pth"))
    test_loss, test_r2, test_rmse, test_mae, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device
    )
    
    if normalize_target and hasattr(dataset, 'target_mean') and hasattr(dataset, 'target_std'):
        test_preds_denorm = np.array(test_preds) * dataset.target_std + dataset.target_mean
        test_targets_denorm = np.array(test_targets) * dataset.target_std + dataset.target_mean
        test_r2 = r2_score(test_targets_denorm, test_preds_denorm)
        test_rmse = np.sqrt(mean_squared_error(test_targets_denorm, test_preds_denorm))
        test_mae = mean_absolute_error(test_targets_denorm, test_preds_denorm)
        test_preds = test_preds_denorm.tolist()
        test_targets = test_targets_denorm.tolist()
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f} GPa")
    print(f"Test MAE: {test_mae:.4f} GPa")
    
    results = {
        'model': 'GNN',
        'test_r2': float(test_r2),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_r2s': train_r2s,
        'val_r2s': val_r2s
    }
    
    with open(output_dir / "gnn_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def train_transformer_large(config, data_path, output_dir, model_dir):
    """Transformerモデルの訓練（大規模データセット用）"""
    print("=" * 80)
    print("Transformer Model Training (Large Dataset Optimized)")
    print("=" * 80)
    
    normalize_target = config.get('normalize_target', True)
    normalize_features = config.get('normalize_features', True)
    
    scaler_path = model_dir / "transformer_feature_scaler.pkl"
    dataset = TransformerDataset(
        str(data_path), 
        max_length=20,
        fit_scaler=normalize_features,
        scaler_path=str(scaler_path) if normalize_features else None,
        normalize_target=normalize_target
    )
    
    if normalize_target:
        target_mean = dataset.target_mean
        target_std = dataset.target_std
        print(f"💾 ターゲット正規化: mean={target_mean:.4f}, std={target_std:.4f}")
        norm_params = {'target_mean': float(target_mean), 'target_std': float(target_std)}
        with open(model_dir / "transformer_target_norm.json", 'w') as f:
            json.dump(norm_params, f, indent=2)
    
    if normalize_features and hasattr(dataset, 'scaler') and dataset.scaler is not None:
        print(f"💾 特徴量スケーラーを保存: {scaler_path}")
    
    data_size = len(dataset)
    print(f"\n📊 総データ数: {data_size}")
    
    # 大規模データ用のバッチサイズ調整
    batch_size = config['batch_size']
    if data_size > 10000:
        batch_size = min(128, max(batch_size, data_size // 50))
    elif data_size > 5000:
        batch_size = min(96, max(batch_size, data_size // 60))
    
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
    
    # DataLoaderの設定
    num_workers = config.get('num_workers', 8) if data_size > 5000 else 4
    pin_memory = config.get('pin_memory', True) and torch.cuda.is_available()
    persistent_workers = config.get('persistent_workers', True) and num_workers > 0
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, collate_fn=collate_transformer,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, collate_fn=collate_transformer,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, collate_fn=collate_transformer,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    print(f"データ: {data_size}サンプル | 分割: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)} | バッチサイズ: {batch_size}")
    
    # 勾配累積
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 2)
    if data_size > 5000:
        gradient_accumulation_steps = max(2, min(4, batch_size // 32))
        print(f"💾 勾配累積ステップ数: {gradient_accumulation_steps}")
    
    # モデル（大規模データ用に大きく・深く）
    model = HEATransformer(
        vocab_size=dataset.vocab_size,
        d_model=256,
        nhead=8,
        num_layers=config.get('transformer_num_layers', 6),
        dim_feedforward=config.get('transformer_dim_feedforward', 1024),
        max_seq_len=20,
        additional_feat_dim=8,
        dropout=config.get('transformer_dropout', 0.3)
    )
    
    device = torch.device(config['device'])
    model = model.to(device)
    
    # 出力層のバイアス初期化
    if normalize_target and hasattr(dataset, 'target_mean'):
        target_mean_normalized = 0.0
    else:
        if len(dataset.df) > 0:
            target_mean_normalized = dataset.df['elastic_modulus'].mean()
        else:
            target_mean_normalized = 150.0
    
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.out_features == 1:
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, float(target_mean_normalized))
                break
    
    # 損失関数とオプティマイザ
    if config.get('use_huber_loss', False):
        criterion = nn.HuberLoss(delta=config.get('huber_delta', 1.0))
    else:
        criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config.get('weight_decay', 3e-4),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 学習率スケジューラー
    warmup_epochs = config.get('warmup_epochs', 30)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            # より緩やかな減衰（大規模データでは長期的な学習が重要）
            progress = (epoch - warmup_epochs) / (config['num_epochs'] - warmup_epochs)
            return 0.2 + 0.6 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=10, verbose=True, min_lr=1e-7
    )
    
    # 訓練ループ
    best_val_loss = float('inf')
    best_val_r2 = float('-inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_r2s = []
    val_r2s = []
    
    for epoch in range(config['num_epochs']):
        train_loss, train_r2, train_rmse, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, device, gradient_accumulation_steps
        )
        
        val_loss, val_r2, val_rmse, val_mae, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        plateau_scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2s.append(train_r2)
        val_r2s.append(val_r2)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{config['num_epochs']} | LR: {current_lr:.2e} | "
              f"Train: Loss={train_loss:.2f}, R²={train_r2:.4f} | "
              f"Val: Loss={val_loss:.2f}, R²={val_r2:.4f}")
        
        # Early stopping（改善版：より柔軟な条件）
        improved = False
        # Val R²が改善した場合（閾値0.005以上、より緩和）
        if val_r2 > best_val_r2 + 0.005:
            best_val_r2 = val_r2
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_dir / "transformer_best_model.pth")
            print("  ✅ Best model saved! (R² improved)")
            improved = True
        # Val Lossが改善し、R²が大きく下がらない場合（90%以上維持、より緩和）
        elif val_loss < best_val_loss * 0.99 and val_r2 >= best_val_r2 * 0.90:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_dir / "transformer_best_model.pth")
            print("  ✅ Best model saved! (Loss improved, R² maintained)")
            improved = True
        # Val R²が負の値になった場合、早期警告
        elif val_r2 < 0 and epoch > 20:
            patience_counter += 2  # 負のR²は深刻なので、patienceを2倍カウント
        
        if not improved:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1} (best Val R²: {best_val_r2:.4f})")
                break
    
    # テスト評価
    print("\n" + "=" * 80)
    print("Test Evaluation")
    print("=" * 80)
    model.load_state_dict(torch.load(model_dir / "transformer_best_model.pth"))
    test_loss, test_r2, test_rmse, test_mae, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device
    )
    
    # デバッグ: 正規化された予測値とターゲットの統計
    print(f"\n[DEBUG] 正規化された値の統計:")
    print(f"  予測値: mean={np.array(test_preds).mean():.4f}, std={np.array(test_preds).std():.4f}, "
          f"min={np.array(test_preds).min():.4f}, max={np.array(test_preds).max():.4f}")
    print(f"  ターゲット: mean={np.array(test_targets).mean():.4f}, std={np.array(test_targets).std():.4f}, "
          f"min={np.array(test_targets).min():.4f}, max={np.array(test_targets).max():.4f}")
    print(f"  正規化パラメータ: mean={dataset.target_mean:.4f}, std={dataset.target_std:.4f}")
    
    if normalize_target and hasattr(dataset, 'target_mean') and hasattr(dataset, 'target_std'):
        test_preds_denorm = np.array(test_preds) * dataset.target_std + dataset.target_mean
        test_targets_denorm = np.array(test_targets) * dataset.target_std + dataset.target_mean
        
        # デバッグ: 非正規化後の統計
        print(f"\n[DEBUG] 非正規化後の値の統計:")
        print(f"  予測値: mean={test_preds_denorm.mean():.4f}, std={test_preds_denorm.std():.4f}, "
              f"min={test_preds_denorm.min():.4f}, max={test_preds_denorm.max():.4f}")
        print(f"  ターゲット: mean={test_targets_denorm.mean():.4f}, std={test_targets_denorm.std():.4f}, "
              f"min={test_targets_denorm.min():.4f}, max={test_targets_denorm.max():.4f}")
        
        test_r2 = r2_score(test_targets_denorm, test_preds_denorm)
        test_rmse = np.sqrt(mean_squared_error(test_targets_denorm, test_preds_denorm))
        test_mae = mean_absolute_error(test_targets_denorm, test_preds_denorm)
        test_preds = test_preds_denorm.tolist()
        test_targets = test_targets_denorm.tolist()
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f} GPa")
    print(f"Test MAE: {test_mae:.4f} GPa")
    
    results = {
        'model': 'Transformer',
        'test_r2': float(test_r2),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_r2s': train_r2s,
        'val_r2s': val_r2s
    }
    
    with open(output_dir / "transformer_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='Large Dataset Training (5000+ samples)')
    parser.add_argument('--data_path', type=str, default=str(DEFAULT_DATA_PATH),
                       help='Path to data CSV file')
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                       help='Output directory for results')
    parser.add_argument('--model_dir', type=str, default=str(DEFAULT_MODEL_DIR),
                       help='Directory for saved models')
    parser.add_argument('--model', type=str, choices=['gnn', 'transformer', 'both'],
                       default='both', help='Which model to train')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    
    return parser.parse_args()


def main():
    """メイン関数"""
    args = parse_args()
    
    # 設定を更新
    config = LARGE_DATASET_CONFIG.copy()
    
    # デバイス設定
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    config['device'] = device
    
    # パス設定
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    
    # ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # データファイルの確認
    if not data_path.exists():
        print(f"❌ データファイルが見つかりません: {data_path}")
        return
    
    # データ数を確認
    df = pd.read_csv(data_path)
    data_count = len(df[df['elastic_modulus'].notna()]) if 'elastic_modulus' in df.columns else len(df)
    print("=" * 80)
    print("Large Dataset Training (5000+ samples optimized)")
    print("=" * 80)
    print(f"📊 データファイル: {data_path}")
    print(f"📊 データ数: {data_count}")
    print(f"📁 出力ディレクトリ: {output_dir}")
    print(f"💾 モデルディレクトリ: {model_dir}")
    print(f"⚙️  デバイス: {config['device']}")
    print("=" * 80)
    
    if data_count < 5000:
        print(f"⚠️  警告: データ数が{data_count}で5000未満です。通常のtrain.pyを使用することを推奨します。")
        response = input("続行しますか？ (y/n): ")
        if response.lower() != 'y':
            return
    
    gnn_results = None
    transformer_results = None
    
    # GNN訓練
    if args.model in ['gnn', 'both']:
        gnn_results = train_gnn_large(config, data_path, output_dir, model_dir)
    
    # Transformer訓練
    if args.model in ['transformer', 'both']:
        transformer_results = train_transformer_large(config, data_path, output_dir, model_dir)
    
    # 結果比較
    if gnn_results and transformer_results:
        print("\n" + "=" * 80)
        print("Model Comparison")
        print("=" * 80)
        print(f"GNN - R²: {gnn_results['test_r2']:.4f}, RMSE: {gnn_results['test_rmse']:.4f} GPa")
        print(f"Transformer - R²: {transformer_results['test_r2']:.4f}, RMSE: {transformer_results['test_rmse']:.4f} GPa")
        
        comparison = {
            'gnn': gnn_results,
            'transformer': transformer_results
        }
        with open(output_dir / "model_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n✅ 結果を保存しました: {output_dir / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
