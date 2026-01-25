"""
訓練スクリプト: GNNとTransformerモデルを訓練
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
# train.pyの場所を基準にパスを設定
_script_dir = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = _script_dir.parent / "data_collection" / "processed_data" / "data_with_features.csv"
DEFAULT_OUTPUT_DIR = _script_dir / "results"
DEFAULT_MODEL_DIR = _script_dir / "models"

# デフォルトハイパーパラメータ（精度向上のため最適化）
DEFAULT_CONFIG = {
    'batch_size': 32,  # より安定した訓練のため32に調整
    'learning_rate': 3e-4,  # より低い学習率で安定化（過学習抑制）
    'num_epochs': 400,  # より長い訓練で精度向上
    'early_stopping_patience': 50,  # より長い忍耐（不安定な訓練に対応）
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'device': 'auto',
    'use_light_model': False,  # フルモデルを使用（精度向上のため）
    'use_huber_loss': True,  # Huber Lossを使用（外れ値に頑健、精度向上）
    'huber_delta': 1.0,  # Huber Lossのデルタ
    'weight_decay': 2e-4,  # より強い正則化で過学習防止
    'gradient_accumulation_steps': 1,  # 勾配累積
    'warmup_epochs': 25,  # ウォームアップ期間を延長
    'num_workers': 4,  # データローダーのワーカー数（マルチプロセッシング）
    'pin_memory': True,  # GPU転送の高速化
    'persistent_workers': True,  # ワーカーの永続化（データ数が多い場合に有効）
    'normalize_target': True,  # ターゲット変数の正規化（精度向上のため推奨）
    'normalize_features': True,  # 追加特徴量の正規化（重要！精度向上に必須）
    'transformer_dropout': 0.25,  # TransformerのDropout率（過学習抑制）
    'transformer_num_layers': 5,  # より深いTransformer層
    'transformer_dim_feedforward': 1024,  # より大きなフィードフォワード層
}

# 後方互換性のためのエイリアス
DATA_PATH = DEFAULT_DATA_PATH
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
MODEL_DIR = DEFAULT_MODEL_DIR
CONFIG = DEFAULT_CONFIG.copy()


def train_epoch(model, dataloader, criterion, optimizer, device, gradient_accumulation_steps=1):
    """1エポックの訓練（改善版）"""
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
        # 勾配累積
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # 勾配累積の更新
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # 勾配クリッピング
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
            
            # 出力とターゲットの形状を一致させる
            if output.dim() == 2:
                output = output.squeeze(-1)  # (batch_size, 1) -> (batch_size,)
            if target.dim() == 2:
                target = target.squeeze(-1)  # (batch_size, 1) -> (batch_size,)
            
            # 形状が一致していることを確認
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


def train_gnn(config, data_path, output_dir, model_dir):
    """GNNモデルの訓練"""
    print("=" * 80)
    print("GNN Model Training")
    print("=" * 80)
    
    # データセット（ターゲット正規化と特徴量正規化を有効化）
    normalize_target = config.get('normalize_target', True)  # デフォルトで有効
    normalize_features = config.get('normalize_features', True)  # デフォルトで有効（重要！）
    dataset = HEADataset(str(data_path), normalize_target=normalize_target, normalize_features=normalize_features)
    
    # 正規化パラメータを保存（推論時に使用）
    if normalize_target:
        target_mean = dataset.target_mean
        target_std = dataset.target_std
        print(f"💾 ターゲット正規化パラメータを保存: mean={target_mean:.4f}, std={target_std:.4f}")
    
    if normalize_features and hasattr(dataset, 'feature_scaler') and dataset.feature_scaler is not None:
        # 特徴量スケーラーを保存
        import pickle
        scaler_path = model_dir / "feature_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(dataset.feature_scaler, f)
        print(f"💾 特徴量スケーラーを保存: {scaler_path}")
    
    # データ数に応じてバッチサイズを自動調整
    data_size = len(dataset)
    batch_size = config['batch_size']
    
    if data_size > 2000:
        batch_size = min(128, max(batch_size, data_size // 40))
    elif data_size > 1000:
        batch_size = min(64, max(batch_size, data_size // 50))
    elif data_size < 200:
        batch_size = min(batch_size, max(8, data_size // 10))
    
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
    
    # DataLoaderの設定（データ数が多い場合の最適化）
    # 5000データの場合、num_workersを有効化して高速化
    num_workers = config.get('num_workers', 4) if data_size > 500 else 0
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
    
    # データ数が多い場合、勾配累積を有効化（メモリ効率向上）
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    if data_size > 2000 and gradient_accumulation_steps == 1:
        # データ数が非常に多い場合、勾配累積を推奨
        gradient_accumulation_steps = max(1, min(4, batch_size // 32))
        print(f"💾 データ数が多いため、勾配累積ステップ数を {gradient_accumulation_steps} に設定しました")
    
    # モデル
    if config['use_light_model']:
        model = HEAGNNLight(
            node_dim=5,
            edge_dim=3,
            hidden_dim=64,
            num_layers=3,
            additional_feat_dim=8,
            dropout=0.1
        )
    else:
        model = HEAGNN(
            node_dim=5,
            edge_dim=3,
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            additional_feat_dim=8,
            dropout=0.1
        )
    
    device = torch.device(config['device'])
    model = model.to(device)
    
    # モデルの重み初期化を改善
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # Xavier初期化（より安定）
            torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                # 出力層のバイアスはターゲットの平均値に近い値で初期化
                if hasattr(m, 'out_features') and m.out_features == 1:
                    # データの平均値を取得（簡易版：100-200 GPaを想定）
                    torch.nn.init.constant_(m.bias, 150.0)
                else:
                    torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # 出力層のバイアスを初期化（正規化されている場合は0、されていない場合は平均値）
    if normalize_target:
        # 正規化されている場合、ターゲットの平均は0なのでバイアスも0で初期化
        target_mean_normalized = 0.0
    else:
        # 正規化されていない場合、生の平均値を使用
        dataset_sample = HEADataset(str(data_path), normalize_target=False)
        if len(dataset_sample.df) > 0:
            target_mean_normalized = dataset_sample.df['elastic_modulus'].mean()
        else:
            target_mean_normalized = 150.0  # フォールバック
    
    # 出力層のバイアスを設定
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
        weight_decay=config.get('weight_decay', 1e-4),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 学習率スケジューラー（Warmup + ReduceLROnPlateau）
    warmup_epochs = config.get('warmup_epochs', 20)
    total_steps = len(train_loader) * config['num_epochs']
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # ウォームアップ：線形に増加
            return (epoch + 1) / warmup_epochs
        else:
            # その後はReduceLROnPlateauに任せる
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
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        # 訓練
        train_loss, train_r2, train_rmse, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, device, gradient_accumulation_steps
        )
        
        # 検証
        val_loss, val_r2, val_rmse, val_mae, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # 学習率スケジューラー更新
        warmup_scheduler.step()
        plateau_scheduler.step(val_loss)
        
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
            torch.save(model.state_dict(), model_dir / "gnn_best_model.pth")
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
    model.load_state_dict(torch.load(model_dir / "gnn_best_model.pth"))
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
    
    # ターゲットが正規化されている場合、元のスケールに戻す
    if normalize_target and hasattr(dataset, 'target_mean') and hasattr(dataset, 'target_std'):
        test_preds_denorm = np.array(test_preds) * dataset.target_std + dataset.target_mean
        test_targets_denorm = np.array(test_targets) * dataset.target_std + dataset.target_mean
        
        # デバッグ: 非正規化後の統計
        print(f"\n[DEBUG] 非正規化後の値の統計:")
        print(f"  予測値: mean={test_preds_denorm.mean():.4f}, std={test_preds_denorm.std():.4f}, "
              f"min={test_preds_denorm.min():.4f}, max={test_preds_denorm.max():.4f}")
        print(f"  ターゲット: mean={test_targets_denorm.mean():.4f}, std={test_targets_denorm.std():.4f}, "
              f"min={test_targets_denorm.min():.4f}, max={test_targets_denorm.max():.4f}")
        
        # 元のスケールでメトリクスを再計算
        test_r2 = r2_score(test_targets_denorm, test_preds_denorm)
        test_rmse = np.sqrt(mean_squared_error(test_targets_denorm, test_preds_denorm))
        test_mae = mean_absolute_error(test_targets_denorm, test_preds_denorm)
        test_preds = test_preds_denorm.tolist()
        test_targets = test_targets_denorm.tolist()
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f} GPa")
    print(f"Test MAE: {test_mae:.4f} GPa")
    
    # 結果を保存
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
    
    # 可視化
    plot_results(test_targets, test_preds, train_losses, val_losses, 
                 train_r2s, val_r2s, "GNN", output_dir)
    
    return results


def train_transformer(config, data_path, output_dir, model_dir):
    """Transformerモデルの訓練"""
    print("=" * 80)
    print("Transformer Model Training")
    print("=" * 80)
    
    # データセット（ターゲット正規化と特徴量正規化を有効化）
    normalize_target = config.get('normalize_target', True)  # デフォルトで有効
    normalize_features = config.get('normalize_features', True)  # デフォルトで有効（重要！）
    
    # 特徴量スケーラーのパス
    scaler_path = model_dir / "transformer_feature_scaler.pkl"
    dataset = TransformerDataset(
        str(data_path), 
        max_length=20,
        fit_scaler=normalize_features,
        scaler_path=str(scaler_path) if normalize_features else None,
        normalize_target=normalize_target
    )
    
    # 正規化パラメータを保存（推論時に使用）
    if normalize_target:
        target_mean = dataset.target_mean
        target_std = dataset.target_std
        print(f"💾 ターゲット正規化パラメータを保存: mean={target_mean:.4f}, std={target_std:.4f}")
        # 正規化パラメータを保存
        norm_params = {'target_mean': float(target_mean), 'target_std': float(target_std)}
        with open(model_dir / "transformer_target_norm.json", 'w') as f:
            json.dump(norm_params, f, indent=2)
    
    if normalize_features and hasattr(dataset, 'scaler') and dataset.scaler is not None:
        print(f"💾 特徴量スケーラーを保存: {scaler_path}")
    
    # データ数に応じてバッチサイズを自動調整
    data_size = len(dataset)
    batch_size = config['batch_size']
    
    if data_size > 2000:
        batch_size = min(128, max(batch_size, data_size // 40))
    elif data_size > 1000:
        batch_size = min(64, max(batch_size, data_size // 50))
    elif data_size < 200:
        batch_size = min(batch_size, max(8, data_size // 10))
    
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
    
    # DataLoaderの設定（データ数が多い場合の最適化）
    # 5000データの場合、num_workersを有効化して高速化
    num_workers = config.get('num_workers', 4) if data_size > 500 else 0
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
    
    # データ数が多い場合、勾配累積を有効化（メモリ効率向上）
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    if data_size > 2000 and gradient_accumulation_steps == 1:
        # データ数が非常に多い場合、勾配累積を推奨
        gradient_accumulation_steps = max(1, min(4, batch_size // 32))
        print(f"💾 データ数が多いため、勾配累積ステップ数を {gradient_accumulation_steps} に設定しました")
    
    # モデル（精度向上のため最適化）
    if config['use_light_model']:
        model = HEATransformerLight(
            vocab_size=dataset.vocab_size,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_feedforward=256,
            max_seq_len=20,
            additional_feat_dim=8,
            dropout=0.1
        )
    else:
        # フルモデル（最適化された設定 - より深く、より大きなモデル）
        model = HEATransformer(
            vocab_size=dataset.vocab_size,
            d_model=256,
            nhead=8,
            num_layers=config.get('transformer_num_layers', 5),  # より深い層
            dim_feedforward=config.get('transformer_dim_feedforward', 1024),  # より大きなフィードフォワード層
            max_seq_len=20,
            additional_feat_dim=8,
            dropout=config.get('transformer_dropout', 0.25)  # より強い正則化
        )
    
    device = torch.device(config['device'])
    model = model.to(device)
    
    # 出力層のバイアスを初期化（正規化されている場合は0、されていない場合は平均値）
    if normalize_target and hasattr(dataset, 'target_mean'):
        # 正規化されたターゲットの平均は0なので、バイアスも0で初期化
        target_mean_normalized = 0.0
    else:
        # 正規化されていない場合、生の平均値を使用
        if len(dataset.df) > 0:
            target_mean_normalized = dataset.df['elastic_modulus'].mean()
        else:
            target_mean_normalized = 150.0  # フォールバック
    
    # 出力層の最後のLinear層のバイアスを設定
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.out_features == 1:
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, float(target_mean_normalized))
                break
    
    # 損失関数（Huber Loss: 外れ値に頑健）
    if config.get('use_huber_loss', False):
        criterion = nn.HuberLoss(delta=config.get('huber_delta', 1.0))
    else:
        criterion = nn.MSELoss()
    
    # オプティマイザ（改善された設定）
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config.get('weight_decay', 2e-4),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 学習率スケジューラー（改善版：より安定した学習率）
    # CosineAnnealingLR with warmup + より積極的な減衰
    total_steps = len(train_loader) * config['num_epochs']
    warmup_epochs = config.get('warmup_epochs', 25)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # ウォームアップ：線形に増加
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing with restarts（より滑らかな減衰）
            progress = (epoch - warmup_epochs) / (config['num_epochs'] - warmup_epochs)
            # より積極的な減衰（最小値0.1まで）
            return 0.1 + 0.4 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ReduceLROnPlateauも併用（より積極的な減衰）
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
    
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    global_step = 0
    
    for epoch in range(config['num_epochs']):
        # 訓練
        train_loss, train_r2, train_rmse, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, device, gradient_accumulation_steps
        )
        global_step += len(train_loader)
        
        # 検証
        val_loss, val_r2, val_rmse, val_mae, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # 学習率スケジューラー更新
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
        
        # Early stopping（改善版：R²の安定性を重視）
        improved = False
        # Val R²が改善した場合（閾値0.01以上）
        if val_r2 > best_val_r2 + 0.01:
            best_val_r2 = val_r2
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_dir / "transformer_best_model.pth")
            print("  ✅ Best model saved! (R² improved)")
            improved = True
        # Val Lossが改善し、R²が大きく下がらない場合（95%以上維持）
        elif val_loss < best_val_loss * 0.98 and val_r2 >= best_val_r2 * 0.95:
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
    
    # ターゲットが正規化されている場合、元のスケールに戻す
    if normalize_target and hasattr(dataset, 'target_mean') and hasattr(dataset, 'target_std'):
        test_preds_denorm = np.array(test_preds) * dataset.target_std + dataset.target_mean
        test_targets_denorm = np.array(test_targets) * dataset.target_std + dataset.target_mean
        
        # デバッグ: 非正規化後の統計
        print(f"\n[DEBUG] 非正規化後の値の統計:")
        print(f"  予測値: mean={test_preds_denorm.mean():.4f}, std={test_preds_denorm.std():.4f}, "
              f"min={test_preds_denorm.min():.4f}, max={test_preds_denorm.max():.4f}")
        print(f"  ターゲット: mean={test_targets_denorm.mean():.4f}, std={test_targets_denorm.std():.4f}, "
              f"min={test_targets_denorm.min():.4f}, max={test_targets_denorm.max():.4f}")
        
        # 元のスケールでメトリクスを再計算
        test_r2 = r2_score(test_targets_denorm, test_preds_denorm)
        test_rmse = np.sqrt(mean_squared_error(test_targets_denorm, test_preds_denorm))
        test_mae = mean_absolute_error(test_targets_denorm, test_preds_denorm)
        test_preds = test_preds_denorm.tolist()
        test_targets = test_targets_denorm.tolist()
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f} GPa")
    print(f"Test MAE: {test_mae:.4f} GPa")
    
    # 結果を保存
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
    
    # 可視化
    plot_results(test_targets, test_preds, train_losses, val_losses, 
                 train_r2s, val_r2s, "Transformer", output_dir)
    
    return results


def plot_results(targets, predictions, train_losses, val_losses, 
                 train_r2s, val_r2s, model_name, output_dir):
    """結果を可視化"""
    # リストをnumpy配列に変換
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 予測 vs 実測
    axes[0, 0].scatter(targets, predictions, alpha=0.6)
    axes[0, 0].plot([targets.min(), targets.max()], 
                    [targets.min(), targets.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Elastic Modulus (GPa)')
    axes[0, 0].set_ylabel('Predicted Elastic Modulus (GPa)')
    axes[0, 0].set_title(f'{model_name}: Predicted vs Actual')
    r2 = r2_score(targets, predictions)
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', 
                    transform=axes[0, 0].transAxes, verticalalignment='top')
    
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
    plt.savefig(output_dir / f"{model_name.lower()}_results.png", dpi=300, bbox_inches='tight')
    plt.close()


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='HEA Elastic Modulus Prediction: GNN & Transformer Training')
    parser.add_argument('--data_path', type=str, default=str(DEFAULT_DATA_PATH),
                       help='Path to data CSV file')
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                       help='Output directory for results')
    parser.add_argument('--model_dir', type=str, default=str(DEFAULT_MODEL_DIR),
                       help='Directory for saved models')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_CONFIG['learning_rate'],
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_CONFIG['num_epochs'],
                       help='Number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=DEFAULT_CONFIG['early_stopping_patience'],
                       help='Early stopping patience')
    parser.add_argument('--train_ratio', type=float, default=DEFAULT_CONFIG['train_ratio'],
                       help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=DEFAULT_CONFIG['val_ratio'],
                       help='Validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=DEFAULT_CONFIG['test_ratio'],
                       help='Test data ratio')
    parser.add_argument('--device', type=str, default=DEFAULT_CONFIG['device'],
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--use_light_model', action='store_true', default=DEFAULT_CONFIG['use_light_model'],
                       help='Use light model')
    parser.add_argument('--no_light_model', dest='use_light_model', action='store_false',
                       help='Use full model (not light)')
    parser.add_argument('--model', type=str, choices=['gnn', 'transformer', 'both'],
                       default='both', help='Which model to train')
    
    return parser.parse_args()


def main():
    """メイン関数"""
    args = parse_args()
    
    # 設定を更新
    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG.update({
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'use_light_model': args.use_light_model,
        'use_huber_loss': getattr(args, 'use_huber_loss', DEFAULT_CONFIG.get('use_huber_loss', True)),
        'huber_delta': getattr(args, 'huber_delta', DEFAULT_CONFIG.get('huber_delta', 1.0)),
        'weight_decay': getattr(args, 'weight_decay', DEFAULT_CONFIG.get('weight_decay', 1e-4)),
        'gradient_accumulation_steps': getattr(args, 'gradient_accumulation_steps', DEFAULT_CONFIG.get('gradient_accumulation_steps', 1)),
        'warmup_epochs': getattr(args, 'warmup_epochs', DEFAULT_CONFIG.get('warmup_epochs', 10)),
    })
    
    # デバイス設定
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    CONFIG['device'] = device
    
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
        print("   先にデータ前処理を実行してください")
        return
    
    print("=" * 80)
    print("HEA Elastic Modulus Prediction: GNN & Transformer Training")
    print("=" * 80)
    print(f"📊 データファイル: {data_path}")
    print(f"📁 出力ディレクトリ: {output_dir}")
    print(f"💾 モデルディレクトリ: {model_dir}")
    print(f"⚙️  デバイス: {CONFIG['device']}")
    print(f"📦 モデルタイプ: {'軽量版' if CONFIG['use_light_model'] else 'フル版'}")
    print("=" * 80)
    
    # グローバル変数として設定を保存（関数内で使用）
    import train
    train.CONFIG = CONFIG
    train.DATA_PATH = DATA_PATH
    train.OUTPUT_DIR = OUTPUT_DIR
    train.MODEL_DIR = MODEL_DIR
    
    gnn_results = None
    transformer_results = None
    
    # GNN訓練
    if args.model in ['gnn', 'both']:
        gnn_results = train_gnn(CONFIG, data_path, output_dir, model_dir)
    
    # Transformer訓練
    if args.model in ['transformer', 'both']:
        transformer_results = train_transformer(CONFIG, data_path, output_dir, model_dir)
    
    # 結果比較
    if gnn_results and transformer_results:
        print("\n" + "=" * 80)
        print("Model Comparison")
        print("=" * 80)
        print(f"GNN - R²: {gnn_results['test_r2']:.4f}, RMSE: {gnn_results['test_rmse']:.4f} GPa")
        print(f"Transformer - R²: {transformer_results['test_r2']:.4f}, RMSE: {transformer_results['test_rmse']:.4f} GPa")
        
    # 比較結果を保存
    comparison = {
        'gnn': gnn_results,
        'transformer': transformer_results
    }
    with open(output_dir / "model_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)
        
        print(f"\n✅ 結果を保存しました: {output_dir / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
