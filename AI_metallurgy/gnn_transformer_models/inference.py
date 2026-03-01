"""
推論スクリプト: 訓練済みモデルで予測を実行
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

from data_loader import HEADataset, TransformerDataset, collate_gnn, collate_transformer
from gnn_model import HEAGNN, HEAGNNLight
from transformer_model import HEATransformer, HEATransformerLight
from torch.utils.data import DataLoader


def load_gnn_model(model_path: str, use_light: bool = True, device: str = 'cpu'):
    """GNNモデルを読み込む"""
    if use_light:
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
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)


def load_transformer_model(model_path: str, vocab_size: int = 20, use_light: bool = True, device: str = 'cpu'):
    """Transformerモデルを読み込む"""
    if use_light:
        model = HEATransformerLight(
            vocab_size=vocab_size,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_feedforward=256,
            max_seq_len=20,
            additional_feat_dim=8,
            dropout=0.1
        )
    else:
        model = HEATransformer(
            vocab_size=vocab_size,
            d_model=256,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            max_seq_len=20,
            additional_feat_dim=8,
            dropout=0.1
        )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)


def predict_gnn(model, data_path: str, device: str = 'cpu'):
    """GNNモデルで予測"""
    dataset = HEADataset(data_path)
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=False, 
        collate_fn=collate_gnn
    )
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            predictions.extend(output.cpu().numpy().flatten())
            targets.extend(batch.y.cpu().numpy().flatten())
    
    return np.array(predictions), np.array(targets)


def predict_transformer(model, data_path: str, device: str = 'cpu'):
    """Transformerモデルで予測"""
    dataset = TransformerDataset(data_path, max_length=20)
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=False,
        collate_fn=collate_transformer
    )
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            token_ids = batch['token_ids'].to(device)
            comp_values = batch['comp_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            additional_features = batch['additional_features'].to(device)
            
            output = model(token_ids, comp_values, attention_mask, additional_features)
            predictions.extend(output.cpu().numpy().flatten())
            targets.extend(batch['target'].cpu().numpy().flatten())
    
    return np.array(predictions), np.array(targets)


def main():
    parser = argparse.ArgumentParser(description='HEA Elastic Modulus Prediction')
    parser.add_argument('--data_path', type=str, 
                       default='../data_collection/processed_data/data_with_features.csv',
                       help='Path to data CSV file')
    parser.add_argument('--gnn_model', type=str,
                       default='models/gnn_best_model.pth',
                       help='Path to GNN model')
    parser.add_argument('--transformer_model', type=str,
                       default='models/transformer_best_model.pth',
                       help='Path to Transformer model')
    parser.add_argument('--model', type=str, choices=['gnn', 'transformer', 'both'],
                       default='both', help='Which model to use')
    parser.add_argument('--use_light', action='store_true',
                       help='Use light model')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # デバイス設定
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # GNN予測
    if args.model in ['gnn', 'both']:
        print("\n" + "=" * 80)
        print("GNN Model Prediction")
        print("=" * 80)
        
        gnn_model_path = Path(args.gnn_model)
        if gnn_model_path.exists():
            model = load_gnn_model(str(gnn_model_path), args.use_light, device)
            predictions, targets = predict_gnn(model, str(data_path), device)
            
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            r2 = r2_score(targets, predictions)
            rmse = np.sqrt(mean_squared_error(targets, predictions))
            mae = mean_absolute_error(targets, predictions)
            
            print(f"R²: {r2:.4f}")
            print(f"RMSE: {rmse:.4f} GPa")
            print(f"MAE: {mae:.4f} GPa")
        else:
            print(f"❌ GNN model not found: {gnn_model_path}")
    
    # Transformer予測
    if args.model in ['transformer', 'both']:
        print("\n" + "=" * 80)
        print("Transformer Model Prediction")
        print("=" * 80)
        
        transformer_model_path = Path(args.transformer_model)
        if transformer_model_path.exists():
            # vocab_sizeを取得（データセットから）
            dataset = TransformerDataset(str(data_path), max_length=20)
            model = load_transformer_model(
                str(transformer_model_path), 
                dataset.vocab_size, 
                args.use_light, 
                device
            )
            predictions, targets = predict_transformer(model, str(data_path), device)
            
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            r2 = r2_score(targets, predictions)
            rmse = np.sqrt(mean_squared_error(targets, predictions))
            mae = mean_absolute_error(targets, predictions)
            
            print(f"R²: {r2:.4f}")
            print(f"RMSE: {rmse:.4f} GPa")
            print(f"MAE: {mae:.4f} GPa")
        else:
            print(f"❌ Transformer model not found: {transformer_model_path}")


if __name__ == "__main__":
    main()
