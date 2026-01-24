"""
推論スクリプト
訓練済みモデルで予測を実行
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .models import FNO1d, DeepONet, MEGNet, CGCNN, NeuralODE, PINN
from .data_loaders import (
    FNODataset, collate_fno,
    DeepONetDataset, collate_deeponet,
    GraphDataset, collate_graph,
    NeuralODEDataset, collate_neural_ode,
    PINNsDataset, collate_pinns
)

DEFAULT_DATA_PATH = _script_dir.parent / "data_collection" / "processed_data" / "data_with_features.csv"
DEFAULT_MODEL_DIR = _script_dir / "checkpoints"


def load_model(model_type: str, model_path: Path, device: torch.device):
    """モデルを読み込む"""
    if model_type == 'fno':
        model = FNO1d(modes=16, width=64, layers=4, input_channels=2, additional_feat_dim=8)
    elif model_type == 'deeponet':
        model = DeepONet(branch_input_dim=64, trunk_input_dim=8, branch_output_dim=128, trunk_output_dim=128)
    elif model_type == 'megnet':
        model = MEGNet(node_dim=5, edge_dim=3, state_dim=64, hidden_dim=128, num_layers=3, additional_feat_dim=8)
    elif model_type == 'cgcnn':
        model = CGCNN(node_dim=5, edge_dim=3, hidden_dim=128, num_layers=3, additional_feat_dim=8)
    elif model_type == 'neural_ode':
        model = NeuralODE(input_dim=64, hidden_dim=128, ode_dim=64, additional_feat_dim=8, num_ode_layers=3)
    elif model_type == 'pinns':
        model = PINN(input_dim=25, hidden_dims=[128, 128, 128, 128])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict(model_type: str, model, dataloader, device):
    """予測を実行"""
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in dataloader:
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
            
            predictions.extend(output.cpu().numpy().flatten())
            targets.extend(target.cpu().numpy().flatten())
    
    return np.array(predictions), np.array(targets)


def main():
    parser = argparse.ArgumentParser(description='Inference with trained models')
    parser.add_argument('--model', type=str, choices=['fno', 'deeponet', 'megnet', 'cgcnn', 'neural_ode', 'pinns', 'all'],
                       default='all', help='Model to use')
    parser.add_argument('--data_path', type=str, default=str(DEFAULT_DATA_PATH), help='Path to data CSV')
    parser.add_argument('--model_dir', type=str, default=str(DEFAULT_MODEL_DIR), help='Model directory')
    parser.add_argument('--output_path', type=str, default=None, help='Output CSV path')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device)
    model_dir = Path(args.model_dir)
    data_path = Path(args.data_path)
    
    if not data_path.exists():
        print(f"❌ データファイルが見つかりません: {data_path}")
        return
    
    # モデルリスト
    models_to_test = ['fno', 'deeponet', 'megnet', 'cgcnn', 'neural_ode', 'pinns'] if args.model == 'all' else [args.model]
    
    all_results = {}
    
    for model_type in models_to_test:
        model_path = model_dir / f"{model_type}_best_model.pth"
        if not model_path.exists():
            print(f"⚠️ {model_type}のモデルファイルが見つかりません: {model_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"{model_type.upper()} Inference")
        print(f"{'='*80}")
        
        # モデル読み込み
        model = load_model(model_type, model_path, device)
        
        # データローダー準備
        if model_type == 'fno':
            dataset = FNODataset(str(data_path), grid_size=64, fit_scaler=False)
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
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        # 予測
        predictions, targets = predict(model_type, model, dataloader, device)
        
        # 評価
        r2 = r2_score(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        
        print(f"R²: {r2:.4f}")
        print(f"RMSE: {rmse:.4f} GPa")
        print(f"MAE: {mae:.4f} GPa")
        
        all_results[model_type] = {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'predictions': predictions.tolist(),
            'targets': targets.tolist()
        }
        
        # 結果をCSVに保存
        if args.output_path:
            df_results = pd.DataFrame({
                'target': targets,
                'prediction': predictions,
                'residual': predictions - targets
            })
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_results.to_csv(output_path, index=False)
            print(f"✅ 結果を保存しました: {output_path}")
    
    # 全モデルの比較
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("Model Comparison")
        print(f"{'='*80}")
        print(f"{'Model':<15} {'R²':<10} {'RMSE (GPa)':<15} {'MAE (GPa)':<15}")
        print("-" * 80)
        for model_type, results in all_results.items():
            print(f"{model_type:<15} {results['r2']:<10.4f} {results['rmse']:<15.4f} {results['mae']:<15.4f}")


if __name__ == "__main__":
    main()
