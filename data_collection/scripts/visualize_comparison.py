#!/usr/bin/env python3
"""
実測データのみ vs 実測+計算データの性能比較を可視化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import glob

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_latest_results():
    """最新の結果ファイルを読み込む"""
    measured_files = sorted(glob.glob(str(RESULTS_DIR / "results_measured_*.json")))
    combined_files = sorted(glob.glob(str(RESULTS_DIR / "results_combined_*.json")))
    
    if not measured_files or not combined_files:
        print("❌ 結果ファイルが見つかりません")
        return None, None
    
    with open(measured_files[-1], 'r') as f:
        results_measured = json.load(f)
    
    with open(combined_files[-1], 'r') as f:
        results_combined = json.load(f)
    
    return results_measured, results_combined

def create_comparison_plots(results_measured, results_combined):
    """比較プロットを作成"""
    
    # データを準備
    models = []
    metrics = []
    values = []
    data_types = []
    
    common_models = set(results_measured['models'].keys()) & set(results_combined['models'].keys())
    
    for model in sorted(common_models):
        if model in results_measured['models'] and model in results_combined['models']:
            # R²
            models.append(model)
            metrics.append('R²')
            values.append(results_measured['models'][model]['test_r2'])
            data_types.append('実測のみ')
            
            models.append(model)
            metrics.append('R²')
            values.append(results_combined['models'][model]['test_r2'])
            data_types.append('実測+計算')
            
            # RMSE
            models.append(model)
            metrics.append('RMSE (GPa)')
            values.append(results_measured['models'][model]['test_rmse'])
            data_types.append('実測のみ')
            
            models.append(model)
            metrics.append('RMSE (GPa)')
            values.append(results_combined['models'][model]['test_rmse'])
            data_types.append('実測+計算')
            
            # MAE
            models.append(model)
            metrics.append('MAE (GPa)')
            values.append(results_measured['models'][model]['test_mae'])
            data_types.append('実測のみ')
            
            models.append(model)
            metrics.append('MAE (GPa)')
            values.append(results_combined['models'][model]['test_mae'])
            data_types.append('実測+計算')
    
    df = pd.DataFrame({
        'Model': models,
        'Metric': metrics,
        'Value': values,
        'Data Type': data_types
    })
    
    # 1. R²比較
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # R²
    df_r2 = df[df['Metric'] == 'R²']
    pivot_r2 = df_r2.pivot(index='Model', columns='Data Type', values='Value')
    pivot_r2.plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'])
    axes[0].set_title('Test R² Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    # RMSE
    df_rmse = df[df['Metric'] == 'RMSE (GPa)']
    pivot_rmse = df_rmse.pivot(index='Model', columns='Data Type', values='Value')
    pivot_rmse.plot(kind='bar', ax=axes[1], color=['#3498db', '#e74c3c'])
    axes[1].set_title('Test RMSE Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('RMSE (GPa)', fontsize=12)
    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # MAE
    df_mae = df[df['Metric'] == 'MAE (GPa)']
    pivot_mae = df_mae.pivot(index='Model', columns='Data Type', values='Value')
    pivot_mae.plot(kind='bar', ax=axes[2], color=['#3498db', '#e74c3c'])
    axes[2].set_title('Test MAE Comparison', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('MAE (GPa)', fontsize=12)
    axes[2].set_xlabel('Model', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'model_comparison_measured_vs_combined.png', dpi=300, bbox_inches='tight')
    print(f"✅ 比較プロットを保存しました: {FIGURES_DIR / 'model_comparison_measured_vs_combined.png'}")
    plt.close()
    
    # 2. 改善率の可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    improvements = []
    model_names = []
    for model in sorted(common_models):
        if model in results_measured['models'] and model in results_combined['models']:
            r2_measured = results_measured['models'][model]['test_r2']
            r2_combined = results_combined['models'][model]['test_r2']
            improvement = (r2_combined - r2_measured) * 100  # パーセンテージ
            improvements.append(improvement)
            model_names.append(model)
    
    colors = ['green' if x > 0 else 'red' for x in improvements]
    bars = ax.barh(model_names, improvements, color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('R² Improvement (%)', fontsize=12)
    ax.set_title('Model Performance: Measured+Calculated vs Measured Only', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 値をバーの上に表示
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax.text(val + (1 if val > 0 else -1), i, f'{val:.1f}%', 
                va='center', ha='left' if val > 0 else 'right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'performance_improvement.png', dpi=300, bbox_inches='tight')
    print(f"✅ 改善率プロットを保存しました: {FIGURES_DIR / 'performance_improvement.png'}")
    plt.close()
    
    # 3. データ数と性能の関係
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_counts = {
        '実測のみ': results_measured.get('feature_count', 0),
        '実測+計算': results_combined.get('feature_count', 0)
    }
    
    best_r2 = {
        '実測のみ': max([r['test_r2'] for r in results_measured['models'].values()]),
        '実測+計算': max([r['test_r2'] for r in results_combined['models'].values()])
    }
    
    x_pos = np.arange(len(data_counts))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, list(data_counts.values()), width, 
                   label='Feature Count', color='#3498db', alpha=0.7)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x_pos + width/2, list(best_r2.values()), width,
                    label='Best R²', color='#e74c3c', alpha=0.7)
    
    ax.set_xlabel('Data Type', fontsize=12)
    ax.set_ylabel('Feature Count', fontsize=12, color='#3498db')
    ax2.set_ylabel('Best R² Score', fontsize=12, color='#e74c3c')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(data_counts.keys()))
    ax.set_title('Data Characteristics and Best Model Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 凡例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'data_characteristics.png', dpi=300, bbox_inches='tight')
    print(f"✅ データ特性プロットを保存しました: {FIGURES_DIR / 'data_characteristics.png'}")
    plt.close()

def create_summary_table(results_measured, results_combined):
    """サマリーテーブルを作成"""
    common_models = set(results_measured['models'].keys()) & set(results_combined['models'].keys())
    
    summary_data = []
    for model in sorted(common_models):
        if model in results_measured['models'] and model in results_combined['models']:
            summary_data.append({
                'Model': model,
                'Measured R²': f"{results_measured['models'][model]['test_r2']:.4f}",
                'Measured RMSE': f"{results_measured['models'][model]['test_rmse']:.2f}",
                'Measured MAE': f"{results_measured['models'][model]['test_mae']:.2f}",
                'Combined R²': f"{results_combined['models'][model]['test_r2']:.4f}",
                'Combined RMSE': f"{results_combined['models'][model]['test_rmse']:.2f}",
                'Combined MAE': f"{results_combined['models'][model]['test_mae']:.2f}",
                'R² Improvement': f"{(results_combined['models'][model]['test_r2'] - results_measured['models'][model]['test_r2']):.4f}",
            })
    
    df_summary = pd.DataFrame(summary_data)
    
    # CSVで保存
    summary_file = RESULTS_DIR / 'comparison_summary.csv'
    df_summary.to_csv(summary_file, index=False)
    print(f"✅ サマリーテーブルを保存しました: {summary_file}")
    
    # 表示
    print("\n" + "=" * 100)
    print("性能比較サマリー")
    print("=" * 100)
    print(df_summary.to_string(index=False))
    
    return df_summary

def main():
    """メイン関数"""
    print("=" * 80)
    print("実測データのみ vs 実測+計算データ 性能比較可視化")
    print("=" * 80)
    
    results_measured, results_combined = load_latest_results()
    
    if results_measured is None or results_combined is None:
        return
    
    print(f"\n📊 実測データのみ:")
    print(f"   特徴量数: {results_measured.get('feature_count', 'N/A')}")
    print(f"   モデル数: {len(results_measured['models'])}")
    
    print(f"\n📊 実測+計算データ:")
    print(f"   特徴量数: {results_combined.get('feature_count', 'N/A')}")
    print(f"   モデル数: {len(results_combined['models'])}")
    
    # 可視化
    create_comparison_plots(results_measured, results_combined)
    
    # サマリーテーブル
    create_summary_table(results_measured, results_combined)
    
    print("\n" + "=" * 80)
    print("✅ 可視化が完了しました！")
    print("=" * 80)

if __name__ == "__main__":
    main()
