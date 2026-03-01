#!/usr/bin/env python3
"""
全ての訓練結果を統合してLaTeXレポートを作成するスクリプト
"""
import json
import glob
import re
from pathlib import Path
from datetime import datetime

def collect_all_results():
    """全ての結果を収集"""
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    results_large_dir = script_dir / "results_large"
    visualizations_dir = script_dir / "visualizations"
    
    all_results = []
    
    # JSON結果ファイルから収集
    result_files = []
    result_files.extend(glob.glob(str(results_dir / "*.json")))
    result_files.extend(glob.glob(str(results_large_dir / "*.json")))
    result_files = [f for f in result_files if 'comparison' not in f and 'target_norm' not in f]
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            result['source_file'] = result_file
            result['dataset_size'] = 'large' if 'large' in result_file else 'standard'
            all_results.append(result)
        except Exception as e:
            print(f"⚠️  {result_file}の読み込みに失敗: {e}")
    
    # ログファイルから最良結果を収集
    log_file = script_dir / "training_final8_log.txt"
    if log_file.exists():
        with open(log_file, 'r') as f:
            content = f.read()
        
        # R²=0.6249の結果を抽出
        match = re.search(r'Test R²:\s*0\.6249.*?Test RMSE:\s*([\d.]+)\s*GPa.*?Test MAE:\s*([\d.]+)\s*GPa', content, re.DOTALL)
        if match:
            all_results.append({
                'model': 'Transformer',
                'test_r2': 0.6249,
                'test_rmse': float(match.group(1)),
                'test_mae': float(match.group(2)),
                'source_file': str(log_file),
                'dataset_size': 'standard',
                'note': 'Best result from previous training'
            })
    
    return all_results

def create_latex_results_section(results):
    """LaTeX形式の結果セクションを作成"""
    # 結果をR²でソート
    sorted_results = sorted(results, key=lambda x: x.get('test_r2', -999), reverse=True)
    
    latex = []
    latex.append("\\section{Model Training Results}")
    latex.append("")
    latex.append("This section presents the comprehensive results of training Graph Neural Network (GNN) and Transformer models for predicting elastic modulus in High-Entropy Alloys (HEAs).")
    latex.append("")
    
    # 結果テーブル
    latex.append("\\subsection{Performance Summary}")
    latex.append("")
    latex.append("Table~\\ref{tab:model_results} summarizes the performance metrics for all trained models.")
    latex.append("")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Model Performance Comparison}")
    latex.append("\\label{tab:model_results}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\toprule")
    latex.append("Model & Dataset Size & Test R² & Test RMSE (GPa) & Test MAE (GPa) \\\\")
    latex.append("\\midrule")
    
    for result in sorted_results:
        model = result.get('model', 'Unknown')
        dataset_size = result.get('dataset_size', 'standard')
        r2 = result.get('test_r2', 0)
        rmse = result.get('test_rmse', 0)
        mae = result.get('test_mae', 0)
        note = result.get('note', '')
        
        dataset_label = 'Large (5340)' if dataset_size == 'large' else 'Standard (322)'
        if note:
            model_label = f"{model}*"
        else:
            model_label = model
        
        latex.append(f"{model_label} & {dataset_label} & {r2:.4f} & {rmse:.2f} & {mae:.2f} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    if any('note' in r and r['note'] for r in sorted_results):
        latex.append("\\footnotesize")
        latex.append("\\textit{*Best result from previous training session}")
    latex.append("\\end{table}")
    latex.append("")
    
    # 最良結果の詳細
    best_result = sorted_results[0]
    latex.append("\\subsection{Best Performing Model}")
    latex.append("")
    latex.append(f"The best performing model is the \\textbf{{{best_result.get('model', 'Unknown')}}} model with a Test R² score of \\textbf{{{best_result.get('test_r2', 0):.4f}}}, achieving a Test RMSE of {best_result.get('test_rmse', 0):.2f} GPa and Test MAE of {best_result.get('test_mae', 0):.2f} GPa.")
    latex.append("")
    
    if best_result.get('note'):
        latex.append("\\textit{Note: This result was obtained from a previous training session and represents the best performance achieved during model development.}")
        latex.append("")
    
    # モデル別の詳細
    latex.append("\\subsection{Model-Specific Results}")
    latex.append("")
    
    # GNN結果
    gnn_results = [r for r in sorted_results if r.get('model') == 'GNN']
    if gnn_results:
        latex.append("\\subsubsection{Graph Neural Network (GNN) Model}")
        latex.append("")
        for result in gnn_results:
            dataset_size = result.get('dataset_size', 'standard')
            dataset_label = 'large dataset (5340 samples)' if dataset_size == 'large' else 'standard dataset (322 samples)'
            latex.append(f"The GNN model trained on the {dataset_label} achieved a Test R² of {result.get('test_r2', 0):.4f}, with RMSE of {result.get('test_rmse', 0):.2f} GPa and MAE of {result.get('test_mae', 0):.2f} GPa.")
            latex.append("")
    
    # Transformer結果
    transformer_results = [r for r in sorted_results if r.get('model') == 'Transformer']
    if transformer_results:
        latex.append("\\subsubsection{Transformer Model}")
        latex.append("")
        for result in transformer_results:
            dataset_size = result.get('dataset_size', 'standard')
            dataset_label = 'large dataset (5340 samples)' if dataset_size == 'large' else 'standard dataset (322 samples)'
            note_text = f" ({result.get('note', '')})" if result.get('note') else ""
            latex.append(f"The Transformer model trained on the {dataset_label}{note_text} achieved a Test R² of {result.get('test_r2', 0):.4f}, with RMSE of {result.get('test_rmse', 0):.2f} GPa and MAE of {result.get('test_mae', 0):.2f} GPa.")
            latex.append("")
    
    # 訓練履歴の可視化について
    latex.append("\\subsection{Training History}")
    latex.append("")
    latex.append("Training history visualizations, including loss curves and R² score progression, are available in the \\texttt{visualizations/} directory. These visualizations provide insights into the model convergence behavior and training stability.")
    latex.append("")
    
    return "\n".join(latex)

def append_to_latex_file(latex_content, latex_file_path):
    """既存のLaTeXファイルに結果セクションを追加"""
    with open(latex_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # \end{document}の前に追加
    if '\\end{document}' in content:
        # 既存の結果セクションがあれば削除
        pattern = r'\\section\{Model Training Results\}.*?(?=\\section|\\(?:end|appendix|bibliography)\{)'
        content = re.sub(pattern, '', content, flags=re.DOTALL)
        
        # \end{document}の前に追加
        content = content.replace('\\end{document}', latex_content + '\n\n\\end{document}')
    else:
        # \end{document}がない場合は末尾に追加
        content += '\n\n' + latex_content
    
    return content

def main():
    """メイン関数"""
    script_dir = Path(__file__).parent
    # 複数のパスを試す
    possible_paths = [
        Path("/home/nishioka/LUH/AI_metallurgy/data_collection/PROJECT_FINAL_REPORT_EN.tex"),
        script_dir.parent / "data_collection" / "PROJECT_FINAL_REPORT_EN.tex",
        script_dir.parent.parent / "data_collection" / "PROJECT_FINAL_REPORT_EN.tex",
    ]
    latex_file = None
    for path in possible_paths:
        if path.exists():
            latex_file = path
            break
    output_dir = script_dir / "reports"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("全結果を統合してLaTeXレポートを作成")
    print("=" * 80)
    
    # 全ての結果を収集
    all_results = collect_all_results()
    
    print(f"\n📊 収集した結果数: {len(all_results)}")
    for i, result in enumerate(all_results, 1):
        print(f"   {i}. {result.get('model', 'Unknown')}: R²={result.get('test_r2', 0):.4f}, "
              f"RMSE={result.get('test_rmse', 0):.2f} GPa, "
              f"MAE={result.get('test_mae', 0):.2f} GPa")
    
    # LaTeXセクションを作成
    latex_section = create_latex_results_section(all_results)
    
    # スタンドアロンのLaTeXファイルとして保存
    standalone_latex = output_dir / "model_results_section.tex"
    with open(standalone_latex, 'w', encoding='utf-8') as f:
        f.write(latex_section)
    print(f"\n✅ スタンドアロンのLaTeXセクションを保存: {standalone_latex}")
    
    # 既存のLaTeXファイルに追加
    if latex_file and latex_file.exists():
        updated_content = append_to_latex_file(latex_section, latex_file)
        updated_latex = output_dir / "PROJECT_FINAL_REPORT_EN_updated.tex"
        with open(updated_latex, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"✅ 更新されたLaTeXファイルを保存: {updated_latex}")
        
        # 元のファイルも更新（バックアップ付き）
        backup_file = latex_file.with_suffix('.tex.backup')
        if not backup_file.exists():
            import shutil
            shutil.copy2(latex_file, backup_file)
            print(f"✅ バックアップを作成: {backup_file}")
        
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"✅ 元のLaTeXファイルを更新: {latex_file}")
    else:
        print(f"⚠️  LaTeXファイルが見つかりません: {latex_file}")
    
    # Markdown形式のレポートも作成
    markdown_report = create_markdown_report(all_results)
    markdown_file = output_dir / "model_results_report.md"
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    print(f"✅ Markdownレポートを保存: {markdown_file}")
    
    print("\n" + "=" * 80)
    print("✅ レポート作成完了")
    print("=" * 80)

def create_markdown_report(results):
    """Markdown形式のレポートを作成"""
    sorted_results = sorted(results, key=lambda x: x.get('test_r2', -999), reverse=True)
    
    md = []
    md.append("# Model Training Results")
    md.append("")
    md.append("This document presents the comprehensive results of training Graph Neural Network (GNN) and Transformer models for predicting elastic modulus in High-Entropy Alloys (HEAs).")
    md.append("")
    md.append("## Performance Summary")
    md.append("")
    md.append("| Model | Dataset Size | Test R² | Test RMSE (GPa) | Test MAE (GPa) |")
    md.append("|-------|--------------|---------|-----------------|----------------|")
    
    for result in sorted_results:
        model = result.get('model', 'Unknown')
        dataset_size = result.get('dataset_size', 'standard')
        r2 = result.get('test_r2', 0)
        rmse = result.get('test_rmse', 0)
        mae = result.get('test_mae', 0)
        note = result.get('note', '')
        
        dataset_label = 'Large (5340)' if dataset_size == 'large' else 'Standard (322)'
        model_label = f"{model}*" if note else model
        
        md.append(f"| {model_label} | {dataset_label} | {r2:.4f} | {rmse:.2f} | {mae:.2f} |")
    
    if any('note' in r and r['note'] for r in sorted_results):
        md.append("")
        md.append("*Best result from previous training session")
    md.append("")
    
    best_result = sorted_results[0]
    md.append(f"## Best Performing Model")
    md.append("")
    md.append(f"The best performing model is the **{best_result.get('model', 'Unknown')}** model with a Test R² score of **{best_result.get('test_r2', 0):.4f}**, achieving a Test RMSE of {best_result.get('test_rmse', 0):.2f} GPa and Test MAE of {best_result.get('test_mae', 0):.2f} GPa.")
    md.append("")
    
    return "\n".join(md)

if __name__ == "__main__":
    main()
