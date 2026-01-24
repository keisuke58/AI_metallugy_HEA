"""
データ処理ユーティリティ関数
"""
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple
from .element_properties import ELEMENT_PROPERTIES


def normalize_composition(compositions: Dict[str, float]) -> Dict[str, float]:
    """
    組成を正規化（合計が1になるように）
    
    Args:
        compositions: 元素名をキー、組成比を値とする辞書
        
    Returns:
        正規化された組成辞書
    """
    total = sum(compositions.values())
    if total > 0:
        return {k: v / total for k, v in compositions.items()}
    return compositions


def get_material_descriptors(compositions: Dict[str, float]) -> Dict[str, float]:
    """
    材料記述子を計算
    
    Args:
        compositions: 正規化された組成辞書
        
    Returns:
        材料記述子の辞書
    """
    if not compositions:
        return {
            'mixing_entropy': 0.0,
            'mixing_enthalpy': 0.0,
            'vec': 0.0,
            'delta_r': 0.0,
            'delta_chi': 0.0,
            'mean_atomic_radius': 0.0,
            'mean_electronegativity': 0.0,
            'mean_atomic_mass': 0.0,
        }
    
    # 組成比のリスト
    comp_values = list(compositions.values())
    elements = list(compositions.keys())
    
    # 原子半径、電気陰性度、VEC、質量を取得
    radii = [ELEMENT_PROPERTIES.get(elem, {}).get('radius', 0) for elem in elements]
    chi = [ELEMENT_PROPERTIES.get(elem, {}).get('electronegativity', 0) for elem in elements]
    vec = [ELEMENT_PROPERTIES.get(elem, {}).get('vec', 0) for elem in elements]
    masses = [ELEMENT_PROPERTIES.get(elem, {}).get('mass', 0) for elem in elements]
    
    # 平均値
    mean_radius = np.average(radii, weights=comp_values) if comp_values else 0.0
    mean_chi = np.average(chi, weights=comp_values) if comp_values else 0.0
    mean_vec = np.average(vec, weights=comp_values) if comp_values else 0.0
    mean_mass = np.average(masses, weights=comp_values) if comp_values else 0.0
    
    # 標準偏差（delta）
    if len(comp_values) > 1:
        delta_r = np.sqrt(np.average([(r - mean_radius)**2 for r in radii], weights=comp_values))
        delta_chi = np.sqrt(np.average([(c - mean_chi)**2 for c in chi], weights=comp_values))
    else:
        delta_r = 0.0
        delta_chi = 0.0
    
    # 混合エントロピー（簡易版: -R * sum(x_i * ln(x_i))）
    R = 8.314  # 気体定数
    mixing_entropy = -R * sum(c * np.log(c) for c in comp_values if c > 0)
    
    # 混合エンタルピー（簡易版: 元素間の相互作用を考慮）
    # 実際の値は実験データから取得する必要があるが、ここでは簡易計算
    mixing_enthalpy = 0.0  # 実際の値はデータから取得
    
    return {
        'mixing_entropy': mixing_entropy,
        'mixing_enthalpy': mixing_enthalpy,
        'vec': mean_vec,
        'delta_r': delta_r,
        'delta_chi': delta_chi,
        'mean_atomic_radius': mean_radius,
        'mean_electronegativity': mean_chi,
        'mean_atomic_mass': mean_mass,
    }


def parse_composition_string(composition_str: str) -> Dict[str, float]:
    """
    組成文字列を解析して元素と組成比を抽出
    
    例: "CoFeNi" -> {'Co': 0.33, 'Fe': 0.33, 'Ni': 0.33}
    例: "Al0.5CoCrCuFeNi" -> {'Al': 0.5, 'Co': 1, 'Cr': 1, 'Cu': 1, 'Fe': 1, 'Ni': 1}
    
    Args:
        composition_str: 組成文字列
        
    Returns:
        組成辞書
    """
    if pd.isna(composition_str) or not isinstance(composition_str, str):
        return {}
    
    # 元素記号と数値を抽出
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, composition_str)
    
    elements = {}
    total = 0
    
    for element, value_str in matches:
        if element in ELEMENT_PROPERTIES:
            if value_str == '':
                value = 1.0
            else:
                value = float(value_str)
            elements[element] = value
            total += value
    
    # 正規化（合計が1になるように）
    if total > 0:
        elements = {k: v/total for k, v in elements.items()}
    
    return elements


def extract_composition_from_row(row: pd.Series) -> Dict[str, float]:
    """
    DataFrameの行から組成を抽出
    1. comp_*カラムから抽出を試みる
    2. なければcompositionカラムの文字列を解析
    3. それもなければalloy_nameカラムの文字列を解析
    
    Args:
        row: DataFrameの行
        
    Returns:
        組成辞書
    """
    compositions = {}
    
    # 方法1: comp_*カラムから抽出
    for col in row.index:
        if col.startswith('comp_'):
            element = col.replace('comp_', '')
            comp_value = row[col]
            if pd.notna(comp_value) and comp_value > 0:
                compositions[element] = float(comp_value)
    
    # 方法2: compositionカラムから抽出
    if not compositions and 'composition' in row.index:
        comp_str = row['composition']
        compositions = parse_composition_string(str(comp_str))
    
    # 方法3: alloy_nameカラムから抽出
    if not compositions and 'alloy_name' in row.index:
        alloy_name = row['alloy_name']
        compositions = parse_composition_string(str(alloy_name))
    
    return normalize_composition(compositions)
