"""
元素の物理特性データベース
"""

ELEMENT_PROPERTIES = {
    'Ti': {'atomic_num': 22, 'radius': 147, 'electronegativity': 1.54, 'vec': 4, 'mass': 47.87},
    'Zr': {'atomic_num': 40, 'radius': 160, 'electronegativity': 1.33, 'vec': 4, 'mass': 91.22},
    'Hf': {'atomic_num': 72, 'radius': 159, 'electronegativity': 1.3, 'vec': 4, 'mass': 178.49},
    'Nb': {'atomic_num': 41, 'radius': 146, 'electronegativity': 1.6, 'vec': 5, 'mass': 92.91},
    'Ta': {'atomic_num': 73, 'radius': 146, 'electronegativity': 1.5, 'vec': 5, 'mass': 180.95},
    'V': {'atomic_num': 23, 'radius': 134, 'electronegativity': 1.63, 'vec': 5, 'mass': 50.94},
    'Cr': {'atomic_num': 24, 'radius': 128, 'electronegativity': 1.66, 'vec': 6, 'mass': 52.00},
    'Mo': {'atomic_num': 42, 'radius': 139, 'electronegativity': 2.16, 'vec': 6, 'mass': 95.96},
    'W': {'atomic_num': 74, 'radius': 139, 'electronegativity': 2.36, 'vec': 6, 'mass': 183.84},
    'Fe': {'atomic_num': 26, 'radius': 126, 'electronegativity': 1.83, 'vec': 8, 'mass': 55.85},
    'Co': {'atomic_num': 27, 'radius': 125, 'electronegativity': 1.88, 'vec': 9, 'mass': 58.93},
    'Ni': {'atomic_num': 28, 'radius': 124, 'electronegativity': 1.91, 'vec': 10, 'mass': 58.69},
    'Cu': {'atomic_num': 29, 'radius': 128, 'electronegativity': 1.9, 'vec': 11, 'mass': 63.55},
    'Al': {'atomic_num': 13, 'radius': 143, 'electronegativity': 1.61, 'vec': 3, 'mass': 26.98},
    'Mn': {'atomic_num': 25, 'radius': 127, 'electronegativity': 1.55, 'vec': 7, 'mass': 54.94},
    'Si': {'atomic_num': 14, 'radius': 111, 'electronegativity': 1.9, 'vec': 4, 'mass': 28.09},
    'Sn': {'atomic_num': 50, 'radius': 145, 'electronegativity': 1.96, 'vec': 4, 'mass': 118.71},
    'Re': {'atomic_num': 75, 'radius': 137, 'electronegativity': 1.9, 'vec': 7, 'mass': 186.21},
    'Ru': {'atomic_num': 44, 'radius': 134, 'electronegativity': 2.2, 'vec': 8, 'mass': 101.07},
    'Pd': {'atomic_num': 46, 'radius': 137, 'electronegativity': 2.2, 'vec': 10, 'mass': 106.42},
}

ELEMENT_LIST = sorted(ELEMENT_PROPERTIES.keys())
ELEMENT_TO_IDX = {elem: idx for idx, elem in enumerate(ELEMENT_LIST)}
