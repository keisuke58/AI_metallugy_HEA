import sys
print("Python環境チェック")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path[:3]}")

# 必要なパッケージのチェック
required = ['torch', 'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'tqdm']
missing = []
for pkg in required:
    try:
        if pkg == 'sklearn':
            __import__('sklearn')
        else:
            __import__(pkg)
        print(f"✅ {pkg}: インストール済み")
    except ImportError:
        print(f"❌ {pkg}: 未インストール")
        missing.append(pkg)

if missing:
    print(f"\n未インストールパッケージ: {', '.join(missing)}")
    print("\nインストール方法:")
    print("1. pipが利用可能な場合: pip install " + " ".join(missing))
    print("2. condaが利用可能な場合: conda install " + " ".join(missing))
    print("3. システムパッケージマネージャーを使用")
else:
    print("\n✅ すべてのパッケージがインストールされています！")
