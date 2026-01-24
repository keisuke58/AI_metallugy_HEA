# 🔑 Materials Project API キー取得ガイド

**作成日**: 2026年1月20日

---

## 📋 Materials Project APIキー取得手順

### ステップ1: アカウント作成

1. **Materials Projectにアクセス**
   - URL: https://materialsproject.org
   - 「Sign Up」または「Register」をクリック

2. **アカウント情報を入力**
   - メールアドレス
   - パスワード
   - 名前、所属など

3. **メール認証**
   - 登録したメールアドレスに認証メールが届く
   - メール内のリンクをクリックして認証

---

### ステップ2: APIキーの取得

1. **ログイン**
   - https://materialsproject.org にログイン

2. **Dashboardにアクセス**
   - 右上のユーザー名をクリック
   - 「Dashboard」を選択
   - または直接: https://materialsproject.org/dashboard

3. **API Keysセクションを確認**
   - Dashboardページの「API Keys」セクションを探す
   - または直接: https://materialsproject.org/api

4. **APIキーをコピー**
   - 「API Key」または「Your API Key」の下に表示されるキーをコピー
   - 例: `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

---

### ステップ3: APIキーの設定

#### 方法1: 環境変数に設定（推奨）

```bash
# 一時的に設定（現在のセッションのみ）
export MP_API_KEY='your_api_key_here'

# 永続的に設定（~/.bashrc または ~/.zshrc に追加）
echo 'export MP_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

#### 方法2: .envファイルに保存

```bash
# プロジェクトディレクトリに.envファイルを作成
cd /home/nishioka/LUH/AI_metallurgy/data_collection
echo 'MP_API_KEY=your_api_key_here' > .env
```

#### 方法3: コマンドライン引数で指定

```bash
python scripts/download_materials_project_enhanced.py --api-key your_api_key_here
```

---

## 📦 必要なパッケージのインストール

```bash
pip install mp-api pandas numpy
```

---

## 🚀 使用方法

### 基本的な使用方法

```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection

# 環境変数が設定されている場合
python scripts/download_materials_project_enhanced.py

# または、コマンドライン引数で指定
python scripts/download_materials_project_enhanced.py --api-key YOUR_KEY
```

---

## ⚠️ 注意事項

1. **APIキーの管理**
   - APIキーは秘密情報です。GitHubなどに公開しないでください
   - `.env`ファイルは`.gitignore`に追加してください

2. **レート制限**
   - Materials Project APIにはレート制限があります（約25リクエスト/秒）
   - 大量のデータを取得する場合は、適切な待機時間を設定してください

3. **データの性質**
   - Materials Projectのデータは**計算データ（DFT）**です
   - 実験データとは異なる可能性があります
   - HEAのデータは限定的です

---

## 📊 期待される結果

- **期待サンプル数**: 500-1000サンプル
- **データ内容**: バルク弾性率、せん断弾性率、Young's modulus（計算値）
- **品質**: 計算データ（実験データではない）

---

## 🔗 参考リンク

- **Materials Project**: https://materialsproject.org
- **API Documentation**: https://docs.materialsproject.org
- **API Getting Started**: https://docs.materialsproject.org/downloading-data/using-the-api/getting-started
- **Dashboard**: https://materialsproject.org/dashboard
- **API Keys**: https://materialsproject.org/api

---

**最終更新**: 2026年1月20日
