# agent_force_test

## 目的

Databricks 環境上で LLM（OpenAI）を活用したプロンプトの実行・評価・管理を行うためのフレームワークです。

営業支援などのユースケースに応じたプロンプトを定義し、生成結果の品質を自動評価しながら、MLflow で実験結果をトラッキングします。

---

## ディレクトリ構成

```
.
├── config/
│   ├── prompt.yaml        # プロンプトと評価項目の定義（YAMLで管理）
│   ├── answer.yaml        # LLM生成結果の保存先
│   └── customer_data.csv  # 入力データ（顧客情報サンプル）
└── src/
    ├── base_prompt.py              # 抽象基底クラス（BacePrompt）
    ├── customer_summarize_prompt.py # 顧客要約ユースケースの実装
    ├── main_notebook.ipynb          # Databricks ノートブック（実行エントリポイント）
    └── set_api_key.py               # APIキー設定ユーティリティ
```

---

## 設計意図

### 抽象基底クラスによる共通処理の集約

`BacePrompt` クラスがすべてのプロンプトクラスの親クラスとなり、以下の共通機能を提供します：

- LLM クライアント（OpenAI）の初期化
- YAML ファイルへのプロンプト・評価項目の読み書き
- MLflow 実験パスの管理
- `generate()` / `evaluate()` / `run()` のテンプレートメソッドパターン

新しいユースケースを追加する場合は `BacePrompt` を継承し、`generate()` / `evaluate()` / `_setup_mlflow_experiment()` の3メソッドを実装するだけで、共通の実行・評価パイプラインが利用できます。

### YAML によるプロンプト設定の外部管理

プロンプトと評価項目は `config/prompt.yaml` で一元管理されています。

- `Common` セクション：すべてのクラスに共通の評価項目（簡潔さ・一貫性など）
- クラス名セクション（例: `CustomerSummarizePrompt`）：クラス固有のプロンプトと評価項目

評価時は共通評価項目とクラス固有評価項目が統合され、LLM による採点が行われます。

### MLflow による実験管理

各実行は MLflow に記録され、以下が追跡されます：

- 使用プロンプト・評価項目のテキスト
- 生成された回答
- 評価スコア（数値メトリクス）と評価理由

これによりプロンプト改善のイテレーションを定量的に管理できます。

---

## 実行方法

Databricks ノートブック `src/main_notebook.ipynb` を開き、セルを順番に実行してください。

```python
# 依存ライブラリのインストール
!pip install openai==2.5.0 mlflow==3.5.0 pyyaml==6.0.2

# 実行
from customer_summarize_prompt import CustomerSummarizePrompt

customer_prompt = CustomerSummarizePrompt(experiment_base_path="/Workspace/your/path/experiments")
result = customer_prompt.run()
```

---

## 注意点

### 実行環境

- **Databricks 専用**：`dbutils`（Databricks Utilities）を使用しているため、ローカル環境やその他クラウドでは動作しません。

### APIキーの管理

- OpenAI の API キーは Databricks Secrets に保存する必要があります。
  - スコープ名: `my-secrets`
  - キー名: `openai-api-key`
- コード中にキーをハードコードしないでください。

### クラス名とYAMLキーの対応

- `prompt.yaml` のセクションキーは **クラス名と完全一致** している必要があります。
  - 例: クラス名が `CustomerSummarizePrompt` であれば、YAMLのキーも `CustomerSummarizePrompt` にする。
- クラス名が一致しない場合、プロンプト・評価項目の読み込みが空になりサイレントに失敗します。

### 評価JSONの形式

- `evaluate()` は LLM のレスポンスを `json.loads()` でパースするため、評価プロンプトテンプレートの JSON 形式指示を変更する際は注意が必要です。パースに失敗した場合はエラーが `evaluation_data` に格納されます。

### MLflow 実験パス

- Databricks Workspace 上の有効なパスを `experiment_base_path` に指定してください。
- 権限がないパスを指定した場合、フォールバックとして `/tmp/` 以下に一時実験が作成されます。

---

## 新しいユースケースの追加方法

1. `BacePrompt` を継承したクラスを `src/` に作成する
2. `generate()` / `evaluate()` / `_setup_mlflow_experiment()` を実装する
3. `config/prompt.yaml` にクラス名と同じキーでプロンプトと評価項目を定義する
4. `main_notebook.ipynb` からインポートして実行する
