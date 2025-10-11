# Agentforce プロンプト評価フレームワーク

Databricks + MLflowでAgentforce用プロンプトを体系的に評価するフレームワーク。

## 特徴

- **YAML完全管理**: プロンプト、システムプロンプト、評価設定、テストデータをすべてYAML管理
- **LLM-as-a-Judge**: GPT-4による自動評価（簡潔性・明瞭性・実用性）
- **MLflow統合**: 実験結果・メトリクス・レポートを一元管理

## セットアップ

```python
!pip install langchain==0.3.27 langchain-core==0.3.79 openai==2.3.0 mlflow==3.4.0 pandas==2.2.3 pyyaml==6.0.2
```

Databricks Secretsに`openai-api-key`を登録。

## 使い方

1. `prompts.yaml`で新プロンプト追加
2. ノートブック実行
3. MLflowで結果確認
4. プロンプト改善・再評価

## ファイル構成

```
/Workspace/Users/<your-email>/
├── prompts.yaml          # プロンプト定義
├── system_prompts.yaml   # システムプロンプト
├── evaluation.yaml       # 評価設定
├── test_data.yaml        # テストデータ
└── prompt_evaluation.py  # 評価ノートブック
```

詳細は各YAMLファイルとノートブックコメント参照。
