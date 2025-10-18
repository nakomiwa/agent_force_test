# Databricks notebook source
# MAGIC %md
# MAGIC # OpenAIのAPIキーをシークレットスコープに設定する

# COMMAND ----------

# pipでインストール
!pip install databricks-cli==0.18.0
dbutils.library.restartPython()
# インストール確認
!databricks --version

# COMMAND ----------

# MAGIC %sh
# MAGIC # ===== ここを自分の値に置き換える =====
# MAGIC DATABRICKS_HOST="ZZZ"  # ← Azure DatabricksのURL
# MAGIC DATABRICKS_TOKEN="YYY"  # ← Personal Access Token
# MAGIC OPENAI_API_KEY="XXX"  # ← OpenAI APIキー
# MAGIC # ====================================
# MAGIC
# MAGIC
# MAGIC # CLI設定
# MAGIC cat > ~/.databrickscfg << EOF
# MAGIC [DEFAULT]
# MAGIC host = ${DATABRICKS_HOST}
# MAGIC token = ${DATABRICKS_TOKEN}
# MAGIC EOF
# MAGIC
# MAGIC echo "✅ Databricks CLI設定完了"
# MAGIC
# MAGIC # スコープ作成
# MAGIC databricks secrets create-scope --scope my-secrets 2>/dev/null || echo "⚠️ スコープ既存"
# MAGIC
# MAGIC # シークレット登録（旧CLI用：--string-value の後に値を直接指定）
# MAGIC databricks secrets put --scope my-secrets --key openai-api-key --string-value "${OPENAI_API_KEY}"
# MAGIC
# MAGIC echo "✅ OpenAI APIキー登録完了"
# MAGIC
# MAGIC # 確認
# MAGIC echo ""
# MAGIC echo "📋 登録されたシークレット:"
# MAGIC databricks secrets list --scope my-secrets

# COMMAND ----------


