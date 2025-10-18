import yaml
import json
import mlflow
from openai import OpenAI
from typing import Dict, Any
import datetime
from databricks.sdk.runtime import dbutils
import logging
import os

class BacePrompt:
    def __init__(self, experiment_base_path: str = "/Workspace/b_poc/b_0048/experiments"):
        self.dbutils = dbutils
        self.llm_client = None
        self._setup_llm_client()
        self.class_name = self.__class__.__name__
        
        # 実験のベースパスを設定
        self.experiment_base_path = experiment_base_path
        self.experiment_path = f"{experiment_base_path}/{self.class_name}"
        
        # プロジェクトルートの設定
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_dir = os.path.join(self.project_root, "config")
        
        # MLflowのログレベルを調整
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        
    def _setup_llm_client(self):
        scope = "my-secrets"
        key = "openai-api-key"
        api_key = self.dbutils.secrets.get(scope=scope, key=key)
        self.llm_client = OpenAI(api_key=api_key)
    
    def load_yaml(self, filename: str = "prompt.yaml") -> Dict[str, Any]:
        """
        YAMLファイルを読み込む
        Args:
            filename: config/フォルダ内のファイル名（デフォルト: prompt.yaml）
        """
        yaml_path = os.path.join(self.config_dir, filename)
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
                return data.get(self.class_name, {}) if data else {}
        except FileNotFoundError:
            print(f"⚠️ ファイルが見つかりません: {yaml_path}")
            return {}
        except Exception as e:
            print(f"❌ YAMLファイル読み込みエラー: {e}")
            return {}
    
    def save_yaml(self, content: Dict[str, Any], filename: str = "answer.yaml"):
        """
        YAMLファイルに保存する
        Args:
            content: 保存する内容
            filename: config/フォルダ内のファイル名（デフォルト: answer.yaml）
        """
        yaml_path = os.path.join(self.config_dir, filename)
        
        try:
            # 既存ファイルを読み込み
            try:
                with open(yaml_path, 'r', encoding='utf-8') as file:
                    data = yaml.safe_load(file) or {}
            except FileNotFoundError:
                data = {}
            
            # クラス名をキーとして内容を更新
            data[self.class_name] = content
            
            # ファイルに保存
            os.makedirs(self.config_dir, exist_ok=True)  # ディレクトリがない場合は作成
            with open(yaml_path, 'w', encoding='utf-8') as file:
                yaml.safe_dump(data, file, allow_unicode=True, default_flow_style=False)
                
            print(f"✅ ファイル保存完了: {yaml_path}")
            
        except Exception as e:
            print(f"❌ YAMLファイル保存エラー: {e}")
    
    def _setup_mlflow_experiment(self):
        """MLflow実験の設定を行う"""
        try:
            # 実験が既に存在するかチェック
            experiment = mlflow.get_experiment_by_name(self.experiment_path)
            
            if experiment is None:
                # 実験が存在しない場合は作成
                print(f"🔄 MLflow実験を作成中: {self.experiment_path}")
                experiment_id = mlflow.create_experiment(self.experiment_path)
                mlflow.set_experiment(experiment_id=experiment_id)
                print(f"✅ MLflow実験を作成しました: {self.experiment_path}")
            else:
                # 実験が存在する場合は設定
                mlflow.set_experiment(experiment_id=experiment.experiment_id)
                print(f"✅ 既存のMLflow実験を使用: {self.experiment_path}")
                
        except Exception as e:
            # フォールバック: 実験名で直接設定を試行
            print(f"⚠️ 実験設定中にエラーが発生しました: {e}")
            try:
                print(f"🔄 フォールバック方式で実験を設定中: {self.experiment_path}")
                mlflow.set_experiment(self.experiment_path)
                print(f"✅ フォールバック設定完了: {self.experiment_path}")
            except Exception as fallback_error:
                print(f"❌ フォールバック設定も失敗しました: {fallback_error}")
                # 最後の手段として一時的な実験名を使用
                temp_experiment = f"/tmp/{self.class_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                print(f"🔄 一時的な実験を作成: {temp_experiment}")
                mlflow.set_experiment(temp_experiment)
    
    def generate(self):
        print(f"🔄 {self.class_name}: 回答生成中...")
        
        prompt_data = self.load_yaml("prompt.yaml")
        prompt = prompt_data.get('prompt', '')
        
        if not prompt:
            print(f"⚠️ プロンプトが見つかりません。config/prompt.yamlを確認してください。")
            return ""
        
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        answer = response.choices[0].message.content
        
        self.save_yaml({
            "answer": answer, 
            "generated_at": datetime.datetime.now().isoformat(),
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt
        }, "answer.yaml")
        
        return answer
    
    def evaluate(self):
        print(f"🔄 {self.class_name}: 評価実行中...")
        
        # generateメソッドで生成した回答をロード
        answer_data = self.load_yaml("answer.yaml")
        answer = answer_data.get('answer', '')
        
        if not answer:
            print(f"⚠️ 評価対象の回答が見つかりません。先にgenerate()を実行してください。")
            return {"error": "回答が見つかりません"}
        
        # 評価プロンプトをロード
        prompt_data = self.load_yaml("prompt.yaml")
        eval_prompt_template = prompt_data.get('eval_prompt', '')
        
        if not eval_prompt_template:
            print(f"⚠️ 評価プロンプトが見つかりません。config/prompt.yamlを確認してください。")
            return {"error": "評価プロンプトが見つかりません"}
        
        eval_prompt = eval_prompt_template.format(answer=answer)
        
        # LLMに評価させる
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.0
        )
        
        evaluation_result = response.choices[0].message.content
        
        try:
            evaluation_data = json.loads(evaluation_result)
        except json.JSONDecodeError:
            print(f"⚠️ 評価結果のJSON解析に失敗しました")
            evaluation_data = {"error": "評価結果の解析に失敗しました", "raw_result": evaluation_result}
        
        # 評価結果を保存
        self.save_yaml({
            "evaluation": evaluation_data,
            "evaluated_at": datetime.datetime.now().isoformat(),
            "eval_prompt_preview": eval_prompt[:100] + "..." if len(eval_prompt) > 100 else eval_prompt
        }, "evaluation.yaml")
        
        # MLFlowに記録
        self._log_to_mlflow(answer, evaluation_data, prompt_data)
        
        return evaluation_data
    
    def _log_to_mlflow(self, answer: str, evaluation_data: Dict[str, Any], prompt_data: Dict[str, Any]):
        # 実験設定
        self._setup_mlflow_experiment()
        
        # run_nameを指定してMLflow実行開始
        run_name = f"{self.class_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🚀 MLflow実行開始: {run_name}")
        
        with mlflow.start_run(run_name=run_name):
            # ============================================================
            # 1. クラス・プロンプトメタデータをログ
            # ============================================================
            mlflow.log_param("クラス名", self.class_name)
            mlflow.log_param("実行日時", datetime.datetime.now().isoformat())
            mlflow.log_param("モデル名", "gpt-4o-mini")
            mlflow.log_param("Temperature", 0.0)
            mlflow.log_param("プロジェクトルート", self.project_root)
            mlflow.log_param("実験パス", self.experiment_path)
            
            # ============================================================
            # 2. プロンプトをログ
            # ============================================================
            if 'prompt' in prompt_data:
                mlflow.log_text(prompt_data['prompt'], f"{self.class_name}_プロンプト.txt")
            if 'eval_prompt' in prompt_data:
                mlflow.log_text(prompt_data['eval_prompt'], f"{self.class_name}_評価プロンプト.txt")
            
            # ============================================================
            # 3. 生成された回答をログ
            # ============================================================
            mlflow.log_text(answer, f"{self.class_name}_回答.txt")
            mlflow.log_param("回答文字数", len(answer))
            
            # ============================================================
            # 4. 評価結果をログ（親クラスではコアな評価のみ）
            # ============================================================
            if isinstance(evaluation_data, dict):
                if '簡潔さ' in evaluation_data:
                    mlflow.log_metric("簡潔さ", evaluation_data['簡潔さ'])
                if '一貫性' in evaluation_data:
                    mlflow.log_metric("一貫性", evaluation_data['一貫性'])
                
                # 評価理由があればログ
                if '評価理由' in evaluation_data:
                    mlflow.log_text(str(evaluation_data['評価理由']), f"{self.class_name}_評価理由.txt")
            
            print(f"✅ MLflow実行完了: {run_name}")
    
    def run(self):
        print(f"🚀 {self.class_name} 実行開始...")
        print(f"📁 プロジェクトルート: {self.project_root}")
        print(f"📁 Configディレクトリ: {self.config_dir}")
        print(f"📊 MLflow実験パス: {self.experiment_path}")
        
        answer = self.generate()
        if not answer:
            return {"error": "回答生成に失敗しました"}
        
        print(f"✅ 回答生成完了 (文字数: {len(answer)})")
        
        evaluation = self.evaluate()
        print(f"✅ 評価完了")
        
        return {"answer": answer, "evaluation": evaluation}