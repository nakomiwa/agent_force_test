from base_prompt import BacePrompt
import mlflow
import datetime
import json
import pandas as pd
import os
from typing import Dict, Any

class CustomerSummarizePrompt(BacePrompt):
    def __init__(self, experiment_base_path: str = "/Workspace/b_poc/b_0048/experiments"):
        super().__init__(experiment_base_path)
        self.temperature = 0
    
    def _setup_mlflow_experiment(self):
        """MLflow実験の設定を行う（抽象メソッドの実装）"""
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
    
    def load_customer_data(self) -> str:
        """顧客データをCSVから読み込み、文字列として返す"""
        csv_path = os.path.join(self.config_dir, "customer_data.csv")
        
        if not os.path.exists(csv_path):
            print(f"❌ 顧客データファイルが見つかりません: {csv_path}")
            if os.path.exists(self.config_dir):
                files = os.listdir(self.config_dir)
                print(f"🔍 Configディレクトリの内容: {files}")
            return ""
        
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            print(f"✅ 顧客データを読み込みました: {len(df)}件")
            customer_data_text = df.to_string(index=False)
            return customer_data_text
            
        except Exception as e:
            print(f"❌ 顧客データ読み込みエラー: {e}")
            return ""
    
    def generate(self) -> str:
        """
        顧客要約を生成する（抽象メソッドの実装）
        """
        print(f"🔄 {self.class_name}: 顧客要約生成中...")
        
        customer_data = self.load_customer_data()
        if not customer_data:
            print(f"⚠️ 顧客データが読み込めませんでした。")
            return ""
        
        prompt_data = self.load_yaml("prompt.yaml")
        prompt_template = prompt_data.get('prompt', '')
        
        if not prompt_template:
            print(f"⚠️ プロンプトが見つかりません。config/prompt.yamlの{self.class_name}セクションを確認してください。")
            return ""

        if '{customer_data}' in prompt_template:
            prompt = prompt_template.format(customer_data=customer_data)
        else:
            prompt = f"{prompt_template}\n\n顧客データ:\n{customer_data}"
        
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        
        answer = response.choices[0].message.content
        
        self.save_yaml({
            "answer": answer, 
            "generated_at": datetime.datetime.now().isoformat(),
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "task_type": "顧客要約",
            "customer_data_preview": customer_data[:200] + "..." if len(customer_data) > 200 else customer_data,
            "customer_data_length": len(customer_data),
            "final_prompt_length": len(prompt)
        }, "answer.yaml")
        
        return answer
    
    def evaluate(self, answer: str = None) -> Dict[str, Any]:
        """
        共通評価項目 + 子クラス固有評価項目を統合して評価（抽象メソッドの実装）
        """
        print(f"🔄 {self.class_name}: 統合評価実行中...")
        
        # 回答の取得
        if answer is None:
            answer_data = self.load_yaml("answer.yaml")
            answer = answer_data.get('answer', '')
        
        if not answer:
            print(f"⚠️ 評価対象の回答が見つかりません。")
            return {"error": "回答が見つかりません"}
        
        # 1. 共通評価項目を取得
        common_data = self.load_common_yaml("prompt.yaml")
        common_eval_items = common_data.get('eval_items', [])
        
        # 2. 子クラス固有の評価項目を取得
        class_data = self.load_yaml("prompt.yaml")
        class_eval_items = class_data.get('eval_items', [])
        
        # 3. 評価項目を統合
        all_eval_items = common_eval_items + class_eval_items
        
        if not all_eval_items:
            print(f"⚠️ 評価項目が見つかりません。")
            return {"error": "評価項目が見つかりません"}
        
        # 4. 統合評価プロンプトを作成
        eval_items_text = "\n".join([f"- {item}" for item in all_eval_items])
        
        eval_prompt_template = common_data.get('eval_prompt_template', '')
        if not eval_prompt_template:
            print(f"⚠️ 評価プロンプトテンプレートが見つかりません。")
            return {"error": "評価プロンプトテンプレートが見つかりません"}
        
        eval_prompt = eval_prompt_template.format(
            answer=answer,
            eval_items=eval_items_text
        )
        
        # 5. LLMに統合評価をさせる
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.0
            )
            
            evaluation_result = response.choices[0].message.content
            
            try:
                evaluation_data = json.loads(evaluation_result)
                print(f"✅ 統合評価完了: {evaluation_data}")
            except json.JSONDecodeError:
                print(f"⚠️ 評価結果のJSON解析に失敗しました")
                evaluation_data = {"error": "評価結果の解析に失敗しました", "raw_result": evaluation_result}
                
        except Exception as e:
            print(f"❌ 統合評価でエラー: {e}")
            evaluation_data = {"error": f"統合評価エラー: {str(e)}"}
        
        # 6. 評価結果を保存
        combined_evaluation = {
            "evaluation": evaluation_data,
            "common_eval_items": common_eval_items,
            "class_eval_items": class_eval_items,
            "evaluated_at": datetime.datetime.now().isoformat(),
            "eval_prompt_preview": eval_prompt[:100] + "..." if len(eval_prompt) > 100 else eval_prompt
        }
        
        self.save_yaml(combined_evaluation, "evaluation.yaml")
        
        # 7. MLFlowに記録
        try:
            self._log_to_mlflow(answer, evaluation_data, class_data, common_data)
        except Exception as e:
            print(f"⚠️ MLflow記録でエラー: {e}")
        
        return combined_evaluation
    
    def _log_to_mlflow(self, answer: str, evaluation_data: Dict[str, Any], prompt_data: Dict[str, Any], common_data: Dict[str, Any]):
        """
        CustomerSummarizePrompt用のMLflowログ
        """
        print(f"🔄 MLflowログ記録開始...")
        
        # 実験設定
        self._setup_mlflow_experiment()
        
        run_name = f"{self.class_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"🚀 MLflow実行開始: {run_name}")
        
        with mlflow.start_run(run_name=run_name):
            # ============================================================
            # 1. 基本メタデータをログ
            # ============================================================
            mlflow.log_param("クラス名", self.class_name)
            mlflow.log_param("実行日時", datetime.datetime.now().isoformat())
            mlflow.log_param("モデル名", "gpt-4o-mini")
            mlflow.log_param("Temperature", self.temperature)
            mlflow.log_param("プロジェクトルート", self.project_root)
            mlflow.log_param("実験パス", self.experiment_path)
            
            # ============================================================
            # 2. プロンプトをログ
            # ============================================================
            if 'prompt' in prompt_data:
                mlflow.log_text(prompt_data['prompt'], f"{self.class_name}_プロンプト.txt")
            
            # 共通評価項目をログ
            if 'eval_items' in common_data:
                common_items_text = "\n".join(common_data['eval_items'])
                mlflow.log_text(common_items_text, f"{self.class_name}_共通評価項目.txt")
            
            # クラス固有評価項目をログ
            if 'eval_items' in prompt_data:
                class_items_text = "\n".join(prompt_data['eval_items'])
                mlflow.log_text(class_items_text, f"{self.class_name}_固有評価項目.txt")
            
            # ============================================================
            # 3. 生成された回答をログ
            # ============================================================
            mlflow.log_text(answer, f"{self.class_name}_回答.txt")
            mlflow.log_param("回答文字数", len(answer))
            
            # ============================================================
            # 4. 評価結果をログ
            # ============================================================
            if isinstance(evaluation_data, dict) and 'error' not in evaluation_data:
                # 共通評価項目のスコア
                if '簡潔さ' in evaluation_data:
                    mlflow.log_metric("共通_簡潔さ", evaluation_data['簡潔さ'])
                if '一貫性' in evaluation_data:
                    mlflow.log_metric("共通_一貫性", evaluation_data['一貫性'])
                
                # クラス固有評価項目のスコア
                if '要約の正確性' in evaluation_data:
                    mlflow.log_metric("固有_要約の正確性", evaluation_data['要約の正確性'])
                if '営業有用性' in evaluation_data:
                    mlflow.log_metric("固有_営業有用性", evaluation_data['営業有用性'])
                
                # 評価理由があればログ
                if '評価理由' in evaluation_data:
                    mlflow.log_text(str(evaluation_data['評価理由']), f"{self.class_name}_評価理由.txt")
            
            print(f"✅ MLflow実行完了: {run_name}")
            print(f"   - 評価結果: {evaluation_data}")