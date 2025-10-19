from base_prompt import BacePrompt
import mlflow
import datetime
import json
import pandas as pd
import os
from typing import Dict, Any

class CustomerSummarizePrompt(BacePrompt):
    """
    顧客要約生成と評価を行うクラス
    
    機能：
    - 顧客データの読み込み
    - 営業向け要約の生成
    - 共通評価項目＋固有評価項目による評価
    - MLflowへの結果記録
    """
    
    def __init__(self, experiment_base_path: str = "/Workspace/b_poc/b_0048/experiments"):
        """
        初期化処理
        
        Args:
            experiment_base_path: MLflow実験のベースパス
        """
        super().__init__(experiment_base_path)
        self.temperature = 0
    
    def _setup_mlflow_experiment(self):
        """
        MLflow実験の設定（抽象メソッドの実装）
        """
        try:
            # 実験が既に存在するかチェック
            experiment = mlflow.get_experiment_by_name(self.experiment_path)
            
            if experiment is None:
                # 実験が存在しない場合は作成
                experiment_id = mlflow.create_experiment(self.experiment_path)
                mlflow.set_experiment(experiment_id=experiment_id)
            else:
                # 実験が存在する場合は設定
                mlflow.set_experiment(experiment_id=experiment.experiment_id)
                
        except Exception as e:
            print(f"MLflow実験設定エラー: {e}")
    
    def load_customer_data(self) -> str:
        """
        顧客データをCSVから読み込み
        
        Returns:
            顧客データの文字列表現
        """
        csv_path = os.path.join(self.config_dir, "customer_data.csv")
        
        if not os.path.exists(csv_path):
            return ""
        
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            return df.to_string(index=False)
        except Exception as e:
            print(f"顧客データ読み込みエラー: {e}")
            return ""
    
    def generate(self) -> str:
        """
        顧客要約を生成（抽象メソッドの実装）
        
        Returns:
            生成された顧客要約
        """
        # 顧客データの読み込み
        customer_data = self.load_customer_data()
        if not customer_data:
            return ""
        
        # プロンプトテンプレートの取得
        prompt_data = self.load_yaml(self.class_name, "prompt.yaml")
        prompt_template = prompt_data.get('generate_prompt', '')
        
        if not prompt_template:
            return ""

        # プロンプト作成
        if '{customer_data}' in prompt_template:
            prompt = prompt_template.format(customer_data=customer_data)
        else:
            prompt = f"{prompt_template}\n\n顧客データ:\n{customer_data}"
        
        # LLMに要約生成を依頼
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        
        answer = response.choices[0].message.content
        
        # 回答をYAMLに保存
        self.save_yaml({
            "answer": answer, 
            "generated_at": datetime.datetime.now().isoformat(),
            "task_type": "顧客要約"
        }, "answer.yaml")
        
        return answer
    
    def evaluate(self, answer: str = None) -> Dict[str, Any]:
        """
        共通評価項目＋固有評価項目による統合評価（抽象メソッドの実装）
        
        Args:
            answer: 評価対象の回答
            
        Returns:
            評価結果
        """
        # 回答の取得
        if answer is None:
            answer_data = self.load_yaml(self.class_name, "answer.yaml")
            answer = answer_data.get('answer', '')
        
        if not answer:
            return {"error": "回答が見つかりません"}
        
        # 共通評価項目を取得
        common_data = self.load_yaml('Common', "prompt.yaml")
        common_eval_items = common_data.get('eval_items', [])
        
        # 子クラス固有の評価項目を取得
        class_data = self.load_yaml(self.class_name, "prompt.yaml")
        class_eval_items = class_data.get('eval_items', [])
        
        # 評価項目を統合
        all_eval_items = common_eval_items + class_eval_items
        
        if not all_eval_items:
            return {"error": "評価項目が見つかりません"}
        
        # 統合評価プロンプトを作成
        eval_items_text = "\n".join([f"- {item}" for item in all_eval_items])
        
        eval_prompt_template = common_data.get('evaluate_prompt', '')
        if not eval_prompt_template:
            return {"error": "評価プロンプトテンプレートが見つかりません"}
        
        eval_prompt = eval_prompt_template.format(
            answer=answer,
            eval_items=eval_items_text
        )
        
        # LLMに統合評価を依頼
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.0
            )
            
            evaluation_result = response.choices[0].message.content
            evaluation_data = json.loads(evaluation_result)
                
        except Exception as e:
            evaluation_data = {"error": f"評価エラー: {str(e)}"}
        
        # MLFlowに記録
        self._log_to_mlflow(answer, evaluation_data, class_data, common_data)
        
        return evaluation_data
    
    def _log_to_mlflow(self, answer: str, evaluation_data: Dict[str, Any], prompt_data: Dict[str, Any], common_data: Dict[str, Any]):
        """
        MLflowに実行結果を記録
        
        Args:
            answer: 生成された回答
            evaluation_data: 評価結果
            prompt_data: プロンプトデータ
            common_data: 共通設定データ
        """
        try:
            # 実験設定
            self._setup_mlflow_experiment()
            
            run_name = f"{self.class_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(run_name=run_name):
                # 基本メタデータ
                mlflow.log_param("クラス名", self.class_name)
                mlflow.log_param("実行日時", datetime.datetime.now().isoformat())
                mlflow.log_param("モデル名", self.model_name)
                mlflow.log_param("Temperature", self.temperature)
                
                # プロンプトをログ
                if 'generate_prompt' in prompt_data:
                    mlflow.log_text(prompt_data['generate_prompt'], f"{self.class_name}_プロンプト.txt")
                
                # 評価項目をログ
                if 'eval_items' in common_data:
                    common_items_text = "\n".join(common_data['eval_items'])
                    mlflow.log_text(common_items_text, f"{self.class_name}_共通評価項目.txt")
                
                if 'eval_items' in prompt_data:
                    class_items_text = "\n".join(prompt_data['eval_items'])
                    mlflow.log_text(class_items_text, f"{self.class_name}_固有評価項目.txt")
                
                # 回答をログ
                mlflow.log_text(answer, f"{self.class_name}_回答.txt")
                mlflow.log_param("回答文字数", len(answer))
                
                # 評価結果をログ
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
                    
                    # 評価理由をログ
                    if '評価理由' in evaluation_data:
                        mlflow.log_text(str(evaluation_data['評価理由']), f"{self.class_name}_評価理由.txt")
                        
        except Exception as e:
            print(f"MLflowログ記録エラー: {e}")