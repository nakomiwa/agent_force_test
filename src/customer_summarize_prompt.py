from base_prompt import BacePrompt
import mlflow
import datetime
import json
from typing import Dict, Any

class CustomerSummarizePrompt(BacePrompt):
    def __init__(self, experiment_base_path: str = "/Workspace/b_poc/b_0048/experiments"):
        super().__init__(experiment_base_path)
        self.temperature = 0
    
    def generate(self):
        print(f"🔄 {self.class_name}: 顧客要約生成中...")
        
        prompt_data = self.load_yaml("prompt.yaml")
        prompt = prompt_data.get('prompt', '')
        
        if not prompt:
            print(f"⚠️ プロンプトが見つかりません。config/prompt.yamlを確認してください。")
            return ""
        
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
            "task_type": "顧客要約"
        }, "answer.yaml")
        
        return answer
    
    def evaluate(self):
        print(f"🔄 {self.class_name}: 顧客要約評価中...")
        
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
            temperature=self.temperature
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
            "eval_prompt_preview": eval_prompt[:100] + "..." if len(eval_prompt) > 100 else eval_prompt,
            "task_type": "顧客要約"
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
            mlflow.log_param("タスク種別", "顧客要約")
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
            if 'eval_prompt' in prompt_data:
                mlflow.log_text(prompt_data['eval_prompt'], f"{self.class_name}_評価プロンプト.txt")
            
            # ============================================================
            # 3. 生成された要約をログ
            # ============================================================
            mlflow.log_text(answer, f"{self.class_name}_要約.txt")
            mlflow.log_param("要約文字数", len(answer))
            
            # ============================================================
            # 4. 評価結果をログ（親クラスの評価項目 + 子クラス特有の評価項目）
            # ============================================================
            if isinstance(evaluation_data, dict):
                # 親クラスの評価項目（簡潔さ、一貫性）
                if '簡潔さ' in evaluation_data:
                    mlflow.log_metric("簡潔さ", evaluation_data['簡潔さ'])
                if '一貫性' in evaluation_data:
                    mlflow.log_metric("一貫性", evaluation_data['一貫性'])
                
                # 子クラス特有の評価項目（要約の正確性）
                if '要約の正確性' in evaluation_data:
                    mlflow.log_metric("要約の正確性", evaluation_data['要約の正確性'])
                
                # ============================================================
                # 5. 総合スコア計算とログ
                # ============================================================
                scores = []
                if '簡潔さ' in evaluation_data:
                    scores.append(evaluation_data['簡潔さ'])
                if '一貫性' in evaluation_data:
                    scores.append(evaluation_data['一貫性'])
                if '要約の正確性' in evaluation_data:
                    scores.append(evaluation_data['要約の正確性'])
                
                if scores:
                    avg_score = sum(scores) / len(scores)
                    mlflow.log_metric("総合スコア", avg_score)
                    mlflow.log_metric("最小スコア", min(scores))
                    mlflow.log_metric("最大スコア", max(scores))
                
                # 評価理由があればログ
                if '評価理由' in evaluation_data:
                    mlflow.log_text(str(evaluation_data['評価理由']), f"{self.class_name}_評価理由.txt")
            
            print(f"✅ MLflow実行完了: {run_name}")