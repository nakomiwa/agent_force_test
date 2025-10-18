import yaml
import json
from openai import OpenAI
from typing import Dict, Any
import datetime
from databricks.sdk.runtime import dbutils
import logging
import os
from abc import ABC, abstractmethod

class BacePrompt(ABC):  # 抽象クラスに変更
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
    
    def load_common_yaml(self, filename: str = "prompt.yaml") -> Dict[str, Any]:
        """
        YAMLファイルのCommonセクションを読み込む
        Args:
            filename: config/フォルダ内のファイル名（デフォルト: prompt.yaml）
        """
        yaml_path = os.path.join(self.config_dir, filename)
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
                return data.get('Common', {}) if data else {}
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
    
    @abstractmethod
    def _setup_mlflow_experiment(self):
        """
        MLflow実験の設定を行う（子クラスで実装必須）
        """
        pass
    
    @abstractmethod
    def generate(self) -> str:
        """
        回答を生成する（子クラスで実装必須）
        Returns:
            生成された回答
        """
        pass
    
    @abstractmethod
    def evaluate(self, answer: str = None) -> Dict[str, Any]:
        """
        評価を実行する（子クラスで実装必須）
        Args:
            answer: 評価対象の回答（省略時はanswer.yamlから読み込み）
        Returns:
            評価結果
        """
        pass
    
    def run(self):
        """
        生成と評価を一括実行
        """
        print(f"🚀 {self.class_name} 実行開始...")
        print(f"📁 プロジェクトルート: {self.project_root}")
        print(f"📁 Configディレクトリ: {self.config_dir}")
        print(f"📊 MLflow実験パス: {self.experiment_path}")
        
        # 回答生成
        answer = self.generate()
        if not answer:
            return {"error": "回答生成に失敗しました"}
        
        print(f"✅ 回答生成完了 (文字数: {len(answer)})")
        
        # 評価実行
        evaluation = self.evaluate(answer)
        print(f"✅ 評価完了")
        
        return {"answer": answer, "evaluation": evaluation}