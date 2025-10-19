import yaml
import json
from openai import OpenAI
from typing import Dict, Any
import datetime
from databricks.sdk.runtime import dbutils
import logging
import os
from abc import ABC, abstractmethod

class BacePrompt(ABC):
    """
    LLMプロンプト実行の基底抽象クラス
    
    共通機能：
    - YAMLファイルの読み書き
    - LLMクライアントの設定
    - プロジェクト構造の管理
    """
    
    def __init__(self, experiment_base_path: str = "/Workspace/b_poc/b_0048/experiments"):
        """
        初期化処理
        
        Args:
            experiment_base_path: MLflow実験のベースパス
        """
        self.dbutils = dbutils
        self.llm_client = None
        self._setup_llm_client()
        self.class_name = self.__class__.__name__
        
        # モデル名を初期化時に設定
        self.model_name = "gpt-4o-mini"
        
        # 実験のベースパスを設定
        self.experiment_base_path = experiment_base_path
        self.experiment_path = f"{experiment_base_path}/{self.class_name}"
        
        # プロジェクトルートの設定
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_dir = os.path.join(self.project_root, "config")
        
        # MLflowのログレベルを調整
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        
    def _setup_llm_client(self):
        """LLMクライアントの設定"""
        scope = "my-secrets"
        key = "openai-api-key"
        api_key = self.dbutils.secrets.get(scope=scope, key=key)
        self.llm_client = OpenAI(api_key=api_key)
    
    def load_yaml(self, section_name: str, filename: str = "prompt.yaml") -> Dict[str, Any]:
        """
        指定されたセクションのYAMLデータを読み込む
        
        Args:
            section_name: 読み込むセクション名（例: クラス名, 'Common'）
            filename: config/フォルダ内のファイル名
            
        Returns:
            YAMLデータ（指定セクション）
        """
        yaml_path = os.path.join(self.config_dir, filename)
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
                return data.get(section_name, {}) if data else {}
        except FileNotFoundError:
            return {}
        except Exception as e:
            print(f"YAMLファイル読み込みエラー: {e}")
            return {}
    
    def save_yaml(self, content: Dict[str, Any], filename: str = "answer.yaml"):
        """
        YAMLファイルにクラス固有データを保存
        
        Args:
            content: 保存する内容
            filename: config/フォルダ内のファイル名
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
            os.makedirs(self.config_dir, exist_ok=True)
            with open(yaml_path, 'w', encoding='utf-8') as file:
                yaml.safe_dump(data, file, allow_unicode=True, default_flow_style=False)
                
        except Exception as e:
            print(f"YAMLファイル保存エラー: {e}")
    
    @abstractmethod
    def _setup_mlflow_experiment(self):
        """
        MLflow実験の設定（子クラスで実装必須）
        """
        pass
    
    @abstractmethod
    def generate(self) -> str:
        """
        回答を生成（子クラスで実装必須）
        
        Returns:
            生成された回答
        """
        pass
    
    @abstractmethod
    def evaluate(self, answer: str = None) -> Dict[str, Any]:
        """
        回答を評価（子クラスで実装必須）
        
        Args:
            answer: 評価対象の回答
            
        Returns:
            評価結果
        """
        pass
    
    def run(self):
        """
        生成と評価を一括実行
        
        Returns:
            実行結果（回答と評価）
        """
        # 回答生成
        answer = self.generate()
        if not answer:
            return {"error": "回答生成に失敗しました"}
        
        # 評価実行
        evaluation = self.evaluate(answer)
        
        return {"answer": answer, "evaluation": evaluation}