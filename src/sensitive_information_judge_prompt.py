from base_prompt import BacePrompt
import pandas as pd
from pyspark.sql import SparkSession

class SensitiveInformationJudgePrompt(BacePrompt):
    """
    顧客要約生成と評価を行うクラス
    
    機能：
    - 接触履歴データの読み込み
    - テスト用ダミーデータの作成
    - LLMによる機微情報の判断
    """
    
    def __init__(self, experiment_base_path: str = "/Workspace/b_poc/b_0048/experiments"):
        """
        初期化処理
        
        Args:
            experiment_base_path: MLflow実験のベースパス
        """
        super().__init__(experiment_base_path)
        self.temperature = 0
        
        # SparkSessionをインスタンス変数として保持
        self.spark = SparkSession.builder.getOrCreate()
        
        # テーブル名とファイルパスをインスタンス変数として保持
        self.contact_table = "koiso_databircks_18.b_poc.contact"
        self.dummy_data_path = "/Volumes/koiso_databircks_18/b_poc/vol/dummy_contact_data.csv"
        self.judge_data_path = "/Volumes/koiso_databircks_18/b_poc/vol/judge_data.csv"
    
    def create_sample_data(self) -> str:
        """
        接触履歴のサンプルをUnityCatalogのTableに作成
                
        Returns:
            サンプルデータ
        """
        # 接触履歴番号、顧客名、活動内容のサンプルをpandas DataFrameとして作成
        contact_data = pd.DataFrame({
            '接触履歴番号': ["1000001", "1000002", "1000003"],
            '顧客名': ['A株式会社', 'B株式会社', 'C株式会社'],
            '活動内容': ['山田社長に○○商品を提案した', '鈴木社長に△△商品を提案した', '田中社長に△△商品を提案した']})

        # UnityCatalogのcontactテーブルに格納
        # pandas DataFrameをSpark DataFrameに変換
        spark_df = self.spark.createDataFrame(contact_data)
        
        # Unity Catalogのテーブルに保存
        spark_df.write \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(self.contact_table)
        
        print(f"データがテーブル {self.contact_table} に正常に保存されました")
        
        return contact_data.to_string(index=False)

    def create_dummy_data(self) -> str:
        """
        サンプルの接触履歴から読み込み、テスト用のダミーデータを作成
                
        Returns:
            ダミーデータ
        """
        # 接触履歴データのサンプルを読み込み
        sample_df = self.spark.table(self.contact_table).toPandas()
        sample_data = sample_df.to_string(index=False)

        # yamlファイルからダミーデータ作成用プロンプトをロード
        prompt_data = self.load_yaml(self.class_name, "prompt.yaml")
        prompt_template = prompt_data.get('dummy_data_create_prompt', '')
        
        if not prompt_template:
            return ""

        # プロンプトにサンプルデータを組み込む
        prompt = prompt_template.format(sample_data=sample_data)

        # LLMにプロンプトを連携し、ダミーデータを作成
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        
        dummy_data_csv = response.choices[0].message.content
        print(dummy_data_csv)

        # 作成したダミーデータをcsv形式でVolumeに保存
        try:
            with open(self.dummy_data_path, 'w', encoding='utf-8') as file:
                file.write(dummy_data_csv)
            
            print(f"ダミーデータが {self.dummy_data_path} に保存されました")
            
            # 保存したデータを確認
            saved_df = pd.read_csv(self.dummy_data_path)
            print(f"生成されたデータ件数: {len(saved_df)}件")
            
        except Exception as e:
            print(f"ファイル保存エラー: {e}")
            return dummy_data_csv

    def generate(self) -> str:
        """
        LLMによる機微情報の判断
        
        Returns:
            機微情報の判断結果
        """
        # テスト対象の接触履歴データの読み込み
        df_check = pd.read_csv(self.dummy_data_path)
        check_data = df_check.to_string(index=False)

        # プロンプトテンプレートの取得
        prompt_data = self.load_yaml(self.class_name, "prompt.yaml")
        prompt_template = prompt_data.get('generate_prompt', '')
        
        if not prompt_template:
            return ""

        # プロンプト作成
        prompt = prompt_template.format(check_data=check_data)
        
        # LLMに機微情報の判断を依頼
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        
        judge_data_csv = response.choices[0].message.content

        # 作成したダミーデータをcsv形式でVolumeに保存
        try:
            with open(self.judge_data_path, 'w', encoding='utf-8') as file:
                file.write(judge_data_csv)
            
            print(f"判断データが {self.judge_data_path} に保存されました")
            
            # 保存したデータを確認
            saved_df = pd.read_csv(self.judge_data_path)
            print(f"生成されたデータ件数: {len(saved_df)}件")
            
        except Exception as e:
            print(f"ファイル保存エラー: {e}")
            return judge_data_csv
    
    def evaluate(self):
        """
        評価処理
        """
        # 人が評価するためpass
        pass