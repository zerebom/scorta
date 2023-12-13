import polars as pl

from scorta.feature.feature import SampleFeature


class TestSampleFeature:
    def test_generate(self) -> None:
        # テスト用のインスタンスを作成
        feature = SampleFeature(output_dir="test_output", feature_cols=["feature"])

        # generateメソッドを呼び出し
        feature_df = feature.fit()

        # 生成されたDataFrameが期待通りか確認
        assert isinstance(feature_df, pl.DataFrame), "generateメソッドはpolarsのDataFrameを返すべきです"
        assert feature_df.shape[0] == 9, "生成されたDataFrameの行数は9であるべきです"
        assert "feature" in feature_df.columns, "'feature'列がDataFrameに含まれているべきです"

    def test_save(self) -> None:
        feature = SampleFeature(output_dir="test_output")
        feature.save(feature.fit())
        assert feature.output_path.exists(), "saveメソッドで指定したパスにファイルが生成されているべきです"
