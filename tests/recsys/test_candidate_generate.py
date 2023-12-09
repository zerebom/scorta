import polars as pl

from scorta.recsys.candidate_generate import SampleCandidate


class TestSampleCandidate:
    def test_generate(self):
        # テスト用のインスタンスを作成
        candidate_generator = SampleCandidate(output_dir="test_output")

        # generateメソッドを呼び出し
        generated_df = candidate_generator.generate()

        # 生成されたDataFrameが期待通りか確認
        assert isinstance(generated_df, pl.DataFrame), "generateメソッドはpolarsのDataFrameを返すべきです"
        assert generated_df.shape[0] == 9, "生成されたDataFrameの行数は9であるべきです"
        assert "score" in generated_df.columns, "'score'列がDataFrameに含まれているべきです"

    def test_save(self):
        candidate = SampleCandidate(output_dir="test_output")
        candidate.save(candidate.generate())
        assert candidate.output_path.exists(), "saveメソッドで指定したパスにファイルが生成されているべきです"
