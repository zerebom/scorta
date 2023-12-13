import polars as pl

from scorta.recsys.candidate_generate import Evaluator, SampleCandidate


class TestSampleCandidate:
    def test_generate(self) -> None:
        # テスト用のインスタンスを作成
        candidate_generator = SampleCandidate(output_dir="test_output")

        # generateメソッドを呼び出し
        generated_df = candidate_generator.generate()

        # 生成されたDataFrameが期待通りか確認
        assert isinstance(generated_df, pl.DataFrame), "generateメソッドはpolarsのDataFrameを返すべきです"
        assert generated_df.shape[0] == 9, "生成されたDataFrameの行数は9であるべきです"
        assert "score" in generated_df.columns, "'score'列がDataFrameに含まれているべきです"

    def test_save(self) -> None:
        candidate = SampleCandidate(output_dir="test_output")
        candidate.save(candidate.generate())
        assert candidate.output_path.exists(), "saveメソッドで指定したパスにファイルが生成されているべきです"


class TestEvaluator:
    def test_evaluate(self) -> None:
        k = 1
        # テスト用のDataFrameを作成
        target_df = pl.DataFrame(
            {
                "user_id": [1, 1, 2],
                "item_id": [1, 2, 1],
                "target": [1, 0, 1],
            }
        )
        evaluator = Evaluator(evaluate_topks=[k], target_df=target_df)
        df = pl.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "item_id": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "score": [0.1, 0.2, 0.3, 0.2, 0.4, 0.6, 0.3, 0.6, 0.9],
                "rank": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            }
        )

        # evaluateメソッドを呼び出し
        result = evaluator.evaluate(df, "test_candidate")

        # 期待通りの結果が得られているか確認
        assert isinstance(result, dict), "evaluateメソッドはdictを返すべきです"
        assert "cg_name" in result[k], "ndcgが結果に含まれているべきです"
        assert f"precision@{k}" in result[k]["eval_dic"][1], "precisionが結果に含まれているべきです"
