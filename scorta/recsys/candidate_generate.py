from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl
from tqdm import tqdm


def calc_candidate_metric(
    target_df: pl.DataFrame, cand_df: pl.DataFrame, topk: int, target: int, join_cols: list[str] = ["user_id", "item_id"]
) -> tuple[float, float, float]:
    cand_df = cand_df.filter(pl.col("rank") <= topk).with_columns(pl.lit(1).alias("has_candidate"))
    joined_df = target_df.join(cand_df, on=join_cols, how="outer")
    has_target_df = joined_df.filter(pl.col("target") >= target)
    num_has_target_and_candidate = has_target_df["has_candidate"].is_not_null().sum()

    precision = (num_has_target_and_candidate) / len(cand_df)
    recall = (num_has_target_and_candidate) / len(has_target_df)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


class Evaluator:
    def __init__(
        self,
        target_df: pl.DataFrame | None = None,
        user_col: str = "user_id",
        item_col: str = "item_id",
        evaluate_topks: list[int] = [1, 5, 10, 50, 100],
        target_values: list[int] = [1],
    ):
        self.evaluate_topks = evaluate_topks
        self.target_df = target_df
        self.user_col = user_col
        self.item_col = item_col
        self.target_values = target_values

    def evaluate(self, cand_df: pl.DataFrame, cg_name: str) -> dict[int, dict[str, Any]]:
        summary_dic = {}
        for k in self.evaluate_topks:
            eval_dic: dict[int, dict[str, float]] = {}
            k_cand_df = cand_df.filter(pl.col("rank") <= k)

            if self.target_df is not None:
                for target in self.target_values:
                    eval_dic[target] = {}
                    precision, recall, f1 = calc_candidate_metric(self.target_df, cand_df, k, target, [self.user_col, self.item_col])
                    eval_dic[target][f"precision@{k}"] = precision
                    eval_dic[target][f"recall@{k}"] = recall
                    eval_dic[target][f"f1@{k}"] = f1

            unique_user_cnt = k_cand_df[self.user_col].n_unique()
            unique_item_cnt = k_cand_df[self.item_col].n_unique()
            pair_cnt = len(k_cand_df)

            summary_dic[k] = {
                "eval_time": datetime.now().strftime("%Y/%m/%d/%H:%M:%S"),
                "cg_name": cg_name,
                "pair_cnt": pair_cnt,
                "unique_user_cnt": unique_user_cnt,
                "unique_item_cnt": unique_item_cnt,
                "eval_dic": eval_dic,
            }
        return summary_dic


class Candidate(ABC):
    def __init__(
        self,
        output_dir: Path | str,
        user_col: str = "user_id",
        item_col: str = "item_id",
        suffix: str | None = None,
        target_df: pl.DataFrame | None = None,
        evaluate_topks: list[int] = [1, 5, 10, 50, 100],
        mode: Literal["train", "test"] | None = None,
    ):
        self.class_name = self.__class__.__name__
        self.file_name = self.__class__.__name__
        self.mode = mode
        if suffix is not None:
            self.file_name += f"_{suffix}"

        self.user_col = user_col
        self.item_col = item_col
        self.target_values = [1]

        self.output_cols = [user_col, item_col, "rank", "score"]
        self.output_path = Path(output_dir) / f"{self.file_name}.parquet"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.evaluator = Evaluator(target_df, user_col=user_col, item_col=item_col, evaluate_topks=evaluate_topks)

    @abstractmethod
    def generate(self) -> pl.DataFrame:
        raise NotImplementedError

    def save(self, df: pl.DataFrame) -> None:
        self.validate_candidates(df)
        df.write_parquet(self.output_path)

    def load(self) -> pl.DataFrame:
        return pl.read_parquet(self.output_path)

    def validate_candidates(self, df: pl.DataFrame) -> None:
        if df.shape[0] == 0:
            raise ValueError("Empty dataframe")

        if set(df.columns) != set(self.output_cols):
            raise ValueError("Invalid columns")


class SampleCandidate(Candidate):
    def __init__(
        self,
        output_dir: Path | str,
        user_col: str = "user_id",
        item_col: str = "item_id",
        suffix: str | None = None,
        evaluate_topks: list[int] = [1, 5, 10, 50, 100],
    ):
        super().__init__(output_dir, user_col, item_col, suffix, evaluate_topks=evaluate_topks)
        self.df = pl.DataFrame(
            {
                self.user_col: [1, 1, 1, 2, 2, 2, 3, 3, 3],
                self.item_col: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                "rank": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            }
        )

    def generate(self) -> pl.DataFrame:
        df = self.df.sort(by="score", descending=True)
        return df


class CandidateMerger(Candidate):
    def __init__(
        self,
        data_loader: Any,
        candidates: list[Candidate],
        output_dir: str | Path,
        mode: Literal["train", "test"] = "train",
        user_col: str = "session_id",
        item_col: str = "yad_no",
        target_df: pl.DataFrame | None = None,
    ):
        super().__init__(output_dir, user_col, item_col, target_df=target_df, suffix=mode)
        self.mode = mode
        self.candidates = candidates
        self.data_loader = data_loader
        self.key_cols = [user_col, item_col]

    def merge(self) -> pl.DataFrame:
        merged_df = pl.DataFrame()
        for idx, c in tqdm(enumerate(self.candidates)):
            cand_df = c.load().rename({"score": f"{c.class_name}_score", "rank": f"{c.class_name}_rank"})
            merged_df = cand_df if idx == 0 else merged_df.join(cand_df, how="outer", on=self.key_cols)

        merged_df = merged_df.with_columns(
            [
                pl.sum_horizontal(pl.col("^.*score$").is_not_null()).alias("cand_selected_cnt"),
                pl.sum_horizontal(pl.col("^.*score$").fill_null(0)).alias("sum_score"),
            ]
        ).with_columns((pl.struct("cand_selected_cnt", "sum_score")).rank(descending=True).over(self.user_col).alias("rank"))
        return merged_df

    def generate(self) -> pl.DataFrame:  # just Alias
        return self.merge()

    def validate_candidates(self, df: pl.DataFrame) -> None:
        if df.shape[0] == 0:
            raise ValueError("Empty dataframe")

        assert df["cand_selected_cnt"].is_not_null().all()
        assert df["sum_score"].is_not_null().all()
