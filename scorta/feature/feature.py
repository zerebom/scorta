from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import polars as pl
from tqdm import tqdm


class Feature(ABC):
    def __init__(
        self,
        output_dir: str | Path,
        feature_cols: list[str],
        user_col: str = "user_id",
        item_col: str = "item_id",
        key_cols: list[str] = ["user_id", "item_id"],
        suffix: str | None = None,
    ):
        self.class_name = self.__class__.__name__
        if suffix is not None:
            self.class_name = f"{self.class_name}_{suffix}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / f"{self.__class__.__name__}.parquet"
        self.feature_cols = feature_cols

        self.user_col = user_col
        self.item_col = item_col
        self.key_cols = key_cols
        self.suffix = suffix

    @abstractmethod
    def fit(self) -> pl.DataFrame:
        pass

    def transform(self) -> pl.DataFrame:
        return self.fit()

    def save(self, df: pl.DataFrame) -> None:
        self.validate_feature(df)
        df.write_parquet(self.output_path)

    def load(self) -> pl.DataFrame:
        return pl.read_parquet(self.output_path)

    def validate_feature(self, df: pl.DataFrame) -> None:
        if df.shape[0] == 0:
            raise ValueError("Empty dataframe")


class SampleFeature(Feature):
    def __init__(
        self,
        output_dir: str | Path,
        feature_cols: list[str] = ["feature"],
        user_col: str = "user_id",
        item_col: str = "item_id",
        key_cols: list[str] = ["user_id", "item_id"],
        suffix: str | None = None,
    ):
        super().__init__(output_dir, feature_cols, user_col, item_col, key_cols, suffix)
        self.df = pl.DataFrame(
            {
                self.user_col: [1, 1, 1, 2, 2, 2, 3, 3, 3],
                self.item_col: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "score": [0.1, 0.2, 0.3, 0.2, 0.4, 0.6, 0.3, 0.6, 0.9],
                "rank": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            }
        )

    def fit(self) -> pl.DataFrame:
        df = self.df.with_columns(pl.lit(1).alias("feature"))
        return df[self.key_cols + self.feature_cols]


class FeatureMerger(Feature):
    def __init__(
        self,
        output_dir: str | Path,
        features: list[Feature],
        user_col: str = "user_id",
        item_col: str = "item_id",
        key_cols: list[str] = ["user_id", "item_id"],
        mode: Literal["train", "test"] | None = None,
    ):
        feautre_cols = [""]
        super().__init__(output_dir, feautre_cols, user_col, item_col, key_cols, suffix=mode)
        self.features = features

    def fit(self) -> pl.DataFrame:  # just alias
        return self.merge()

    def merge(self, df: pl.DataFrame) -> pl.DataFrame:
        for f in tqdm(self.features):
            feat_df = f.load()
            df = df.join(feat_df, on=f.key_cols, how="left")
            print(f"{f.class_name}", df.shape)
        return df
