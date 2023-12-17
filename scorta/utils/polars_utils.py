import polars as pl


def over_rank(src_col: str, over_col: str, method: str = "ordinal", descending: bool = True) -> pl.Expr:
    return pl.col(src_col).rank(method=method, descending=descending).over(over_col)


def min_max_scaler(col: str, min_val: int = 0, max_val: int = 1) -> pl.Expr:
    x = pl.col(col)
    x_std = (x - x.min()) / (x.max() - x.min())
    x_scaled = x_std * (max_val - min_val) + min_val
    return x_scaled
