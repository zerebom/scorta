"""
Created on Wed May  9 15:28:58 2018
@author: kazuki.onodera
"""

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss


@dataclass
class Metric:
    name: str
    method: Callable
    is_binary: bool = False


class TopNAccuracyScore:
    def __init__(self, n: int = 3):
        self.n = n

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        top_n_labels = np.argsort(y_pred, axis=1)[:, -self.n :]
        same_labels = np.sum(top_n_labels == y_true.reshape(-1, 1), axis=1)
        return same_labels.sum() / len(same_labels)  # type: ignore


def top_n_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, n: int = 1) -> float:
    return TopNAccuracyScore(n=n)(y_true, y_pred)


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, pred_label: str | None = None) -> dict:
    """
    他クラス分類問題に関する metric を一気に計算する

    Args:
        y_true:
            ground truth. shape = (n_samples,)
        y_pred:
            predict probability. shape = (n_samples, n_labels)
        pred_label:
            predict label (optional).
            指定がない場合には sample ごとの argmax を label とする

    Returns:
        score: dict
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be same length")

    metrics = [
        Metric(name="logloss", is_binary=False, method=log_loss),
        Metric(name="accuracy", is_binary=True, method=accuracy_score),
        Metric(name="micro_f1_score", is_binary=True, method=lambda *x: f1_score(*x, average="micro")),
        Metric(name="macro_f1_score", is_binary=True, method=lambda *x: f1_score(*x, average="macro")),
        *[Metric(name="top_n_accuracy@{}".format(n), is_binary=False, method=TopNAccuracyScore(n=n)) for n in [2, 3, 4, 5]],
    ]

    score = {}

    for m in metrics:
        pred_i = pred_label if m.is_binary else y_pred
        score[m.name] = m.method(y_true, pred_i)

    return score


def visualize_confusion_matrix(
    y_true: np.ndarray, pred_label: np.ndarray, height: float = 0.6, labels: str | None = None
) -> tuple[plt.Figure, plt.Axes]:
    conf = confusion_matrix(y_true=y_true, y_pred=pred_label, normalize="true")

    n_labels = len(conf)
    size = n_labels * height
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(conf, cmap="Blues", ax=ax, annot=True, fmt=".2f")
    ax.set_ylabel("Label")
    ax.set_xlabel("Predict")

    if labels is not None:
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels)
        ax.tick_params("y", labelrotation=0)
        ax.tick_params("x", labelrotation=90)
    return fig, ax


def visualize_importance(feature_importances: list[int], feat_train_df: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    feature_importance_df = pd.DataFrame()
    for i, fe in enumerate(feature_importances):
        _df = pd.DataFrame()
        _df["feature_importance"] = fe
        _df["column"] = feat_train_df.columns
        _df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

    order = feature_importance_df.groupby("column").sum()[["feature_importance"]].sort_values("feature_importance", ascending=False).index[:50]

    fig, ax = plt.subplots(figsize=(12, max(4, len(order) * 0.2)))
    sns.boxenplot(data=feature_importance_df, y="column", x="feature_importance", order=order, ax=ax, palette="viridis")
    fig.tight_layout()
    ax.grid()
    return fig, ax


def df_info(target_df: pd.DataFrame, topN: int = 10) -> pd.DataFrame:
    print("heeeeeeeeee")
    max_row = target_df.shape[0]
    print(f"Shape: {target_df.shape}")

    df = target_df.dtypes.to_frame()
    df.columns = ["DataType"]
    df["#Nulls"] = target_df.isnull().sum()
    df["#Uniques"] = target_df.nunique()

    # stats
    df["Min"] = target_df.min(numeric_only=True)
    df["Mean"] = target_df.mean(numeric_only=True)
    df["Max"] = target_df.max(numeric_only=True)
    df["Std"] = target_df.std(numeric_only=True)

    # top 10 values
    df[f"top{topN} val"] = "0"
    df[f"top{topN} cnt"] = "0"
    df[f"top{topN} raito"] = "0"
    for c in df.index:
        vc = target_df[c].value_counts().head(topN)
        val = list(vc.index)
        cnt = list(vc.values)
        raito = list((vc.values / max_row).round(2))
        df.loc[c, f"top{topN} val"] = str(val)
        df.loc[c, f"top{topN} cnt"] = str(cnt)
        df.loc[c, f"top{topN} raito"] = str(raito)

    return df


def top_categories(df: pd.DataFrame, category_feature: list[str], topN: int = 30) -> pd.Index:
    return df[category_feature].value_counts().head(topN).index


def count_categories(df: pd.DataFrame, category_features: list[str], topN: int = 30, sort: str = "freq", df2: pd.DataFrame | None = None) -> None:
    for c in category_features:
        target_value = df[c].value_counts().head(topN).index
        if sort == "freq":
            order = target_value
        elif sort == "alphabetic":
            order = df[c].value_counts().head(topN).sort_index().index

        if df2 is not None:
            plt.subplot(1, 2, 1)
        sns.countplot(x=c, data=df[df[c].isin(order)], order=order)
        plt.xticks(rotation=90)

        if df2 is not None:
            plt.subplot(1, 2, 2)
            sns.countplot(x=c, data=df2[df2[c].isin(order)], order=order)
            plt.xticks(rotation=90)

        if df2 is not None:
            plt.suptitle(f"{c} TOP{topN}", size=25)
        else:
            plt.title(f"{c} TOP{topN}", size=25)
        plt.tight_layout()
        plt.show()

    return


def hist_continuous(df: pd.DataFrame, continuous_features: list[str], bins: int = 30, df2: pd.DataFrame | None = None) -> None:
    for c in continuous_features:
        if df2 is not None:
            plt.subplot(1, 2, 1)
        df[c].hist(bins=bins)

        if df2 is not None:
            plt.subplot(1, 2, 2)
            df2[c].hist(bins=bins)

        if df2 is not None:
            plt.suptitle(f"{c}", size=25)
        else:
            plt.title(f"{c}", size=25)
        plt.tight_layout()
        plt.show()

    return
