from scorta.model.gradient_boost import GBTWrapper
import pytest
import numpy as np

from typing import TypeAlias, Literal

GBDType: TypeAlias = Literal["cat", "xgb", "lgb"]


def create_dummy_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.random.rand(100, 10)  # 100行10列のダミー特徴量データ
    y = np.random.randint(0, 2, 100)  # 2クラスのダミー目標変数
    return X, y


@pytest.mark.parametrize("gbt_type", ["lgb", "xgb", "cat"])
def test_fit(gbt_type: GBDType) -> None:
    X, y = create_dummy_data()
    wrapper = GBTWrapper(gbt_type=gbt_type)
    _, oof = wrapper.fit(X, y)
    assert oof.shape == (len(y), len(np.unique(y)))


@pytest.mark.parametrize("gbt_type", ["lgb", "xgb", "cat"])
def test_feature_importance(gbt_type: GBDType) -> None:
    X, y = create_dummy_data()
    wrapper = GBTWrapper(gbt_type=gbt_type)
    _, oof = wrapper.fit(X, y)
    feature_importances = wrapper.feature_importance()

    assert len(np.mean(feature_importances, axis=0)) == X.shape[1]  # type: ignore
