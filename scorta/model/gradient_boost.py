from typing import Any, Callable, Literal, TypeAlias

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoost, Pool
from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm.callback import early_stopping
from sklearn.model_selection import StratifiedKFold
from xgboost.core import Booster

Params = dict[str, Any]

GBDTBooster: TypeAlias = CatBoost | LGBMClassifier | Booster
GBDType: TypeAlias = Literal["cat", "xgb", "lgb"]
TaskType: TypeAlias = Literal["bin", "reg", "multi", "rank"]


class GBTWrapper:
    def __init__(self, gbt_type: GBDType = "cat", task_type: TaskType = "bin"):
        self.gbt_type = gbt_type
        self.task_type = task_type

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: dict | None = None,
        verbose: int = 50,
        callback: Callable | None = None,
        cv: Callable | None = None,
    ) -> tuple[GBDTBooster, np.ndarray]:
        if self.gbt_type == "cat":
            self.models, self.oof = fit_catboost(X, y, params, verbose, callback, self.task_type, cv)

        elif self.gbt_type == "lgb":
            self.models, self.oof = fit_lgbm(X, y, params, verbose, callback, self.task_type, cv)

        elif self.gbt_type == "xgb":
            self.models, self.oof = fit_xgb(X, y, params, verbose, callback, self.task_type, cv)

        return self.models, self.oof

    def get_models(self) -> list[GBDTBooster] | None:
        return self.models

    def feature_importance(self) -> list[np.ndarray | list[float | int]]:
        if self.gbt_type == "cat":
            return [model.get_feature_importance() for model in self.models]  # type: ignore

        if self.gbt_type == "lgb":
            return [model.feature_importances_ for model in self.models]  # type: ignore

        if self.gbt_type == "xgb":
            # Zero-importance features will not be included対策
            # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.get_score
            def fill_dic(dic: dict, feats: int) -> dict:
                feats = model.num_features()
                for i in range(feats):
                    key = f"f{i}"
                    if key not in dic:
                        dic[key] = 0
                return dic

            importances = []
            for model in self.models:
                feats = model.num_features()
                feat_dic = model.get_score(importance_type="gain")
                feat_dic = fill_dic(feat_dic, feats)
                importances.append([feat_dic[key] for key in sorted(feat_dic)])

            return importances  # type: ignore

    def predict_proba(self, test_X: pd.DataFrame, model_idx: int) -> list[float]:
        if self.gbt_type == "cat":
            test_pool = Pool(test_X)
            return self.models[model_idx].predict(test_pool, prediction_type="Probability")  # type: ignore

        if self.gbt_type == "lgb":
            return self.models[model_idx].predict_proba(test_X)  # type: ignore

        if self.gbt_type == "xgb":
            dtest = xgb.DMatrix(test_X)
            return self.models[model_idx].predict(dtest)  # type: ignore

    def predict(self, test_X: pd.DataFrame, model_idx: int) -> Any | np.ndarray[Any, Any]:
        if self.gbt_type == "cat":
            test_pool = Pool(test_X)
            return self.models[model_idx].predict(test_pool)

        if self.gbt_type == "lgb":
            return self.models[model_idx].predict(test_X)

        if self.gbt_type == "xgb":
            dtest = xgb.DMatrix(test_X)
            return self.models[model_idx].predict(dtest)


def fit_xgb(
    X: np.ndarray,
    y: np.ndarray,
    params: dict | None = None,
    verbose: int = 50,
    callback: Callable | None = None,
    task_type: TaskType = "bin",
    cv: Callable | None = None,
) -> tuple[list[GBDTBooster], np.ndarray]:
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if params is None:
        # https://xgboost.readthedocs.io/en/stable/parameter.html
        params = {}
        clf_params = {"early_stopping_rounds": 1, "num_boost_round": 1, "objective": "binary:logistic"}

    n_classes = len(np.unique(y))
    oof = np.zeros(shape=(len(y), n_classes), dtype=np.float32)
    models = []

    copy_X = np.copy(X)
    for fold_idx, (idx_tr, idx_val) in enumerate(cv.split(X, y)):  # type: ignore
        X = np.copy(copy_X)

        if callback:
            tr_feat, te_feat = callback(fold_idx=fold_idx)
            X = np.hstack([X, tr_feat])

        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_val, y_val = X[idx_val], y[idx_val]
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        evals = [(dtrain, "train"), (dval, "eval")]

        clf = xgb.train(
            params,
            dtrain,
            evals=evals,
            early_stopping_rounds=clf_params["early_stopping_rounds"],
            num_boost_round=clf_params["num_boost_round"],
            verbose_eval=verbose,
        )

        pred_i = clf.predict(dval)
        oof[idx_val] = np.transpose(np.array([1 - pred_i, pred_i]))
        models.append(clf)

    return models, oof


def fit_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    params: dict | None = None,
    verbose: int = 50,
    callback: Callable | None = None,
    task_type: TaskType = "bin",
    cv: Callable | None = None,
) -> tuple[list[GBDTBooster], np.ndarray]:
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if params is None:
        params = {"early_stopping_rounds": 1}

    n_classes = len(np.unique(y))

    if task_type == "bin":
        oof = np.zeros(shape=(len(y), n_classes), dtype=np.float32)
    else:
        oof = np.zeros(shape=(len(y)), dtype=np.float32)
    models = []

    copy_X = np.copy(X)
    for fold_idx, (idx_tr, idx_val) in enumerate(cv.split(X, y)):  # type: ignore
        X = np.copy(copy_X)

        if callback:
            tr_feat, te_feat = callback(fold_idx=fold_idx)
            X = np.hstack([X, tr_feat])

        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_val, y_val = X[idx_val], y[idx_val]

        if task_type == "bin":
            clf = LGBMClassifier(**params)
            clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(params["early_stopping_rounds"], verbose=verbose)],  # type: ignore
            )

            pred_i = clf.predict_proba(X_val)
            oof[idx_val] = pred_i

        elif task_type == "reg":
            clf = LGBMRegressor(**params)
            clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(params["early_stopping_rounds"], verbose=verbose)],  # type: ignore
            )

            pred_i = clf.predict(X_val)
            oof[idx_val] = pred_i

        models.append(clf)

    return models, oof  # type: ignore


def fit_catboost(
    X: pd.DataFrame,
    y: pd.DataFrame,
    params: dict[str, Any] | None,
    verbose: int = 50,
    callback: Callable | None = None,
    task_type: TaskType = "bin",
    cv: Callable | None = None,
) -> tuple[list[GBDTBooster], np.ndarray]:
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # parameter が指定されないときには空の dict で置き換えします
    # https://catboost.ai/en/docs/references/training-parameters/
    if params is None:
        params = {"n_estimators": 1}

    n_classes = len(np.unique(y))
    oof = np.zeros(shape=(len(y), n_classes), dtype=np.float32)
    models = []

    copy_X = np.copy(X)
    for fold_idx, (idx_tr, idx_val) in enumerate(cv.split(X, y)):  # type: ignore
        X = np.copy(copy_X)

        if callback:
            tr_feat, te_feat = callback(fold_idx=fold_idx)
            X = np.hstack([X, tr_feat])

        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_val, y_val = X[idx_val], y[idx_val]

        clf = CatBoost(params=params)
        clf_train = Pool(X_tr, y_tr)
        clf_val = Pool(X_val, y_val)
        clf.fit(clf_train, eval_set=[clf_val])

        pred_i = clf.predict(X_val, prediction_type="Probability")

        oof[idx_val] = pred_i
        models.append(clf)

    return models, oof


## ↑ OLD CODES
class GBDT:
    def __init__(
        self,
        gbdt_type: GBDType = "cat",
        task_type: TaskType = "bin",
        model_params: dict[str, Any] | None = None,
        train_params: dict[str, Any] | None = None,
        cv: Callable | None = None,
        callbacks: list[Callable] | None = None,
        **kwargs: Any,
    ):
        self.gbt_type = gbdt_type
        self.task_type = task_type
        self.cv = cv

        if self.cv is None:
            self.cv = KFold(n_splits=5, shuffle=True, random_state=42)

        self.model_params = model_params
        self.train_params = train_params
        self.callbacks = callbacks
        self.kwargs = kwargs


class LGB(GBDT):
    """
    ex. callbacks:
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=True),
            lgb.log_evaluation(10),
        ],
    ex. train_params: {"num_boost_round": 10000}
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_classes = len(np.unique(y))
        self.models = []
        match self.task_type:
            case "bin" | "multi":
                self.model = LGBMClassifier(**self.model_params)
                oof = np.zeros(shape=(len(y), n_classes), dtype=np.float32)
            case "reg":
                self.model = LGBMRegressor(**self.model_params)
                oof = np.zeros(shape=(len(y)), dtype=np.float32)
            case "rank":
                self.model = LGBMRanker(**self.model_params)
                oof = np.zeros(shape=(len(y)), dtype=np.float32)

        for idx_tr, idx_val in cv.split(X, y):  # type: ignore
            X_tr, y_tr = X[idx_tr], y[idx_tr]
            X_val, y_val = X[idx_val], y[idx_val]

            self.model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=self.callbacks,  # type: ignore
            )

            oof[idx_val] = self.model.predict_proba(X_val) if self.task_type in ("bin", "multi") else self.model.predict(X_val)
            self.models.append(self.model)
        return self.models, oof

    def predict(self, X: np.ndarray):
        return self.model.predict(X)

    def feature_importance(self):
        return [model.feature_importances_ for model in self.models]


class CAT(GBDT):
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_classes = len(np.unique(y))
        self.models = []

        match self.task_type:
            case "bin" | "multi":
                self.model = CatBoost(params=self.model_params)
                oof = np.zeros(shape=(len(y), n_classes), dtype=np.float32)

        for idx_tr, idx_val in cv.split(X, y):  # type: ignore
            X_tr, y_tr = X[idx_tr], y[idx_tr]
            X_val, y_val = X[idx_val], y[idx_val]

            clf_train = Pool(X_tr, y_tr)
            clf_val = Pool(X_val, y_val)
            self.model.fit(clf_train, eval_set=[clf_val])

            oof[idx_val] = (
                self.model.predict(X_val, prediction_type="Probability") if self.task_type in ("bin", "multi") else self.model.predict(X_val)
            )
            self.models.append(self.model)
        return self.models, oof


"""
lgb_wrapper = LGB(
    gbdt_type="lgb",
    task_type="bin",
    model_params=cfg.lgb.model_params,
    train_params=cfg.lgb.train_params,
    cv=cv,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(50),
    ],
)
models, oof = lgb_wrapper.fit(X_train, y_train)
"""
