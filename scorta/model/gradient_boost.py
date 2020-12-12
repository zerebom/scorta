from hydra.utils import call
from sklearn.model_selection import StratifiedKFold
import numpy as np
from lightgbm import LGBMClassifier
from catboost import CatBoost
from catboost import Pool
import xgboost as xgb


class GBTWrapper:
    def __init__(self, gbt_type='cat'):
        self.gbt_type = gbt_type

    def fit(self, X, y, params=None, verbose=50, callback=None):
        if self.gbt_type == 'cat':
            self.models, self.oof = fit_catboost(
                X, y, params, verbose, callback)

        elif self.gbt_type == 'lgb':
            self.models, self.oof = fit_lgbm(X, y, params, verbose, callback)

        elif self.gbt_type == 'xgb':
            self.models, self.oof = fit_xgb(X, y, params, verbose, callback)

        return self.models, self.oof

    def get_models(self):
        return self.models

    def feature_importance(self):
        if self.gbt_type == 'cat':
            return [model.get_feature_importance() for model in self.models]

        if self.gbt_type == 'lgb':
            return [model.feature_importances_ for model in self.models]

        if self.gbt_type == 'xgb':
            return [model.get_score(importance_type='gain').values() for model in self.models]


    def predict_proba(self, test_X, model_idx):
        if self.gbt_type == 'cat':
            test_pool = Pool(test_X)
            return self.models[model_idx].predict(test_pool, prediction_type='Probability')

        if self.gbt_type == 'lgb':
            return self.models[model_idx].predict_proba(test_X)

        if self.gbt_type == 'xgb':
            dtest = xgb.DMatrix(test_X)
            return self.models[model_idx].predict(dtest)


def fit_xgb(X, y, params=None, verbose=50, callback=None):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if params is None:
        params = {}

    n_classes = len(np.unique(y))
    oof = np.zeros(shape=(len(y), n_classes), dtype=np.float)
    models = []

    copy_X = np.copy(X)
    for fold_idx, (idx_tr, idx_val) in enumerate(cv.split(X, y)):
        X = np.copy(copy_X)

        if callback:
            tr_feat, te_feat = callback(fold_idx=fold_idx)
            X = np.hstack([X, tr_feat])

        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_val, y_val = X[idx_val], y[idx_val]
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        evals = [(dtrain, 'train'), (dval, 'eval')]

        clf = xgb.train(params, dtrain, evals=evals,
                        early_stopping_rounds=params['early_stopping_rounds'], num_boost_round=params['num_boost_round'], verbose_eval=verbose)

        pred_i = clf.predict(dval)
        oof[idx_val] = pred_i
        models.append(clf)

    return models, oof


def fit_lgbm(X, y, params=None, verbose=50, callback=None):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # parameter が指定されないときには空の dict で置き換えします
    if params is None:
        params = {}

    n_classes = len(np.unique(y))
    oof = np.zeros(shape=(len(y), n_classes), dtype=np.float)
    models = []

    copy_X = np.copy(X)
    for fold_idx, (idx_tr, idx_val) in enumerate(cv.split(X, y)):
        X = np.copy(copy_X)

        if callback:
            tr_feat, te_feat = callback(fold_idx=fold_idx)
            X = np.hstack([X, tr_feat])

        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_val, y_val = X[idx_val], y[idx_val]

        clf = LGBMClassifier(**params)
        clf.fit(X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=100,
                verbose=verbose)

        pred_i = clf.predict_proba(X_val)
        oof[idx_val] = pred_i
        models.append(clf)

    return models, oof


def fit_catboost(X, y, params=None, verbose=50, callback=None):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # parameter が指定されないときには空の dict で置き換えします
    if params is None:
        params = {}

    n_classes = len(np.unique(y))
    oof = np.zeros(shape=(len(y), n_classes), dtype=np.float)
    models = []

    copy_X = np.copy(X)
    for fold_idx, (idx_tr, idx_val) in enumerate(cv.split(X, y)):
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

        pred_i = clf.predict(X_val, prediction_type='Probability')

        oof[idx_val] = pred_i
        models.append(clf)

    return models, oof