import os
import pytest
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from scorta.feature_store import BaseFeature
from scorta.testing import get_temp_directory
import re


@pytest.fixture
def save_dir():
    pass


class ImplFeat(BaseFeature):
    input_dir = Path('./input')
    output_dir = Path('./feature')

    @staticmethod
    def camel2snake(camel):
        return re.sub("([A-Z])", lambda x: "_" + x.group(1).lower(), camel)

    def __init__(self, debug=True, replace=True, input_dir=None, output_dir=None):
        super(ImplFeat, self).__init__(debug, replace, input_dir)

        self.debug = debug
        self.name = self.camel2snake(self.__class__.__name__)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()

    def gen(self):
        self.read_and_save()

    def load_data(self):
        train_path = self.input_dir / 'train.csv'
        test_path = self.input_dir / 'test.csv'
        card_path = self.input_dir / 'num_card_trades.csv'
        nrows = 1000 if self.debug else None

        train = pd.read_csv(train_path, nrows=nrows)
        test = pd.read_csv(test_path, nrows=nrows)
        card = pd.read_csv(card_path, nrows=nrows)

        return train, test, card

    def save_feature(self, train, test):
        train_feature_name = self.name + "_train"
        test_feature_name = self.name + "_test"
        self.save(train, train_feature_name, directory=self.output_dir)
        self.save(test, test_feature_name, directory=self.output_dir)

    def read_and_save(self):
        train, test, card = self.load_data()
        train_feature, test_feature = self.make_features(train, test, card)
        self.save_feature(train_feature, test_feature)

    @classmethod
    def main(cls, debug, replace, input_dir=None,output_dir=None):
        instance = cls(debug, replace, input_dir,output_dir)
        instance.gen()


def test_load_features():
    df = pd.DataFrame()

    df['a'] = np.arange(100).astype(float)
    df['b'] = np.arange(100).astype(int)
    df['c'] = np.arange(100).astype(int)

    base_feature = BaseFeature(True)

    with get_temp_directory() as tmp:
        # feature_nameとdfの列は1対1対応していなくて良い
        base_feature.save(df[['b']], 0, tmp)
        base_feature.save(df[['c']], 1, tmp)

        df_loaded = base_feature.loads(df[['a']], [0, 1], tmp)
        assert_frame_equal(df, df_loaded)


def test_make_features(save_dir):

    class SumFeature(ImplFeat):
        def make_features(self, train, test, card):
            self.train["col_sum"] = train["a"] + train["a"]
            self.test["col_sum"] = test["b"] + test["a"]
            _ = card
            return self.train, self.test

    df = pd.DataFrame()
    df['a'] = np.arange(100).astype(float)
    df['b'] = np.arange(100).astype(int)
    df['c'] = np.arange(100).astype(int)

    with get_temp_directory() as tmp:
        os.makedirs(os.path.join(tmp, 'input/'))
        input_dir = os.path.join(tmp, 'input/')
        output_dir = os.path.join(tmp, 'feature/')

        df.to_csv(os.path.join(tmp, 'input/train.csv'))
        df.to_csv(os.path.join(tmp, 'input/test.csv'))
        df.to_csv(os.path.join(tmp, 'input/num_card_trades.csv'))

        SumFeature.main(debug=True, replace=True,
                        input_dir=input_dir, output_dir=output_dir)
        import glob
        print(glob.glob(f'{tmp}/*/*'))

        assert os.path.exists(os.path.join(tmp, 'feature/_sum_feature_train.ftr'))
