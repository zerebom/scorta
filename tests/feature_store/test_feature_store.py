import os

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import pytest
from scorta.feature_store import BaseFeature
from scorta.testing import get_temp_directory


@pytest.fixture
def save_dir():
    pass

def test_make_features(save_dir):

    class SumFeature(BaseFeature):

        def import_columns(self):
            return ['col_A',"col_B"]

        def make_features(self,train,test):
            self.train["col_sum"] = train["col_A"] + train["col_B"]
            self.test["col_sum"] = test["col_A"] + test["col_B"]

            return self.train, self.test

    with get_temp_directory() as tmp:
        SumFeature.gen(save_dir=save_dir, replace=True)
        assert os.path.exists(os.path.join(tmp, 'sum_feature.feather'))


