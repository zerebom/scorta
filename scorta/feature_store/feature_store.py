import re
import os
import warnings
from typing import List, Optional, Union,Tuple
from logging import Logger, StreamHandler, INFO, Formatter
import time
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from tqdm import tqdm


class BaseFeature(metaclass=ABCMeta):
    input_dir = './input'
    output_dir = './feature'

    def __init__(self, debug: bool = False, replace=False, input_dir=None):

        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.replace = replace
        self.debug = debug

        if input_dir is not None:
            self.input_dir = Path(input_dir)

        self.train_path = Path(self.input_dir) / f'{self.name}_train.ftr'
        self.test_path = Path(self.input_dir) / f'{self.name}_test.ftr'

    @staticmethod
    def validate_train_test_difference(train: pd.Series, test: pd.Series):
        # % of nulls
        if test.isnull().mean() == 1.0:
            raise RuntimeError('Error in feature {}: all values in test data is null'.format(train.name))

    def validate_feature(self,df: pd.DataFrame, y: pd.Series):
        if len(y) < len(df):
            # assuming that the first part of the dataframe is train part
            train = df.iloc[:len(y), :]
            test = df.iloc[len(y):, :]
        else:
            train = df[~y.isnull()]
            test = df[y.isnull()]

        for c in df.columns:
            self.validate_train_test_difference(train[c], test[c])
            pass

    def save(self, df: pd.DataFrame, feature_name: Union[int, str], directory: str = './features/',
             with_csv_dump: bool = False, create_directory: bool = True,
             reference_target_variable: Optional[pd.Series] = None):

        if create_directory:
            os.makedirs(directory, exist_ok=True)

        if reference_target_variable is not None:
            self.validate_feature(df, reference_target_variable)

        path = os.path.join(directory, str(feature_name) + '.ftr')

        if not self.replace and os.path.exists(path):
            raise RuntimeError('File already exists')

        df.to_feather(path)

        if with_csv_dump:
            df.head(1000).to_csv(os.path.join(
                directory, str(feature_name) + '.csv'), index=False)

    def load(self, feature_name: Union[int, str], directory: str = './features/',
                    ignore_columns: List[str] = None) -> pd.DataFrame:
        """
        Load feature as pandas DataFrame.
        Args:
            feature_name:
                The name of the feature (used in ``save_feature``).
            directory:
                The directory where the feature is stored.
            ignore_columns:
                The list of columns that will be dropped from the loaded dataframe.
        Returns:
            The feature dataframe
        """
        path = os.path.join(directory, str(feature_name) + '.ftr')

        df = pd.read_feather(path)
        if ignore_columns:
            return df.drop([c for c in ignore_columns if c in df.columns], axis=1)
        else:
            return df

    def loads(self, base_df: Optional[pd.DataFrame],
                    feature_names: List[Union[int, str]], directory: str = './features/',
                    ignore_columns: List[str] = None, create_directory: bool = True,
                    rename_duplicate: bool = True) -> pd.DataFrame:
        """
        Load features and returns concatenated dataframe
        Args:
            base_df:
                The base dataframe. If not None, resulting dataframe will consist of base and loaded feature columns.
            feature_names:
                The list of feature names to be loaded.
            directory:
                The directory where the feature is stored.
            ignore_columns:
                The list of columns that will be dropped from the loaded dataframe.
            create_directory:
                If True, create directory if not exists.
            rename_duplicate:
                If True, duplicated column name will be renamed automatically (feature name will be used as suffix).
                If False, duplicated columns will be as-is.
        Returns:
            The merged dataframe
        """
        if create_directory:
            os.makedirs(directory, exist_ok=True)

        dfs = [self.load(f, directory=directory, ignore_columns=ignore_columns) for f in tqdm(feature_names)]

        if base_df is None:
            base_df = dfs[0]
            dfs = dfs[1:]
            feature_names = feature_names[1:]

        columns = list(base_df.columns)

        for df, feature_name in zip(dfs, feature_names):
            if len(df) != len(base_df):
                raise RuntimeError('DataFrame length are different. feature={}'.format(feature_name))

            for c in df.columns:
                if c in columns:
                    warnings.warn('A feature name {} is duplicated.'.format(c))

                    if rename_duplicate:
                        while c in columns:
                            c += '_' + str(feature_name)
                        warnings.warn('The duplicated name in feature={} will be renamed to {}'.format(feature_name, c))
                columns.append(c)

        concatenated = pd.concat([base_df] + dfs, axis=1)
        concatenated.columns = columns
        return concatenated


