"""Base Class for time-series."""

import torch
from typing import Tuple, Dict, Any, Union, List
import pandas as pd
import polars as pl
from functools import lru_cache
from torch.utils.data import Dataset
import abc
import inspect
from pytorch_forecasting.utils import repr_class


class BaseTimeSeriesDataSet(Dataset, abc.ABC):
    """Base class for Time Series Datasets.""" 

    def __init__(self,
        data: Union[pd.DataFrame, pl.DataFrame],
        time_idx: str,
        target: Union[str, List[str]],
        group_ids: List[str],
        ) -> None:
        """Initialize a new instance of BaseTimeSeriesDataset.
        Args:
            data (Union[pd.DataFrame, pl.DataFrame): dataframe with sequence data - each row can be identified with
                ``time_idx`` and the ``group_ids``
            time_idx (str): integer column denoting the time index. This columns is used to determine
                the sequence of samples.
                If there no missings observations, the time index should increase by ``+1`` for each subsequent sample.
                The first time_idx for each series does not necessarily have to be ``0`` but any value is allowed.
            target (Union[str, List[str]]): column denoting the target or list of columns denoting the target -
                categorical or continous.
            group_ids (List[str]): list of column names identifying a time series. This means that the ``group_ids``
                identify a sample together with the ``time_idx``. If you have only one timeseries, set this to the
                name of column that is constant.
        """
        self.target = target
        self.time_idx = time_idx
        self.group_ids = [] + group_ids

        assert isinstance(data, (pd.DataFrame, pl.DataFrame)), "Currently only support pandas or polars dataframe for argument 'data'"

    @abc.abstractmethod # this method forces subclasses to implement it
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get sample for model.

        Args:
            idx (int): index of prediction (between ``0`` and ``len(dataset) - 1``)

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: x and y for model
        """

    @abc.abstractmethod
    def _construct_index(self, data: Union[pd.DataFrame, pl.DataFrame]) -> pd.DataFrame:
        """
        Create index of samples.

        Args:
            data (pd.DataFrame): preprocessed data
            predict_mode (bool): if to create one same per group with prediction length equals ``max_decoder_length``

        Returns:
            pd.DataFrame: index dataframe for timesteps and index dataframe for groups.
                It contains a list of all possible subsequences.
            # potentially this is really large so should also be a more memory efficient
            datastructure?
        """

    @abc.abstractmethod
    def _preprocess_data(self, data: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Scale continuous variables, encode categories and set aside target and weight.

        Args:
            data (pd.DataFrame): original data

        Returns:
            pd.DataFrame: pre-processed dataframe
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Length of dataset.

        Returns:
            int: length
        """

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters that can be used with :py:meth:`~from_parameters` to create a new dataset with the same scalers.

        Returns:
            Dict[str, Any]: dictionary of parameters
        """
        kwargs = {
            name: getattr(self, name)
            for name in inspect.signature(self.__class__.__init__).parameters.keys()
            if name not in ["data", "self"]
        }
        kwargs["categorical_encoders"] = self.categorical_encoders
        kwargs["scalers"] = self.scalers
        return kwargs

    def __repr__(self) -> str:
        return repr_class(self, attributes=self.get_parameters(), extra_attributes=dict(length=len(self)))

    @property
    @lru_cache(None)
    def target_names(self) -> List[str]:
        """
        List of targets.

        Returns:
            List[str]: list of targets
        """
        if self.multi_target:
            return self.target
        else:
            return [self.target]

    @property
    def multi_target(self) -> bool:
        """
        If dataset encodes one or multiple targets.

        Returns:
            bool: true if multiple targets
        """
        return isinstance(self.target, (list, tuple))