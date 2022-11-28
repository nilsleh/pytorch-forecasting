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

    def __init__(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        time_idx: str,
        target: Union[str, List[str]],
        group_ids: List[str],
        max_encoder_length: int = 30,
        min_encoder_length: int = None,
        min_prediction_length: int = None,
        max_prediction_length: int = 1,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_known_categoricals: List[str] = [],
        time_varying_known_reals: List[str] = [],
        time_varying_unknown_categoricals: List[str] = [],
        time_varying_unknown_reals: List[str] = [],
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
            max_encoder_length (int): maximum length to encode.
                This is the maximum history length used by the time series dataset.
            min_encoder_length (int): minimum allowed length to encode. Defaults to max_encoder_length.
            max_prediction_length (int): maximum prediction/decoder length (choose this not too short as it can help
                convergence)
            min_prediction_length (int): minimum prediction/decoder length. Defaults to max_prediction_length
            static_categoricals (List[str]): list of categorical variables that do not change over time,
                entries can be also lists which are then encoded together
                (e.g. useful for product categories)
            static_reals (List[str]): list of continuous variables that do not change over time
            time_varying_known_categoricals (List[str]): list of categorical variables that change over
                time and are known in the future, entries can be also lists which are then encoded together
                (e.g. useful for special days or promotion categories)
            time_varying_known_reals (List[str]): list of continuous variables that change over
                time and are known in the future (e.g. price of a product, but not demand of a product)
            time_varying_unknown_categoricals (List[str]): list of categorical variables that change over
                time and are not known in the future, entries can be also lists which are then encoded together
                (e.g. useful for weather categories). You might want to include your target here.
            time_varying_unknown_reals (List[str]): list of continuous variables that change over
                time and are not known in the future.  You might want to include your target here.
        """
        self.target = target
        self.time_idx = time_idx
        self.group_ids = [] + group_ids

        assert isinstance(
            data, (pd.DataFrame, pl.DataFrame)
        ), "Currently only support pandas or polars dataframe for argument 'data'"

        self.max_encoder_length = max_encoder_length
        assert isinstance(self.max_encoder_length, int), "max encoder length must be integer"
        if min_encoder_length is None:
            min_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        assert (
            self.min_encoder_length <= self.max_encoder_length
        ), "max encoder length has to be larger equals min encoder length"
        assert isinstance(self.min_encoder_length, int), "min encoder length must be integer"
        self.max_prediction_length = max_prediction_length
        assert isinstance(self.max_prediction_length, int), "max prediction length must be integer"
        if min_prediction_length is None:
            min_prediction_length = max_prediction_length
        self.min_prediction_length = min_prediction_length
        assert (
            self.min_prediction_length <= self.max_prediction_length
        ), "max prediction length has to be larger equals min prediction length"
        assert self.min_prediction_length > 0, "min prediction length must be larger than 0"
        assert isinstance(self.min_prediction_length, int), "min prediction length must be integer"

        self.static_categoricals = static_categoricals
        self.static_reals = static_reals
        self.time_varying_known_categoricals = time_varying_known_categoricals
        self.time_varying_known_reals = time_varying_known_categoricals
        self.time_varying_unknown_categoricals = time_varying_unknown_categoricals
        self.time_varying_unknown_reals = time_varying_unknown_reals

    @abc.abstractmethod  # this method forces subclasses to implement it
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

    @property
    def categoricals(self) -> List[str]:
        """
        Categorical variables as used for modelling.

        Returns:
            List[str]: list of variables
        """
        return self.static_categoricals + self.time_varying_known_categoricals + self.time_varying_unknown_categoricals

    @property
    def reals(self) -> List[str]:
        """
        Continous variables as used for modelling.

        Returns:
            List[str]: list of variables
        """
        return self.static_reals + self.time_varying_known_reals + self.time_varying_unknown_reals
