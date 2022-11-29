"""
Large Timeseries datasets.

Timeseries data is special and has to be processed and fed to algorithms in a special way. This module
defines a class that is able to handle a wide variety of timeseries data problems. More specifically,
this class is intended to work with datasets that do not fit into memory and hence
utilizes the polars library for memory efficience.
"""

from typing import Union, List, Any, Dict, Tuple
from timeit import default_timer as timer
import pandas as pd
import torch
import numpy as np
import polars as pl
from pytorch_forecasting.data.base_timeseries import BaseTimeSeriesDataSet
from pytorch_forecasting.data.utils import _find_end_indices


class LargeTimeSeriesDataSet(BaseTimeSeriesDataSet):
    """
    Large Time Series Dataset for fitting models.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        time_idx: str,
        target: Union[str, List[str]],
        group_ids: List[str],
        max_encoder_length: int = 30,
        min_encoder_length: int = None,
        min_prediction_length: int = None,
        max_prediction_length: int = 1,
        min_prediction_idx: int = None,
        allow_missing_timesteps: bool = False,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_known_categoricals: List[str] = [],
        time_varying_known_reals: List[str] = [],
        time_varying_unknown_categoricals: List[str] = [],
        time_varying_unknown_reals: List[str] = [],
    ) -> None:
        """Timeseries dataset holding data for models.

        In contrast to :py:class:`~timeseries.TimeSeriesDataSet`, which utilizes pandas,
        this class is intended for datasets that do not fit into memory and therefore uses
        the `polars <https://www.pola.rs/>`_ library.

        Args:
            data (pl.DataFrame): dataframe with sequence data - each row can be identified with
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
            min_prediction_idx (int): minimum ``time_idx`` from where to start predictions. This parameter
                can be useful to create a validation or test set.
            allow_missing_timesteps (bool): if to allow missing timesteps that are automatically filled up. Missing
                values refer to gaps in the ``time_idx``, e.g. if a specific timeseries has only samples for
                1, 2, 4, 5, the sample for 3 will be generated on-the-fly.
                Allow missings does not deal with ``NA`` values. You should fill NA values before
                passing the dataframe to the TimeSeriesDataSet.
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
        super().__init__(
            data,
            time_idx,
            target,
            group_ids,
            max_encoder_length,
            min_encoder_length,
            min_prediction_length,
            max_prediction_length,
            static_categoricals,
            static_reals,
            time_varying_known_categoricals,
            time_varying_known_reals,
            time_varying_unknown_categoricals,
            time_varying_unknown_reals,
        )

        self.min_prediction_idx = min_prediction_idx
        self.max_time_step = data[time_idx].max()
        self.min_time_step = data[self.time_idx].min()
        self.allow_missing_timesteps = allow_missing_timesteps

        # validate
        self._validate_data(data)

        data = self._preprocess_data(data)
        print("made it throug preprocess")

        # create index
        self.index = self._construct_index(data)
        print("constructed index")

        # keep data as polars df
        self.data = data

    def _preprocess_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """Preprocess data."""
        # target expression list
        # expr_list = [pl.col(target).alias(f"__target__{target}") for target in self.target_names]

        # # group idx expression list
        # expr_list.extend([pl.col(group).alias(f"__group_id__{group}") for group in self.group_ids])
        # # time_idx expression
        # expr_list.append(pl.col(self.time_idx).alias(f"__time_idx__"))

        # data = data.with_columns(expr_list)

        print("up to encoder")
        # encode the collection of unique group_ids into one numerical sequence
        groups = pl.col("sequence_id").first().over(self.group_ids)
        data = (
            data
            .with_row_count(name="sequence_id")
            .with_column(groups.is_first().cumsum())
            .with_column(groups - 1)
            .with_column(pl.col("sequence_id").cast(pl.Int32))
        )

        return data

    def first_method(self, data):
        data = (
            data.join(
                data.select(self.group_ids).unique().with_row_count(name="sequence_id"),
                on=self.group_ids,
            )
        )
        return data

    def second_method(self, data):
        
        return data

    def _construct_index(self, data: pl.DataFrame) -> pl.DataFrame:
        """Construct index to sample sequences from."""
        df_index_query = (
            data.lazy()
            .groupby(self.group_ids)
            .agg(
                [
                    # pl.first(self.time_idx).alias("first_time_idx"),
                    # pl.last(self.time_idx).alias("last_time_idx"),
                    pl.col(self.time_idx).diff(1).shift(-1).fill_null(1).alias("time_idx_diff_to_next"),
                ]
            )
            .with_columns(
                [
                    # (pl.col("last_time_idx") - pl.col("first_time_idx")).alias("diff_first_last_time_idx") + 1,
                    (pl.col("time_idx_diff_to_next").arr.max() != 1).alias("allow_missing"),
                    pl.arange(0, pl.count()).alias("sequence_id"),
                ]
            )
            .drop("count")  # a count column is created during sequence id
        )
        df_index = df_index_query.collect()

        # find all the unique timesteps
        unique_ids = (
            data.select(self.group_ids + [self.time_idx])
            .groupby(self.group_ids)
            .agg_list()
            .select(self.time_idx)
        )

        df_index = pl.concat([df_index, unique_ids], how="horizontal")

        # min_sequence_length = self.min_prediction_length + self.min_encoder_length
        max_sequence_length = self.max_prediction_length + self.max_encoder_length

        # if there are missing timesteps, we cannot say directly what is the last timestep to include
        # therefore we iterate until it is found
        if df_index["allow_missing"].any():
            assert (
                self.allow_missing_timesteps
            ), "Time difference between steps has been idenfied as larger than 1 - set allow_missing_timesteps=True"

        df_index = df_index.drop(columns=["time_idx_diff_to_next", "allow_missing"] + self.group_ids)
        df_index = self.find_subsequences(df_index, "sequence_id", "time_idx", max_sequence_length)

        return df_index

    def find_subsequences(
        self, index_df: pl.DataFrame, group_idx: str, time_idx: str, max_sequence_length: int
    ) -> pl.DataFrame:
        """Find possible subsequences in time-series.

        Args:
            index_df: dataframe to find subsequences in
            group_idx: name of group index
            time_idx: name of time index
            max_sequence_length: maximum length of generated subsequences

        Returns:
            dataframe with computed valid subsequences start and end time idx
        """

        range_index_df = (
            pl.arange(self.min_time_step, self.max_time_step + 1, eager=True, dtype=pl.Int64)
            .alias(time_idx)
            .to_frame()
            .lazy()
        )
        slice_size = 1000
        result = pl.concat(
            [
                (
                    range_index_df.join(
                        index_df.lazy().slice(next_index, slice_size).select(group_idx),
                        how="cross",
                    )
                    .join(
                        index_df.lazy()
                        .slice(next_index, slice_size)
                        .explode(time_idx)
                        .with_column(pl.col(time_idx).alias("time_idx_nulls")),
                        on=[group_idx, time_idx],
                        how="left",
                    )
                    .groupby_rolling(
                        index_column=time_idx,
                        by=group_idx,
                        period=str(max_sequence_length) + "i",
                    )
                    .agg(pl.col("time_idx_nulls"))
                    .filter(pl.col("time_idx_nulls").arr.lengths() == max_sequence_length)
                    .with_columns(
                        [
                            pl.col("time_idx_nulls").arr.first().alias("start_idx"),
                            pl.col("time_idx_nulls").arr.last().alias("end_idx"),
                        ]
                    )
                    .select([group_idx, "start_idx", "end_idx"])
                    .collect()
                )
                for next_index in range(0, index_df.height + 1, slice_size)
            ]
        )

        result = result.drop_nulls()

        # add a sample_idx to query and add sequence_length
        result = result.with_columns(
            [
                pl.arange(0, len(result), eager=True, dtype=pl.Int64).alias("sample_idx"),
                (pl.col("end_idx") - pl.col("start_idx") + 1).alias("sequence_length"),
            ]
        )

        return result

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Retrieve sample from index."""
        index = self.index.filter(pl.col("sample_idx") == idx)
        sample_data = self.data.filter(
            (pl.col("sequence_id") == index.select("sequence_id")[0, 0])
            & (pl.col(self.time_idx) >= index.select("start_idx")[0, 0])
            & (pl.col(self.time_idx) <= index.select("end_idx")[0, 0])
        ).to_pandas()

        # cast all data to torch double float32 or int32
        sample_data[sample_data.select_dtypes(np.float64).columns] = sample_data.select_dtypes(np.float64).astype(
            np.float32
        )
        sample_data[sample_data.select_dtypes(np.int64).columns] = sample_data.select_dtypes(np.int64).astype(np.int32)
        try:
            time_first = sample_data.loc[0, self.time_idx]
        except KeyError:
            import pdb

            pdb.set_trace()
        time_last = sample_data.iloc[-1, sample_data.columns.get_loc(self.time_idx)]

        sequence_length = len(sample_data)
        full_sequence_length = index.select("sequence_length")[0, 0]
        if sequence_length < full_sequence_length:
            assert self.allow_missing_timesteps, "allow_missing_timesteps should be True if sequences have gaps"

            filled_time_idx = pd.DataFrame(
                {"filled_time_idx": np.arange(time_first, time_first + full_sequence_length)}
            )
            sample_data = pd.merge(
                left=filled_time_idx, right=sample_data, how="left", left_on="filled_time_idx", right_on=self.time_idx
            )

            # repeat previous value with ffill
            sample_data = (
                sample_data.fillna(method="ffill")
                .drop(columns=self.time_idx)
                .rename(columns={"filled_time_idx": self.time_idx})
            )

            # update time last and sequence length
            time_last = sample_data.iloc[-1, sample_data.columns.get_loc(self.time_idx)]
            sequence_length = len(sample_data)

        # determine data window
        assert (
            sequence_length >= self.min_prediction_length
        ), "Sequence length should be at least minimum prediction length"
        decoder_length = self.calculate_decoder_length(time_last, sequence_length)
        encoder_length = sequence_length - decoder_length
        assert (
            decoder_length >= self.min_prediction_length
        ), "Decoder length should be at least minimum prediction length"
        assert encoder_length >= self.min_encoder_length, "Encoder length should be at least minimum encoder length"

        target_data = sample_data[self.target_names]

        # if user defined target as list, output should be list, otherwise tensor
        if self.multi_target:
            encoder_target = [
                torch.from_numpy(target_data[target_col].iloc[:encoder_length].values).squeeze(-1)
                for target_col in target_data
            ]
            target = [
                torch.from_numpy(target_data[target_col].iloc[encoder_length:].values).squeeze(-1)
                for target_col in target_data
            ]
        else:
            encoder_target = torch.from_numpy(target_data.iloc[:encoder_length].values).squeeze(-1)
            target = torch.from_numpy(target_data.iloc[encoder_length:].values).squeeze(-1)

        data_cont = torch.from_numpy(sample_data[self.reals].values)
        data_cat = torch.from_numpy(sample_data[self.categoricals].values)

        groups = torch.tensor([sample_data.loc[0, "sequence_id"]])

        return (
            dict(
                x_cat=data_cat,
                x_cont=data_cont,
                encoder_length=encoder_length,
                decoder_length=decoder_length,
                encoder_target=encoder_target,
                encoder_time_idx_start=torch.tensor(time_first),
                groups=groups,
            ),
            (target, None),
        )

    def calculate_decoder_length(
        self,
        time_last: Union[int, pd.Series, np.ndarray],
        sequence_length: Union[int, pd.Series, np.ndarray],
    ) -> Union[int, pd.Series, np.ndarray]:
        """
        Calculate length of decoder.

        Args:
            time_last (Union[int, pd.Series, np.ndarray]): last time index of the sequence
            sequence_length (Union[int, pd.Series, np.ndarray]): total length of the sequence

        Returns:
            Union[int, pd.Series, np.ndarray]: decoder length(s)
        """
        if isinstance(time_last, int):
            decoder_length = min(
                time_last - (self.min_prediction_idx - 1),  # not going beyond min prediction idx
                self.max_prediction_length,  # maximum prediction length
                sequence_length - self.min_encoder_length,  # sequence length - min decoder length
            )
        else:
            decoder_length = np.min(
                [
                    time_last - (self.min_prediction_idx - 1),
                    sequence_length - self.min_encoder_length,
                ],
                axis=0,
            ).clip(max=self.max_prediction_length)
        return decoder_length

    def __len__(self) -> int:
        return self.index.shape[0]

    def _validate_data(self, data: pl.DataFrame) -> None:
        """
        Validate that data will not cause hick-ups later on.
        """
        # assert datatypes
        assert "polars.datatypes.Int" in str(data["time_idx"].dtype)

        # assert that the column names are proper
        warning_str = "is a protected column and must not be present in data."
        assert "__time_idx__" not in data.columns, f"__time_idx__ {warning_str}"

        for target in self.target_names:
            assert f"__target__{target}" not in data.columns, f"__target__{target} {warning_str}"

        for group in self.group_ids:
            assert f"__group_id__{group}" not in data.columns, f"__target__{target} {warning_str}"
