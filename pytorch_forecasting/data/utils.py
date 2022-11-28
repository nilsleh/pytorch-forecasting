"""Data utility functions."""

import numpy as np
import torch
from typing import List, Union, Tuple


def _find_end_indices(diffs: np.ndarray, max_lengths: np.ndarray, min_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify end indices in series even if some values are missing.

    Args:
        diffs (np.ndarray): array of differences to next time step. nans should be filled up with ones
        max_lengths (np.ndarray): maximum length of sequence by position.
        min_length (int): minimum length of sequence.

    Returns:
        Tuple[np.ndarray, np.ndarray]: tuple of arrays where first is end indices and second is list of start
            and end indices that are currently missing.
    """
    missing_start_ends = []
    end_indices = []
    length = 1
    start_idx = 0
    max_idx = len(diffs) - 1
    max_length = max_lengths[start_idx]

    for idx, diff in enumerate(diffs):
        if length >= max_length:
            # import pdb
            # pdb.set_trace()
            while length >= max_length:
                if length == max_length:
                    end_indices.append(idx)
                else:
                    end_indices.append(idx - 1)
                length -= diffs[start_idx]
                if start_idx < max_idx:
                    start_idx += 1
                max_length = max_lengths[start_idx]
        elif length >= min_length:
            missing_start_ends.append([start_idx, idx])
        length += diff
    if len(missing_start_ends) > 0:  # required for numba compliance
        return np.asarray(end_indices), np.asarray(missing_start_ends)
    else:
        return np.asarray(end_indices), np.empty((0, 2), dtype=np.int64)


def check_for_nonfinite(tensor: torch.Tensor, names: Union[str, List[str]]) -> torch.Tensor:
    """
    Check if 2D tensor contains NAs or inifinite values.

    Args:
        names (Union[str, List[str]]): name(s) of column(s) (used for error messages)
        tensor (torch.Tensor): tensor to check

    Returns:
        torch.Tensor: returns tensor if checks yield no issues
    """
    if isinstance(names, str):
        names = [names]
        assert tensor.ndim == 1
        nans = (~torch.isfinite(tensor).unsqueeze(-1)).sum(0)
    else:
        assert tensor.ndim == 2
        nans = (~torch.isfinite(tensor)).sum(0)
    for name, na in zip(names, nans):
        if na > 0:
            raise ValueError(
                f"{na} ({na/tensor.size(0):.2%}) of {name} "
                "values were found to be NA or infinite (even after encoding). NA values are not allowed "
                "`allow_missing_timesteps` refers to missing rows, not to missing values. Possible strategies to "
                f"fix the issue are (a) dropping the variable {name}, "
                "(b) using `NaNLabelEncoder(add_nan=True)` for categorical variables, "
                "(c) filling missing values and/or (d) optionally adding a variable indicating filled values"
            )
    return tensor
