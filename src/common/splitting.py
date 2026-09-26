"""
Data splitting and ground truth loading utilities for the entity resolution pipeline.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import schema


def load_train_ground_truth(
    path: Union[str, Path] = "dataset/train/train_ground_truth.tsv",
) -> pd.DataFrame:
    """
    Load train ground truth TSV into a pandas DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to train_ground_truth.tsv.

    Returns
    -------
    pd.DataFrame
        DataFrame with source1_entity_id and matched_entity_ids columns.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Ground truth file not found at {path}")
    return pd.read_csv(p, sep="\t")


def load_ground_truth(
    path: Union[str, Path] = "dataset/train/train_ground_truth.tsv",
) -> pd.DataFrame:
    """Alias for load_train_ground_truth."""
    return load_train_ground_truth(path)


def split_train_val(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train and validation sets.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to split.
    test_size : float, default 0.2
        Fraction of data to allocate to validation set.
    random_state : int, default 42
        Random seed for reproducibility.
    stratify_col : str, optional
        Column name to stratify by (e.g. label).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, val_df)
    """
    stratify = None
    if stratify_col and stratify_col in df.columns:
        # Only stratify if all classes have at least 2 instances
        val_counts = df[stratify_col].value_counts()
        if (val_counts >= 2).all() and len(val_counts) > 1:
            stratify = df[stratify_col]

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
