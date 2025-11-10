from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional,  Sequence

import pandas as pd


class Loader(ABC):
    """Abstract data loader contract.

    Subclasses should implement how to load the raw data (load_data) and how to
    extract domain-specific records (extract). Different sources (HF datasets,
    CSV files, databases, APIs) can implement their own loading strategies.
    """

    def __init__(self, source: str):
        self.source = source

    @abstractmethod
    def load_data(self, split: Optional[str] = None) -> Any:
        """Load raw data for an optional split and return a convenient object.

        Return type is intentionally Any; concrete loaders should document a more
        precise type (e.g. pandas.DataFrame, DatasetDict, list[dict]).
        """

    @abstractmethod
    def extract(self, split: Optional[str] = None) -> Any:
        """Transform loaded data into the project-specific record format.

        Typically returns a list of dataclass records (e.g., ClaimRecord).
        """

    # ------------------------------------------------------------------
    # Utility helpers available to all concrete loaders
    # ------------------------------------------------------------------
    def preview_dataframe(
        self,
        df: pd.DataFrame,
        n: int = 5,
        columns: Optional[Sequence[str]] = None,
        show_index: bool = True,
        header: Optional[str] = None,
    ) -> None:
        """Print full (untruncated) text content for specified columns row-by-row.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to preview.
        n : int, default 5
            Number of rows to display (capped at len(df)).
        columns : Sequence[str] | None
            Which columns to print. If None, all object (string-like) columns are used.
        show_index : bool, default True
            Whether to include the row index on each block.
        header : str | None
            Optional title printed once at the top.
        """
        if not isinstance(df, pd.DataFrame):  # defensive
            raise TypeError("preview_dataframe expects a pandas DataFrame")

        if header:
            print("=" * 120)
            print(header)
        # Ensure no truncation of columns
        with pd.option_context(
            "display.max_colwidth", None,
            "display.width", 200,
        ):
            if columns is None:
                # Heuristic: choose object dtype columns (likely to contain text)
                columns = [c for c in df.columns if df[c].dtype == object]
            n = min(n, len(df))
            for i in range(n):
                print("=" * 120)
                if show_index:
                    print(f"ROW {i}")
                    print("-" * 120)
                for col in columns:
                    print(f"[{col}]")
                    val = df.iloc[i][col]
                    # Represent None/NaN clearly
                    if pd.isna(val):
                        print("<NA>")
                    else:
                        print(str(val))
                    print("-" * 120)
            print("=" * 120)
