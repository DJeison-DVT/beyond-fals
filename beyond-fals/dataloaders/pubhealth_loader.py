from __future__ import annotations

from typing import List
from datasets import load_dataset, DatasetDict, Dataset
from typing import cast
import pandas as pd
from .schema import ClaimRecord
from ..utils.normalizers import stable_id, guess_domain, normalize_text


class Loader:
    """Loader for the PubHealth BigBio pairs dataset.

    Encapsulates split loading and extraction logic.
    """

    DATASET = ("bigbio/pubhealth", "pubhealth_bigbio_pairs")

    def __init__(self, cache_dir: str = "data/raw") -> None:
        self.cache_dir = cache_dir

    def _load_split(self, split: str = "train") -> Dataset:
        ds_any = load_dataset(*self.DATASET, cache_dir=self.cache_dir)
        ds_dict: DatasetDict = cast(DatasetDict, ds_any)
        return ds_dict[split]

    def _get_df(self, split: str = "train") -> pd.DataFrame:
        ds_split: Dataset = self._load_split(split)
        try:
            # HuggingFace Datasets provide to_pandas at runtime
            df = ds_split.to_pandas()  # type: ignore[attr-defined]
        except Exception:
            # Fallback: convert via dictionaries of columns
            df = pd.DataFrame(ds_split.to_dict())  # type: ignore[attr-defined]
        # Ensure type is recognized as DataFrame
        df = cast(pd.DataFrame, df)
        # Normalize column names if needed
        if "text_1" not in df.columns and "text 1" in df.columns:
            rename_map = {"text 1": "text_1"}
            if "text 2" in df.columns and "text_2" not in df.columns:
                rename_map["text 2"] = "text_2"
            df = df.rename(columns=rename_map)
        return df

    def extract(self, split: str = "train") -> List[ClaimRecord]:
        df = self._get_df(split)
        records: List[ClaimRecord] = []
        for _, row in df.iterrows():
            records.append(
                ClaimRecord(
                    claim_id=stable_id(source="pubhealth",
                                       text=str(row["text_1"])),
                    text=normalize_text(row["text_1"]),
                    source="pubhealth",
                    veracity=normalize_text(row.get("label") or ""),
                    domain=guess_domain(normalize_text(row["text_1"])),
                    falsifiability=("NonFalsifiable" if str(row["label"]).strip().lower() == "true" else
                                    "Falsifiable")
                )
            )
        return records

    def print_first_rows_texts(self, n: int = 5, split: str = "train") -> None:
        df = self._get_df(split)
        pd.set_option("display.max_colwidth", None)
        n = min(n, len(df))
        for i in range(n):
            print("=" * 100)
            print(f"ROW {i}")
            print("-" * 100)
            print("text_1:")
            print(df.loc[i, "text_1"])
            print("-" * 100)
            print("text_2:")
            print(df.loc[i, "text_2"])
            print("-" * 100)
            print(f"label: {df.loc[i, 'label']}")
        print("=" * 100)
