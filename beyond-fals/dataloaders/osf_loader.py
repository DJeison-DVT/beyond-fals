from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import tarfile
import tempfile
import shutil

from .loader import Loader
from ..utils.normalizers import stable_id, normalize_text


class OSFLoader(Loader):
    """Loader for OSF health ads dataset with metadata CSV + content archive.

    The OSF dataset has two components:
    1. osf_health_ads_metadata.csv - structured metadata (topics, URLs, deceptive flags, conditions)
    2. content.tar.gz - extracted landing page HTML/text content keyed by ad_id

    This loader:
    - Reads metadata CSV
    - Extracts content.tar.gz lazily to a cache directory
    - Joins metadata with extracted content by ad_id
    - Returns unified DataFrame with both metadata and text content
    """

    def __init__(
        self,
        metadata_path: Optional[str] = None,
        content_archive_path: Optional[str] = None,
        content_cache_dir: Optional[str] = None,
    ):
        super().__init__(source="osf")
        root = Path(__file__).resolve().parents[2]
        self.metadata_path = metadata_path or str(
            root / "data" / "raw" / "osf_health_ads_metadata.csv")
        self.content_archive_path = content_archive_path or str(
            root / "data" / "raw" / "osf_health_ads_content.tar.gz")
        # Cache extracted content here to avoid re-extraction
        self.content_cache_dir = content_cache_dir or str(
            root / "data" / "processed" / "osf_content_cache")
        self._content_map: Optional[Dict[str, str]] = None

        self._content_map: Optional[Dict[str, str]] = None

    def _extract_content(self) -> Dict[str, str]:
        """Extract content.tar.gz to cache dir and return ad_id → text mapping.

        Only extracts if cache doesn't exist or is empty. Returns a dict mapping
        ad_id (filename without extension) to file content.
        """
        if self._content_map is not None:
            return self._content_map

        cache_path = Path(self.content_cache_dir)
        archive_path = Path(self.content_archive_path)

        # Check if content archive exists
        if not archive_path.exists():
            print(
                f"Warning: Content archive not found at {archive_path}. Returning empty content map.")
            self._content_map = {}
            return self._content_map

        # Extract if cache is empty
        if not cache_path.exists() or not any(cache_path.iterdir()):
            print(f"Extracting {archive_path} to {cache_path}...")
            cache_path.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=cache_path)
            print("Extraction complete.")

        # Build ad_id → content mapping
        content_map: Dict[str, str] = {}
        for file_path in cache_path.rglob("*"):
            if file_path.is_file():
                # Use stem (filename without extension) as ad_id key
                ad_id = file_path.stem
                try:
                    content_map[ad_id] = file_path.read_text(
                        encoding="utf-8", errors="ignore")
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
                    content_map[ad_id] = ""

        self._content_map = content_map
        return content_map

    # split unused
    def load_data(self, split: Optional[str] = None) -> pd.DataFrame:
        """Load metadata CSV and optionally join with extracted content."""
        metadata_path = Path(self.metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"OSF metadata CSV not found at {metadata_path}")

        df = pd.read_csv(metadata_path)

        # Extract content and add as a new column
        content_map = self._extract_content()
        if content_map:
            # Join content by ad_id
            df["content"] = df["ad_id"].astype(str).map(content_map).fillna("")
        else:
            # No content available; leave empty
            df["content"] = ""

        return df

    def extract(self, split: Optional[str] = None):
        """Build ClaimRecord objects from metadata + extracted content."""
        from .schema import ClaimRecord

        df = self.load_data(split)
        self.preview_dataframe(df)
        records: list[ClaimRecord] = []

        for _, row in df.iterrows():  # type: ignore[attr-defined]
            # Prefer extracted content; fallback to LP URL or empty
            text_content = normalize_text(str(row.get("content", "")))
            if not text_content.strip():
                text_content = normalize_text(str(row.get("lp_url", "")))

            records.append(
                ClaimRecord(
                    claim_id=stable_id(
                        source=self.source, text=f"{row['ad_id']}_{text_content[:100]}"),
                    text=text_content,
                    source=self.source,
                    veracity=None,  # OSF doesn't label veracity directly
                    domain=str(row.get("main_topic", "")) or None,
                    falsifiability=None,  # Derive later if needed
                )
            )
        return records


if __name__ == "__main__":
    """Quick preview of first 5 ads."""
    loader = OSFLoader()
    # Preview metadata + content columns
    df = loader.load_data()
    print(f"Loaded {len(df)} ads.")
    print("\nColumns:", df.columns.tolist())
    loader.preview_dataframe(
        df,
        n=3,
        columns=["ad_id", "main_topic", "sub_topic",
                 "deceptive", "product", "condition", "content"],
        header="OSF Health Ads Preview (first 3 rows)"
    )
