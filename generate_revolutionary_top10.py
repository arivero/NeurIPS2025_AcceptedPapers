#!/usr/bin/env python3
"""Generate the top-10 most revolutionary NeurIPS 2025 papers.

The ranking is based on a composite score computed from the innovation,
interdisciplinary, and session-adjusted highlight scores. The script
normalizes each score using min-max normalization and applies a set of
weights that emphasize innovation while still rewarding cross-disciplinary
impact and session recognition.

Outputs are written to the ``top10`` directory:
- ``revolutionary_papers.csv`` with detailed metrics for the top 10 papers.
- ``revolutionary_papers.md`` providing a Markdown summary table.

Usage::

    python generate_revolutionary_top10.py

"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


SCORE_FILES: Dict[str, str] = {
    "innovation": "innovation_score.csv",
    "interdisciplinary": "interdisciplinary_score.csv",
    "session_adjusted_highlight": "session_adjusted_highlight_score.csv",
}

# Weights favor innovation heavily while still rewarding breadth and recognition.
SCORE_WEIGHTS: Dict[str, float] = {
    "innovation": 0.6,
    "interdisciplinary": 0.25,
    "session_adjusted_highlight": 0.15,
}


@dataclass
class PaperMetadata:
    """Lightweight container for auxiliary paper metadata."""

    title: str
    decision: str | None
    openreview_url: str | None


def _load_scores(scores_dir: Path) -> pd.DataFrame:
    """Load and merge all score CSV files into a single DataFrame."""
    data_frames: List[pd.DataFrame] = []
    for score_name, filename in SCORE_FILES.items():
        csv_path = scores_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Score file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        expected_columns = {"paper_id", "title", "score"}
        if set(df.columns) != expected_columns:
            raise ValueError(
                f"Unexpected columns in {csv_path}. Expected {expected_columns}, got {set(df.columns)}"
            )
        renamed = df.rename(
            columns={
                "score": f"{score_name}_score",
            }
        )
        data_frames.append(renamed)

    merged = data_frames[0]
    for next_df in data_frames[1:]:
        merged = merged.merge(
            next_df,
            on=["paper_id", "title"],
            how="inner",
            validate="one_to_one",
        )
    return merged


def _min_max_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a pandas Series.

    If all values are identical the normalized score defaults to 1.0.
    """
    min_value = series.min()
    max_value = series.max()
    if pd.isna(min_value) or pd.isna(max_value):
        raise ValueError("Cannot normalize series with NaN values")

    if max_value == min_value:
        return pd.Series([1.0] * len(series), index=series.index)

    return (series - min_value) / (max_value - min_value)


def _compute_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized and weighted composite scores."""
    result = df.copy()
    for score_name in SCORE_FILES.keys():
        score_column = f"{score_name}_score"
        normalized_column = f"{score_name}_normalized"
        result[normalized_column] = _min_max_normalize(result[score_column])

    # Weighted composite score emphasising innovation.
    result["composite_score"] = 0.0
    for score_name, weight in SCORE_WEIGHTS.items():
        normalized_column = f"{score_name}_normalized"
        result["composite_score"] += result[normalized_column] * weight

    return result


def _load_metadata(json_path: Path) -> Dict[int, PaperMetadata]:
    """Load metadata such as decisions and OpenReview URLs."""
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    metadata: Dict[int, PaperMetadata] = {}
    for entry in payload.get("results", []):
        paper_id = entry.get("id")
        if paper_id is None:
            continue
        metadata[int(paper_id)] = PaperMetadata(
            title=entry.get("name", ""),
            decision=entry.get("decision"),
            openreview_url=entry.get("paper_url") or entry.get("virtualsite_url"),
        )
    return metadata


def _attach_metadata(df: pd.DataFrame, metadata: Dict[int, PaperMetadata]) -> pd.DataFrame:
    """Augment the DataFrame with metadata columns."""
    records: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        paper_id = int(row["paper_id"])
        meta = metadata.get(paper_id)
        records.append(
            {
                "paper_id": paper_id,
                "title": row["title"],
                "decision": meta.decision if meta else None,
                "openreview_url": meta.openreview_url if meta else None,
            }
        )

    meta_df = pd.DataFrame.from_records(records)
    return df.merge(meta_df, on=["paper_id", "title"], how="left", validate="one_to_one")


def export_top10(df: pd.DataFrame, output_dir: Path) -> None:
    """Persist the top 10 papers to CSV and Markdown outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    export_columns = [
        "rank",
        "paper_id",
        "title",
        "composite_score",
        "innovation_score",
        "interdisciplinary_score",
        "session_adjusted_highlight_score",
        "innovation_normalized",
        "interdisciplinary_normalized",
        "session_adjusted_highlight_normalized",
        "decision",
        "openreview_url",
    ]

    csv_path = output_dir / "revolutionary_papers.csv"
    df.to_csv(csv_path, index=False, columns=export_columns)

    markdown_lines: List[str] = [
        "# Top 10 Revolutionary NeurIPS 2025 Papers",
        "",
        "The ranking is computed using min-max normalized scores with weights of "
        "0.60 (innovation), 0.25 (interdisciplinary), and 0.15 (session-adjusted highlight).",
        "",
        "| Rank | Title | Composite Score | OpenReview |",
        "| --- | --- | --- | --- |",
    ]

    for _, row in df.iterrows():
        title = row["title"].replace("|", "\\|")
        composite = f"{row['composite_score']:.3f}"
        openreview = row.get("openreview_url") or "N/A"
        if openreview != "N/A":
            openreview = f"[Link]({openreview})"
        markdown_lines.append(
            f"| {row['rank']} | {title} | {composite} | {openreview} |"
        )

    md_path = output_dir / "revolutionary_papers.md"
    md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    scores_dir = repo_root / "scores"
    json_path = repo_root / "neurips-2025-orals-posters.json"
    output_dir = repo_root / "top10"

    scores = _load_scores(scores_dir)
    scored = _compute_composite_scores(scores)
    metadata = _load_metadata(json_path)
    enriched = _attach_metadata(scored, metadata)

    ranked = enriched.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    top10 = ranked.head(10)
    export_top10(top10, output_dir)

    print(f"Generated top 10 revolutionary papers at: {output_dir}")


if __name__ == "__main__":
    main()
