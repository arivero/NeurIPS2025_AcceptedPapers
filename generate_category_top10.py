#!/usr/bin/env python3
"""Generate category-specific top-10 lists for NeurIPS 2025 papers.

This script expects the score CSV files produced by :mod:`generate_scores`
so that every accepted paper has innovation, interdisciplinary, and
session-adjusted highlight scores.  It then builds a composite score and
exports four rankings:

* Large language model (LLM) papers
* Papers led by private companies
* Papers led by universities
* Papers led by European institutions

The outputs are written to the ``top10`` directory as both CSV files and a
single Markdown report summarising all categories.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import pandas as pd


SCORE_FILES: Dict[str, str] = {
    "innovation": "innovation_score.csv",
    "interdisciplinary": "interdisciplinary_score.csv",
    "session_adjusted_highlight": "session_adjusted_highlight_score.csv",
}

SCORE_WEIGHTS: Dict[str, float] = {
    "innovation": 0.6,
    "interdisciplinary": 0.25,
    "session_adjusted_highlight": 0.15,
}

LLM_TERMS: Iterable[str] = {
    "large language model",
    "llm",
    "language model",
    "foundation model",
    "instruction tuning",
    "in-context learning",
    "reasoning model",
}

PRIVATE_COMPANY_KEYWORDS: Iterable[str] = {
    "google",
    "deepmind",
    "google research",
    "google deepmind",
    "meta",
    "facebook ai",
    "amazon",
    "aws",
    "microsoft",
    "openai",
    "anthropic",
    "nvidia",
    "apple",
    "ibm",
    "salesforce",
    "bytedance",
    "tiktok",
    "alibaba",
    "tencent",
    "huawei",
    "baidu",
    "sensetime",
    "bloomberg",
    "adobe",
    "intel",
    "samsung",
    "qualcomm",
    "sony",
    "xiaomi",
    "naver",
    "yandex",
    "oracle",
    "databricks",
    "graphcore",
    "arm",
    "bosch",
    "siemens",
    "sap",
    "servicenow",
    "snowflake",
    "uber",
    "lyft",
}

UNIVERSITY_KEYWORDS: Iterable[str] = {
    "university",
    "université",
    "universitat",
    "università",
    "universidade",
    "universidad",
    "college",
    "polytechnic",
    "institute of technology",
    "école",
    "schule",
    "school of",
}

EUROPE_KEYWORDS: Iterable[str] = {
    "united kingdom",
    "uk",
    "england",
    "scotland",
    "wales",
    "ireland",
    "germany",
    "france",
    "spain",
    "portugal",
    "italy",
    "netherlands",
    "belgium",
    "switzerland",
    "zurich",
    "lausanne",
    "austria",
    "sweden",
    "norway",
    "finland",
    "denmark",
    "iceland",
    "estonia",
    "latvia",
    "lithuania",
    "poland",
    "czech",
    "slovak",
    "hungary",
    "romania",
    "bulgaria",
    "croatia",
    "serbia",
    "slovenia",
    "greece",
    "cyprus",
    "malta",
    "luxembourg",
    "liechtenstein",
    "monaco",
    "andorra",
    "san marino",
    "vatican",
    "norwegian",
    "french",
    "italian",
    "german",
    "dutch",
    "spanish",
    "portuguese",
    "danish",
    "swiss",
    "swedish",
    "finnish",
    "polish",
}


@dataclass
class PaperMetadata:
    """Container for auxiliary metadata used in category detection."""

    title: str
    decision: str | None
    openreview_url: str | None
    abstract: str
    keywords: List[str]
    institutions: List[str]


@dataclass
class CategoryDefinition:
    """Describe an individual ranking."""

    name: str
    label: str
    predicate: Callable[[PaperMetadata], bool]


def _load_scores(scores_dir: Path) -> pd.DataFrame:
    """Load and merge all score CSV files into a single DataFrame."""

    frames: List[pd.DataFrame] = []
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
        frames.append(df.rename(columns={"score": f"{score_name}_score"}))

    merged = frames[0]
    for other in frames[1:]:
        merged = merged.merge(
            other,
            on=["paper_id", "title"],
            how="inner",
            validate="one_to_one",
        )
    return merged


def _min_max_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalise a numeric series."""

    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return pd.Series([1.0] * len(series), index=series.index)
    return (series - min_value) / (max_value - min_value)


def _compute_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the weighted composite score."""

    result = df.copy()
    for score_name in SCORE_FILES:
        score_column = f"{score_name}_score"
        normalised_column = f"{score_name}_normalised"
        result[normalised_column] = _min_max_normalize(result[score_column])

    result["composite_score"] = 0.0
    for score_name, weight in SCORE_WEIGHTS.items():
        normalised_column = f"{score_name}_normalised"
        result["composite_score"] += result[normalised_column] * weight
    return result


def _normalise_institution(raw: str | None) -> str:
    if not raw:
        return ""
    return raw.strip()


def _load_metadata(json_path: Path) -> Dict[int, PaperMetadata]:
    """Load metadata such as abstracts, keywords, and institutions."""

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    results: Dict[int, PaperMetadata] = {}
    for entry in payload.get("results", []):
        paper_id = entry.get("id")
        if paper_id is None:
            continue

        institutions = [
            _normalise_institution(author.get("institution"))
            for author in entry.get("authors", [])
            if _normalise_institution(author.get("institution"))
        ]
        metadata = PaperMetadata(
            title=entry.get("name", ""),
            decision=entry.get("decision"),
            openreview_url=entry.get("paper_url")
            or entry.get("virtualsite_url")
            or entry.get("sourceurl"),
            abstract=entry.get("abstract", ""),
            keywords=list(entry.get("keywords", []) or []),
            institutions=institutions,
        )
        results[int(paper_id)] = metadata
    return results


def _attach_metadata(df: pd.DataFrame, metadata: Dict[int, PaperMetadata]) -> pd.DataFrame:
    """Append metadata columns to the scores DataFrame."""

    records: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        paper_id = int(row["paper_id"])
        meta = metadata.get(paper_id)
        if not meta:
            meta = PaperMetadata("", None, None, "", [], [])
        records.append(
            {
                "paper_id": paper_id,
                "title": meta.title or row["title"],
                "decision": meta.decision,
                "openreview_url": meta.openreview_url,
                "abstract": meta.abstract,
                "keywords": meta.keywords,
                "institutions": meta.institutions,
            }
        )

    meta_df = pd.DataFrame.from_records(records)
    merged = df.drop(columns=["title"]).merge(
        meta_df,
        on="paper_id",
        how="left",
        validate="one_to_one",
    )
    # Ensure title column is last merge result to avoid duplicate naming
    ordered_columns = [
        "paper_id",
        "title",
        "composite_score",
        "innovation_score",
        "interdisciplinary_score",
        "session_adjusted_highlight_score",
        "innovation_normalised",
        "interdisciplinary_normalised",
        "session_adjusted_highlight_normalised",
        "decision",
        "openreview_url",
        "abstract",
        "keywords",
        "institutions",
    ]
    return merged[ordered_columns]


def _text_blob(meta: PaperMetadata) -> str:
    keyword_text = " ".join(meta.keywords)
    components = [meta.title, meta.abstract, keyword_text]
    return " ".join(filter(None, components)).lower()


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _is_llm(meta: PaperMetadata) -> bool:
    return _contains_any(_text_blob(meta), LLM_TERMS)


def _has_private_company(meta: PaperMetadata) -> bool:
    for institution in meta.institutions:
        lowered = institution.lower()
        if any(keyword in lowered for keyword in PRIVATE_COMPANY_KEYWORDS):
            return True
    return False


def _has_university(meta: PaperMetadata) -> bool:
    for institution in meta.institutions:
        lowered = institution.lower()
        if any(keyword in lowered for keyword in UNIVERSITY_KEYWORDS):
            return True
    return False


def _is_european(meta: PaperMetadata) -> bool:
    for institution in meta.institutions:
        lowered = institution.lower()
        if any(keyword in lowered for keyword in EUROPE_KEYWORDS):
            return True
    return False


def _format_institutions(institutions: List[str]) -> str:
    if not institutions:
        return "N/A"
    unique = []
    seen = set()
    for inst in institutions:
        if inst not in seen:
            unique.append(inst)
            seen.add(inst)
    return "; ".join(unique)


def _build_ranked_table(df: pd.DataFrame, metadata: Dict[int, PaperMetadata], predicate: Callable[[PaperMetadata], bool]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for paper_id, meta in metadata.items():
        if not predicate(meta):
            continue
        score_row = df[df["paper_id"] == paper_id]
        if score_row.empty:
            continue
        record = score_row.iloc[0].to_dict()
        record.update(
            {
                "paper_id": paper_id,
                "title": meta.title,
                "decision": meta.decision,
                "openreview_url": meta.openreview_url,
                "institutions": meta.institutions,
            }
        )
        rows.append(record)

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame.from_records(rows)
    table = table.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
    table["rank"] = table.index + 1
    table["institutions_formatted"] = table["institutions"].apply(_format_institutions)
    return table[
        [
            "rank",
            "paper_id",
            "title",
            "composite_score",
            "innovation_score",
            "interdisciplinary_score",
            "session_adjusted_highlight_score",
            "decision",
            "openreview_url",
            "institutions_formatted",
        ]
    ]


def _write_category_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False)


def _append_category_markdown(lines: List[str], definition: CategoryDefinition, table: pd.DataFrame) -> None:
    lines.append(f"## {definition.label}")
    lines.append("")
    if table.empty:
        lines.append("No papers matched this category.\n")
        return

    lines.append("| Rank | Title | Composite Score | Lead Institutions | OpenReview |")
    lines.append("| --- | --- | --- | --- | --- |")
    for _, row in table.iterrows():
        title = row["title"].replace("|", "\\|")
        composite = f"{row['composite_score']:.3f}"
        institutions = row["institutions_formatted"].replace("|", "\\|")
        openreview = row.get("openreview_url") or "N/A"
        if openreview != "N/A":
            openreview = f"[Link]({openreview})"
        lines.append(
            f"| {row['rank']} | {title} | {composite} | {institutions} | {openreview} |"
        )
    lines.append("")


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    scores_dir = repo_root / "scores"
    json_path = repo_root / "neurips-2025-orals-posters.json"
    output_dir = repo_root / "top10"

    scores = _load_scores(scores_dir)
    scored = _compute_composite_scores(scores)
    metadata = _load_metadata(json_path)
    enriched = _attach_metadata(scored, metadata)

    categories = [
        CategoryDefinition("llm", "Top 10 Large Language Model Papers", _is_llm),
        CategoryDefinition(
            "private_companies", "Top 10 Papers from Private Companies", _has_private_company
        ),
        CategoryDefinition(
            "universities", "Top 10 Papers from Universities", _has_university
        ),
        CategoryDefinition(
            "europe", "Top 10 European Papers", _is_european
        ),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_lines = [
        "# Category Top 10 Lists for NeurIPS 2025",
        "",
        "Each ranking uses the same composite score as the revolutionary list but is filtered",
        "to highlight specific slices of the community.",
        "",
    ]

    for definition in categories:
        table = _build_ranked_table(enriched, metadata, definition.predicate).head(10)
        csv_path = output_dir / f"category_top10_{definition.name}.csv"
        _write_category_csv(table, csv_path)
        _append_category_markdown(markdown_lines, definition, table)

    markdown_path = output_dir / "category_top10.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    print(f"Generated category-specific top 10 lists in: {output_dir}")


if __name__ == "__main__":
    main()
