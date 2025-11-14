#!/usr/bin/env python3
"""Generate category-specific top-10 rankings for NeurIPS 2025 papers.

The script assumes that ``generate_scores.py`` has already been executed so
that per-paper scores exist in the ``scores`` directory. It computes the same
composite metric as ``generate_revolutionary_top10.py`` and then filters papers
into several stakeholder-focused leaderboards:

* Top 10 papers focused on large language models (LLMs).
* Top 10 papers featuring private company participation.
* Top 10 papers led by universities (no industry-affiliated authors).
* Top 10 papers featuring European institutions.

Outputs are saved in ``top10`` as both CSV files and a consolidated Markdown
report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Set

import pandas as pd

from generate_revolutionary_top10 import (
    _attach_metadata,
    _compute_composite_scores,
    _load_metadata,
    _load_scores,
)

REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "neurips-2025-orals-posters.json"
SCORES_DIR = REPO_ROOT / "scores"
OUTPUT_DIR = REPO_ROOT / "top10"

LLM_KEYWORDS: Set[str] = {
    "large language model",
    "large-language model",
    "language model",
    "llm",
    "foundation model",
    "in-context learning",
    "instruction tuning",
    "alignment",
    "reasoning model",
    "chatbot",
    "decoder-only",
    "multi-modal large model",
    "prompt learning",
}

PRIVATE_COMPANY_KEYWORDS: Set[str] = {
    "google",
    "deepmind",
    "waymo",
    "x company",
    "openai",
    "anthropic",
    "meta",
    "facebook",
    "microsoft",
    "amazon",
    "aws",
    "apple",
    "nvidia",
    "ibm",
    "salesforce",
    "bytedance",
    "tencent",
    "alibaba",
    "baidu",
    "sensetime",
    "sense time",
    "huawei",
    "intel",
    "qualcomm",
    "samsung",
    "oracle",
    "uber",
    "adobe",
    "snap",
    "xiaomi",
    "databricks",
    "cohere",
    "megvii",
    "megvii technology",
    "kuaishou",
    "jd.com",
    "jd explore",
    "meituan",
    "ant group",
    "ping an",
    "horizon robotics",
    "hewlett packard",
    "asml",
    "stability ai",
    "sony",
    "china telecom",
    "memtensor",
    "didi",
    "hpc-ai",
    "hugging face",
}

UNIVERSITY_KEYWORDS: Set[str] = {
    "university",
    "université",
    "universita",
    "universitat",
    "college",
    "institut",
    "institute of technology",
    "polytechnic",
    "école",
    "school of",
}

EUROPE_KEYWORDS: Set[str] = {
    "austria",
    "belgium",
    "croatia",
    "cyprus",
    "czech",
    "denmark",
    "estonia",
    "finland",
    "france",
    "germany",
    "greece",
    "hungary",
    "iceland",
    "ireland",
    "italy",
    "latvia",
    "liechtenstein",
    "lithuania",
    "luxembourg",
    "malta",
    "netherlands",
    "norway",
    "poland",
    "portugal",
    "romania",
    "slovakia",
    "slovenia",
    "spain",
    "sweden",
    "switzerland",
    "turkey",
    "united kingdom",
    "uk",
    "england",
    "scotland",
    "wales",
}


def _load_full_metadata(path: Path) -> Dict[int, Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results", [])
    return {int(entry["id"]): entry for entry in results if entry.get("id") is not None}


def _collect_institutions(paper: Mapping[str, object]) -> List[str]:
    institutions: List[str] = []
    for author in paper.get("authors", []) or []:
        inst = (author.get("institution") or "")
        inst = inst.strip(" ,;")
        if inst:
            institutions.append(inst)
    return institutions


def _normalise_text(values: Iterable[str]) -> List[str]:
    return [value.lower() for value in values]


def _paper_text_blurb(paper: Mapping[str, object]) -> str:
    components: List[str] = [paper.get("name") or "", paper.get("abstract") or ""]
    keywords = paper.get("keywords") or []
    if isinstance(keywords, Sequence):
        components.extend(str(keyword) for keyword in keywords)
    return " \n ".join(component.lower() for component in components if component)


def _matches_any(text: str, keywords: Set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def is_llm_paper(paper: Mapping[str, object]) -> bool:
    return _matches_any(_paper_text_blurb(paper), LLM_KEYWORDS)


def is_private_company_paper(paper: Mapping[str, object]) -> bool:
    institutions = _normalise_text(_collect_institutions(paper))
    return any(any(keyword in inst for keyword in PRIVATE_COMPANY_KEYWORDS) for inst in institutions)


def is_university_led_paper(paper: Mapping[str, object]) -> bool:
    institutions = _normalise_text(_collect_institutions(paper))
    if not institutions:
        return False
    has_university = any(any(keyword in inst for keyword in UNIVERSITY_KEYWORDS) for inst in institutions)
    has_private = any(any(keyword in inst for keyword in PRIVATE_COMPANY_KEYWORDS) for inst in institutions)
    return has_university and not has_private


def is_european_paper(paper: Mapping[str, object]) -> bool:
    institutions = _normalise_text(_collect_institutions(paper))
    return any(any(keyword in inst for keyword in EUROPE_KEYWORDS) for inst in institutions)


CategoryPredicate = Callable[[Mapping[str, object]], bool]


def _prepare_scored_dataframe() -> pd.DataFrame:
    scores = _load_scores(SCORES_DIR)
    composite = _compute_composite_scores(scores)
    metadata = _load_metadata(DATA_PATH)
    enriched = _attach_metadata(composite, metadata)
    enriched["paper_id"] = enriched["paper_id"].astype(int)
    return enriched


def _rank_category(
    scored_df: pd.DataFrame,
    full_metadata: Mapping[int, Mapping[str, object]],
    predicate: CategoryPredicate,
    top_n: int = 10,
) -> pd.DataFrame:
    matches = []
    for _, row in scored_df.iterrows():
        paper_id = int(row["paper_id"])
        paper_meta = full_metadata.get(paper_id)
        if paper_meta is None:
            continue
        if not predicate(paper_meta):
            continue
        institutions = ", ".join(
            sorted(
                {
                    inst
                    for inst in _collect_institutions(paper_meta)
                    if inst
                }
            )
        )
        matches.append(
            {
                "paper_id": paper_id,
                "title": row["title"],
                "composite_score": row["composite_score"],
                "innovation_score": row["innovation_score"],
                "interdisciplinary_score": row["interdisciplinary_score"],
                "session_adjusted_highlight_score": row["session_adjusted_highlight_score"],
                "decision": row.get("decision"),
                "openreview_url": row.get("openreview_url"),
                "institutions": institutions,
            }
        )

    if not matches:
        return pd.DataFrame()

    result = pd.DataFrame(matches)
    ranked = result.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    return ranked.head(top_n)


def _write_category_outputs(category: str, ranking: pd.DataFrame) -> None:
    if ranking.empty:
        return

    csv_path = OUTPUT_DIR / f"top10_{category}.csv"
    ranking.to_csv(csv_path, index=False)


def _build_markdown_report(category_tables: Dict[str, pd.DataFrame]) -> None:
    lines: List[str] = [
        "# Category-Specific Top 10 NeurIPS 2025 Papers",
        "",
        "These rankings are derived from the self-scored dataset covering all 5,945 accepted papers.",
    ]

    for category, table in category_tables.items():
        if table.empty:
            continue
        pretty_name = {
            "llm": "Large Language Model Focus",
            "private_company": "Private Company Participation",
            "university": "University-Led Research",
            "europe": "European Institutions",
        }[category]
        lines.extend(["", f"## {pretty_name}", "", "| Rank | Title | Composite | Institutions | OpenReview |", "| --- | --- | --- | --- | --- |"])
        for _, row in table.iterrows():
            title = row["title"].replace("|", "\\|")
            composite = f"{row['composite_score']:.3f}"
            institutions = row["institutions"].replace("|", "\\|")
            openreview = row.get("openreview_url") or "N/A"
            if openreview != "N/A":
                openreview = f"[Link]({openreview})"
            lines.append(
                f"| {int(row['rank'])} | {title} | {composite} | {institutions} | {openreview} |"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    markdown_path = OUTPUT_DIR / "category_rankings.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scored_df = _prepare_scored_dataframe()
    full_metadata = _load_full_metadata(DATA_PATH)

    categories: Dict[str, CategoryPredicate] = {
        "llm": is_llm_paper,
        "private_company": is_private_company_paper,
        "university": is_university_led_paper,
        "europe": is_european_paper,
    }

    rankings: Dict[str, pd.DataFrame] = {}
    for name, predicate in categories.items():
        ranking = _rank_category(scored_df, full_metadata, predicate)
        rankings[name] = ranking
        _write_category_outputs(name, ranking)

    _build_markdown_report(rankings)

    print("Generated category-specific top 10 reports in 'top10'.")


if __name__ == "__main__":
    main()
