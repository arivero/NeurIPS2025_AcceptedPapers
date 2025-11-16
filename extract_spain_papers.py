#!/usr/bin/env python3
"""Extract NeurIPS 2025 papers with authors affiliated in Spain.

This utility filters the master ``neurips-2025-orals-posters.json`` payload for
papers that list at least one author affiliated with a Spanish institution. It
uses city- and institution-level keywords to match Spain-based affiliations and
writes the filtered set to ``top10/spain_papers.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "neurips-2025-orals-posters.json"
OUTPUT_PATH = REPO_ROOT / "top10" / "spain_papers.csv"

SPAIN_KEYWORDS: Dict[str, str] = {
    "spain": "Country name",
    "barcelona": "Barcelona metropolitan institutions",
    "madrid": "Madrid metropolitan institutions",
    "valencia": "Valencian community institutions",
    "sevilla": "Seville institutions (Spanish spelling)",
    "seville": "Seville institutions (English spelling)",
    "zaragoza": "Zaragoza institutions",
    "granada": "Granada institutions",
    "malaga": "Malaga institutions",
    "alicante": "Alicante institutions",
    "bilbao": "Bilbao institutions",
    "pamplona": "Pamplona institutions",
    "navarra": "Navarra institutions",
    "basque": "Basque country institutions",
    "santiago de compostela": "Galician institutions",
    "catalonia": "Catalan institutions",
    "catalunya": "Catalan institutions (Catalan spelling)",
    "pompeu fabra": "Universitat Pompeu Fabra",
    "universitat autónoma de barcelona": "UAB",
    "universitat oberta de catalunya": "UOC",
    "universitat pompeu fabra": "UPF",
    "universidad politécnica de madrid": "UPM",
    "universidad carlos iii": "UC3M",
    "universidad politécnica de valencia": "UPV",
    "polytechnic university of valencia": "UPV (English)",
    "university of the basque country": "UPV/EHU",
    "basque center for applied mathematics": "BCAM",
    "barcelona supercomputing center": "BSC",
    "computer vision center barcelona": "CVC Barcelona",
    "universitat autónoma de barcelona": "UAB",
}


def _collect_institutions(paper: Dict[str, object]) -> List[str]:
    institutions: List[str] = []
    for author in paper.get("authors", []) or []:
        inst = (author.get("institution") or "").strip(" ,;")
        if inst:
            institutions.append(inst)
    return institutions


def _normalise(values: Iterable[str]) -> List[str]:
    return [value.lower() for value in values]


def _matches_spain(institutions: Iterable[str]) -> bool:
    for inst in _normalise(institutions):
        if any(keyword in inst for keyword in SPAIN_KEYWORDS):
            return True
    return False


def _extract_openreview_url(paper: Dict[str, object]) -> str:
    for media in paper.get("eventmedia", []) or []:
        if media.get("name") == "OpenReview" and media.get("uri"):
            return str(media.get("uri"))
    return ""


def extract_spain_papers() -> pd.DataFrame:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    results = payload.get("results", [])
    rows: List[Dict[str, object]] = []

    for paper in results:
        institutions = _collect_institutions(paper)
        if not institutions or not _matches_spain(institutions):
            continue

        author_names = [
            author.get("fullname")
            for author in paper.get("authors", []) or []
            if author.get("fullname")
        ]

        rows.append(
            {
                "paper_id": paper.get("id"),
                "title": paper.get("name"),
                "decision": paper.get("decision"),
                "track_source": paper.get("sourceurl"),
                "openreview_url": _extract_openreview_url(paper),
                "author_names": "; ".join(author_names),
                "author_institutions": "; ".join(institutions),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("title").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} Spain-affiliated papers to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    extract_spain_papers()
