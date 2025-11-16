import json
from pathlib import Path
from typing import Dict, List

DATA_FILE = Path("neurips-2025-orals-posters.json")
OUTPUT_FILE = Path("prefix_tuning_papers.md")

KEYWORDS = [
    "soft prompt",
    "soft-prompt",
    "softprompt",
    "prefix tuning",
    "prompt tuning",
    "p-tuning",
    "p tuning",
    "soft prefix",
    "prefix prompt",
    "prompt-based tuning",
]


def load_data(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def matches_keywords(entry: Dict) -> bool:
    text_bits: List[str] = [
        entry.get("name") or "",
        entry.get("abstract") or "",
    ]
    keywords = entry.get("keywords")
    if isinstance(keywords, list):
        text_bits.extend([kw or "" for kw in keywords])
    text = " ".join(text_bits).casefold()
    return any(keyword in text for keyword in KEYWORDS)


def matched_terms(entry: Dict) -> List[str]:
    text_bits: List[str] = [
        entry.get("name") or "",
        entry.get("abstract") or "",
    ]
    keywords = entry.get("keywords")
    if isinstance(keywords, list):
        text_bits.extend([kw or "" for kw in keywords])
    text = " ".join(text_bits).casefold()
    return sorted({kw for kw in KEYWORDS if kw in text})


def format_authors(entry: Dict) -> str:
    authors = entry.get("authors") or []
    names = [author.get("fullname", "") for author in authors]
    return ", ".join(filter(None, names))


def format_entry(entry: Dict) -> str:
    lines = [f"## {entry.get('name', 'Untitled Paper')}", ""]
    authors = format_authors(entry)
    if authors:
        lines.append(f"- Authors: {authors}")
    session = entry.get("session") or entry.get("event_type") or entry.get("eventtype")
    if session:
        lines.append(f"- Session: {session}")
    pdf_url = entry.get("paper_pdf_url")
    if pdf_url:
        lines.append(f"- PDF: {pdf_url}")
    matched = matched_terms(entry)
    if matched:
        lines.append(f"- Matched terms: {', '.join(matched)}")
    abstract = entry.get("abstract")
    if abstract:
        lines.append("")
        lines.append(abstract.strip())
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    data = load_data(DATA_FILE)
    results = [entry for entry in data.get("results", []) if matches_keywords(entry)]
    results.sort(key=lambda entry: entry.get("name", ""))

    header = [
        "# NeurIPS 2025 papers using soft prompts or prefix-based fine-tuning",
        "",
        f"Found {len(results)} papers matching keywords: {', '.join(KEYWORDS)}.",
        "",
    ]

    sections = [format_entry(entry) for entry in results]
    OUTPUT_FILE.write_text("\n".join(header + sections))
    print(f"Wrote {len(results)} entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
