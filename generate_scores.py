import csv
import json
from pathlib import Path

DATA_PATH = Path('neurips-2025-orals-posters.json')
OUTPUT_DIR = Path('scores')

NOVELTY_TERMS = {
    'novel',
    'first of its kind',
    'first-ever',
    'state-of-the-art',
    'sota',
    'benchmark',
    'dataset',
    'new method',
    'new approach',
    'innovative',
    'introduce',
    'propose',
    'framework',
    'architecture',
}
REPRO_TERMS = {
    'code',
    'open-source',
    'release',
    'github',
    'available',
    'reproduc',
}


def load_data(path: Path):
    with path.open('r', encoding='utf-8') as fh:
        payload = json.load(fh)
    return payload['results']


def count_term_hits(text: str, terms):
    text_lower = text.lower()
    return sum(1 for term in terms if term in text_lower)


def compute_innovation_score(paper):
    abstract = (paper.get('abstract') or '').lower()
    keywords = ' '.join(paper.get('keywords') or [])
    combined_text = f"{abstract}\n{keywords.lower()}"

    novelty_hits = count_term_hits(combined_text, NOVELTY_TERMS)
    reproducibility_hits = count_term_hits(combined_text, REPRO_TERMS)

    # Additional credit for longer, information-dense abstracts
    word_count = len(abstract.split())
    length_bonus = 0
    if word_count >= 200:
        length_bonus = 1.0
    elif word_count >= 120:
        length_bonus = 0.5

    score = 1.0 + 0.8 * novelty_hits + 0.4 * reproducibility_hits + length_bonus
    return round(max(1.0, min(5.0, score)), 2)


def compute_interdisciplinary_score(paper):
    institutions = {
        (author.get('institution') or '').strip().lower()
        for author in paper.get('authors') or []
        if (author.get('institution') or '').strip()
    }
    unique_count = len(institutions)

    if unique_count == 0:
        score = 1.5
    elif unique_count == 1:
        score = 1.8
    elif unique_count == 2:
        score = 2.8
    elif unique_count == 3:
        score = 3.6
    elif unique_count == 4:
        score = 4.3
    else:
        score = 5.0

    # Light boost when authors span multiple continents inferred by simple heuristics
    continent_hits = 0
    continent_keywords = {
        'university of': 'na',
        'institute of technology': 'global',
        'china': 'asia',
        'india': 'asia',
        'japan': 'asia',
        'korea': 'asia',
        'united kingdom': 'europe',
        'uk': 'europe',
        'germany': 'europe',
        'france': 'europe',
        'spain': 'europe',
        'switzerland': 'europe',
        'australia': 'oceania',
        'canada': 'na',
        'usa': 'na',
        'u.s.': 'na',
        'italy': 'europe',
        'brazil': 'south america',
        'mexico': 'north america',
        'singapore': 'asia',
    }
    continents = {
        continent
        for inst in institutions
        for key, continent in continent_keywords.items()
        if key in inst
    }
    if len(continents) >= 2:
        score = min(5.0, score + 0.4)
    return round(score, 2)


def compute_session_adjusted_highlight(paper, innovation_score, interdisciplinary_score):
    keywords = paper.get('keywords') or []
    keyword_diversity = min(1.0, len({kw.lower() for kw in keywords}) / 10.0)

    innovation_norm = innovation_score / 5.0
    interdisciplinary_norm = interdisciplinary_score / 5.0
    base_score = (
        0.5 * innovation_norm
        + 0.3 * interdisciplinary_norm
        + 0.2 * keyword_diversity
    ) * 5.0

    event_type = (paper.get('event_type') or paper.get('eventtype') or '').lower()
    if 'poster' in event_type:
        base_score *= 1.1
    elif 'oral' in event_type:
        base_score *= 0.95
    elif 'spotlight' in event_type:
        base_score *= 1.05

    # Encourage visibility for works with supplementary media
    media_entries = paper.get('eventmedia') or []
    if media_entries:
        base_score += min(0.5, 0.1 * len(media_entries))

    return round(max(1.0, min(5.0, base_score)), 2)


def write_csv(path: Path, rows):
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['paper_id', 'title', 'score'])
        writer.writerows(rows)


def main():
    papers = load_data(DATA_PATH)
    OUTPUT_DIR.mkdir(exist_ok=True)

    innovation_rows = []
    interdisciplinary_rows = []
    session_rows = []

    for paper in papers:
        paper_id = paper.get('id')
        title = paper.get('name')

        innovation_score = compute_innovation_score(paper)
        interdisciplinary_score = compute_interdisciplinary_score(paper)
        session_score = compute_session_adjusted_highlight(
            paper, innovation_score, interdisciplinary_score
        )

        innovation_rows.append((paper_id, title, innovation_score))
        interdisciplinary_rows.append((paper_id, title, interdisciplinary_score))
        session_rows.append((paper_id, title, session_score))

    write_csv(OUTPUT_DIR / 'innovation_score.csv', innovation_rows)
    write_csv(OUTPUT_DIR / 'interdisciplinary_score.csv', interdisciplinary_rows)
    write_csv(OUTPUT_DIR / 'session_adjusted_highlight_score.csv', session_rows)


if __name__ == '__main__':
    main()
