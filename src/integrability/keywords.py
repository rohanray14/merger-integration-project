"""Hoberg & Phillips (2018) p.17 integration-difficulty keyword rule."""
import re


def _pattern(words):
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.I)


def make_rule(label_cfg: dict):
    pats = [_pattern(label_cfg[k]) for k in ("acquisition_words", "integration_words", "difficulty_words")]

    def is_hit(paragraph: str) -> bool:
        return all(p.search(paragraph) for p in pats)

    return is_hit


def score(paragraphs: list[str], is_hit) -> dict:
    """paragraphs should already exclude the risk-factor section."""
    hits = sum(is_hit(p) for p in paragraphs)
    n = len(paragraphs)
    return {"hit": int(hits > 0), "hit_paragraphs": hits, "paragraphs": n,
            "intensity": hits / n if n else 0.0}
