"""
Validate the performance of OCR approaches against hand-made ground truth.

One-way recall: every ground-truth word must be found in the model annotation,
not the reverse (the GT only holds the words worth transcribing). Matching is
word-per-word and fuzzy (normalised Levenshtein), bounded to the section the GT
word lives in (meta vs content). A GT word found only in the *other* section is
"misplaced" -- counted separately so category leakage is visible.
"""

import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from ocr.annotation import Annotation

SECTIONS = ("meta", "content")


def normalize(text: str) -> str:
    """Lowercase + strip accents (é->e) so matching is form-insensitive."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tokenize(lines: list[str]) -> list[str]:
    """Section lines -> word tokens. Splits on whitespace and punctuation."""
    return re.findall(r"\w+", normalize(" ".join(lines)))


@dataclass
class WordResult:
    gt: str
    match: str | None  # best in-section candidate
    sim: float  # best in-section similarity [0,1]
    status: str  # found | misplaced | missed


@dataclass
class SectionMetrics:
    """Recall/quality numbers for one section (or a micro-averaged pool)."""

    n_gt_words: int
    recall_at_tau: float | None  # found / n_gt -- detection in correct section
    recall_any_section: float | None  # (found+misplaced)/n_gt -- placement-blind
    mean_best_sim: float | None  # avg best in-section sim, all GT words (soft)
    cer: float | None  # char error rate over matched (found) pairs
    misplaced_rate: float | None  # found only in the wrong section
    missed_rate: float | None  # found nowhere
    n_found: int
    n_misplaced: int
    n_missed: int

    @classmethod
    def from_words(cls, words: list[WordResult]) -> "SectionMetrics":
        n = len(words)
        c = {
            s: sum(w.status == s for w in words)
            for s in ("found", "misplaced", "missed")
        }
        found = [w for w in words if w.status == "found"]
        cer_num = sum(Levenshtein.distance(w.gt, w.match or "") for w in found)
        cer_den = sum(len(w.gt) for w in found)
        return cls(
            n_gt_words=n,
            recall_at_tau=_frac(c["found"], n),
            recall_any_section=_frac(c["found"] + c["misplaced"], n),
            mean_best_sim=round(sum(w.sim for w in words) / n, 4) if n else None,
            cer=round(cer_num / cer_den, 4) if cer_den else None,
            misplaced_rate=_frac(c["misplaced"], n),
            missed_rate=_frac(c["missed"], n),
            n_found=c["found"],
            n_misplaced=c["misplaced"],
            n_missed=c["missed"],
        )


@dataclass
class SectionResult:
    metrics: SectionMetrics
    words: list[WordResult]


@dataclass
class ImageResult:
    meta: SectionResult
    content: SectionResult


@dataclass
class Config:
    tau: float
    tokenizer: str = r"accent-stripped, lowercased \w+"
    note: str = "one-way recall (GT->prediction); matching bounded per section"


@dataclass
class Report:
    config: Config
    overall: dict[str, SectionMetrics]  # meta | content | combined
    images: dict[str, ImageResult]


def _frac(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _best(token: str, candidates: list[str]) -> tuple[str | None, float]:
    """Highest normalised-Levenshtein similarity of token among candidates."""
    if not candidates:
        return None, 0.0
    hit = process.extractOne(
        token, candidates, scorer=Levenshtein.normalized_similarity
    )
    return hit[0], hit[1]  # (candidate, score) -- score already in [0,1]


def _score_section(
    gt_tokens: list[str], same: list[str], other: list[str], tau: float
) -> list[WordResult]:
    """Match each GT token in its own section; flag cross-section-only hits."""
    words = []
    for tok in gt_tokens:
        match, sim = _best(tok, same)
        _, sim_other = _best(tok, other)
        if sim >= tau:
            status = "found"
        elif sim_other >= tau:
            status = "misplaced"
        else:
            status = "missed"
        words.append(WordResult(tok, match, round(sim, 4), status))
    return words


def validate_pair(gt: Annotation, pred: Annotation, tau: float) -> ImageResult:
    """Score one (ground-truth, prediction) annotation pair, per section."""
    pred_tok = {s: tokenize(getattr(pred, s)) for s in SECTIONS}
    sections = {}
    for s in SECTIONS:
        other = "content" if s == "meta" else "meta"
        words = _score_section(
            tokenize(getattr(gt, s)), pred_tok[s], pred_tok[other], tau
        )
        sections[s] = SectionResult(SectionMetrics.from_words(words), words)
    return ImageResult(**sections)


def validate(
    pairs: dict[str, tuple[Annotation, Annotation]], tau: float = 0.8
) -> Report:
    """Full report: config, overall (micro-averaged), and per-image breakdowns.

    `pairs` maps an image stem to its (ground_truth, prediction) annotations.
    """
    images = {stem: validate_pair(gt, pred, tau) for stem, (gt, pred) in pairs.items()}

    def pool(*sections: str) -> SectionMetrics:
        words = [
            w
            for img in images.values()
            for s in sections
            for w in getattr(img, s).words
        ]
        return SectionMetrics.from_words(words)

    overall = {
        "meta": pool("meta"),
        "content": pool("content"),
        "combined": pool("meta", "content"),
    }
    return Report(Config(tau=tau), overall, images)


def _selfcheck():
    gt = Annotation(meta=["Kerk Vlaskeerders"], content=["anno 1923", "Gent"])
    pred = Annotation(
        meta=["kerk vlaskerders"],  # 1 exact, 1 fuzzy (1 char off)
        content=["Gent", "1923"],  # "anno" missing, "1923" found, "Gent" found
    )
    rep = validate({"img1": (gt, pred)}, tau=0.8)

    meta = rep.images["img1"].meta
    assert meta.metrics.n_found == 2, meta.metrics  # kerk exact, vlaskeerders fuzzy
    assert meta.metrics.cer is not None

    content = rep.images["img1"].content
    assert content.metrics.n_found == 2, content.metrics  # gent, 1923
    assert content.metrics.n_missed == 1, content.metrics  # anno
    anno = next(w for w in content.words if w.gt == "anno")
    assert anno.status == "missed", anno

    # category leak: content word landed only in the prediction's meta
    gt2 = Annotation(content=["zeppelin"])
    pred2 = Annotation(meta=["zeppelin"], content=[])
    rep2 = validate({"i": (gt2, pred2)}, tau=0.8)
    assert rep2.images["i"].content.words[0].status == "misplaced"

    assert rep.overall["combined"].n_gt_words == 5
    print("ok")


if __name__ == "__main__":
    _selfcheck()
