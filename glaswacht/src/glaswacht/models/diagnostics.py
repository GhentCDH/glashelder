"""Data model for a single-image generation diagnostics trace.

Deliberately free of torch model code and of Flask: everything here works
against a plain tokenizer object, so it is unit-testable on CPU without a
checkpoint, and the viewer consumes only `DiagnosticsTrace.to_dict()`.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

RATING_CANDIDATES: tuple[str, ...] = ("Safe", "Unsafe", "Potentially Problematic")

_PROB_DIGITS = 6


def _round(value: float) -> float:
    return round(float(value), _PROB_DIGITS)


def token_pieces(tokenizer, ids: list[int]) -> list[str]:
    """Display text per token, such that ''.join(result) == tokenizer.decode(ids).

    Per-token `decode([id])` corrupts multi-byte characters and
    `convert_ids_to_tokens` leaks BPE artifacts, so decode incrementally and
    hold a token back until its bytes complete a character.
    """
    pieces, text_so_far, pending = [""] * len(ids), "", []
    for j in range(len(ids)):
        pending.append(j)
        cur = tokenizer.decode(ids[: j + 1])
        if cur.endswith("�"):
            continue
        pieces[pending[-1]] = cur[len(text_so_far) :]
        text_so_far, pending = cur, []
    return pieces


@dataclass
class TokenAlternative:
    token_id: int
    token: str
    raw_token: str
    prob: float

    def to_dict(self) -> dict:
        return {
            "token_id": self.token_id,
            "token": self.token,
            "raw_token": self.raw_token,
            "prob": _round(self.prob),
        }


@dataclass
class TokenStep:
    i: int
    token_id: int
    token: str
    raw_token: str
    prob: float
    logprob: float
    alternatives: list[TokenAlternative] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "i": self.i,
            "token_id": self.token_id,
            "token": self.token,
            "raw_token": self.raw_token,
            "prob": _round(self.prob),
            "logprob": _round(self.logprob),
            "alternatives": [a.to_dict() for a in self.alternatives],
        }


@dataclass
class RatingDistribution:
    raw: dict[str, float]
    normalized: dict[str, float]
    residual: float
    exact: bool
    candidate_first_tokens: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "raw": {k: _round(v) for k, v in self.raw.items()},
            "normalized": {k: _round(v) for k, v in self.normalized.items()},
            "residual": _round(self.residual),
            "exact": self.exact,
            "candidate_first_tokens": self.candidate_first_tokens,
        }


@dataclass
class Counterfactual:
    kind: str
    position: int
    forced_text: str
    forced_token_ids: list[int]
    generated_text: str
    parsed: dict
    steps: list[TokenStep] = field(default_factory=list)

    @property
    def mean_logprob(self) -> float | None:
        """Mean per-token logprob of the continuation generated *after* the
        forced prefix (the forced token itself has no logprob here: its
        probability is already visible in the main trace's rating_distribution).
        """
        if not self.steps:
            return None
        return sum(s.logprob for s in self.steps) / len(self.steps)

    def to_dict(self) -> dict:
        mean_logprob = self.mean_logprob
        return {
            "kind": self.kind,
            "position": self.position,
            "forced_text": self.forced_text,
            "forced_token_ids": list(self.forced_token_ids),
            "generated_text": self.generated_text,
            "parsed": self.parsed,
            "steps": [s.to_dict() for s in self.steps],
            "mean_logprob": _round(mean_logprob) if mean_logprob is not None else None,
        }


@dataclass
class DiagnosticsTrace:
    image: dict
    model: dict
    decoding: dict
    top_k: int
    prefill: str
    prompt_token_count: int
    generated_text: str
    parsed: dict
    rating_distribution: RatingDistribution | None
    steps: list[TokenStep] = field(default_factory=list)
    counterfactuals: list[Counterfactual] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)
    generated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": self.generated_at or utc_now(),
            "image": self.image,
            "model": self.model,
            "decoding": self.decoding,
            "top_k": self.top_k,
            "prefill": self.prefill,
            "prompt_token_count": self.prompt_token_count,
            "generated_text": self.generated_text,
            "parsed": self.parsed,
            "rating_distribution": (
                self.rating_distribution.to_dict() if self.rating_distribution else None
            ),
            "steps": [s.to_dict() for s in self.steps],
            "counterfactuals": [c.to_dict() for c in self.counterfactuals],
            "timings_s": {k: round(v, 3) for k, v in self.timings_s.items()},
        }


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def candidate_first_tokens(
    tokenizer, candidates: tuple[str, ...] = RATING_CANDIDATES
) -> dict[str, int]:
    return {c: tokenizer.encode(c, add_special_tokens=False)[0] for c in candidates}


def rating_distribution(
    tokenizer,
    step0_probs,
    candidates: tuple[str, ...] = RATING_CANDIDATES,
) -> RatingDistribution:
    """Probability mass on each rating, read off generation step 0.

    The prefill ends on `"` and tokenization is boundary-stable, so a
    candidate's bare first token id is the one the model would emit for it.
    `exact` records that those ids are distinct: if two candidates shared a
    first token the proxy would be ambiguous and the numbers would need exact
    multi-token scoring instead.
    """
    first_tokens = candidate_first_tokens(tokenizer, candidates)
    exact = len(set(first_tokens.values())) == len(first_tokens)
    if not exact:
        logger.warning(
            "Rating candidates share a first token (%s); "
            "the step-0 distribution is approximate",
            first_tokens,
        )

    raw = {c: float(step0_probs[tid]) for c, tid in first_tokens.items()}
    total = sum(raw.values())
    normalized = {c: (v / total if total > 0 else 0.0) for c, v in raw.items()}
    return RatingDistribution(
        raw=raw,
        normalized=normalized,
        residual=1.0 - total,
        exact=exact,
        candidate_first_tokens=first_tokens,
    )
