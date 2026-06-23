import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Annotation:
    meta: list[str] = field(default_factory=list)
    content: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.meta and not self.content

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Annotation":
        return cls(**json.loads(path.read_text()))

    @classmethod
    def from_str(cls, data: str) -> "Annotation":
        return cls(**json.loads(data))

    @classmethod
    def from_dict(cls, data: dict) -> "Annotation":
        return cls(**data)


# section headers in the editor buffer
_CONTENT = "[content]"
_META = "[meta]"
# user types this on its own line to stop the loop
QUIT = "quit"


def annotation_from_ocr(text: str) -> Annotation:
    """VLM OUTPUT_SCHEMA json -> Annotation. Caller wraps in try/except."""
    d = json.loads(text)
    return Annotation(
        meta=[e["text"] for e in d.get("meta", [])],
        content=[e["text"] for e in d.get("content", [])],
    )


def render_buffer(name: str, seed: Annotation, i: int, count: int) -> str:
    """Editor buffer pre-filled with the VLM transcription for the user to correct."""
    lines = [
        f"# image: {name}  ({i}/{count}) - pre-filled by VLM, correct as needed",
        "# Lines under [content] = body text; under [meta] = margin/label text.",
        f"# Save & close to store. Empty = skip. A line '{QUIT}' or :cq = stop.",
        _CONTENT,
        *seed.content,
        _META,
        *seed.meta,
    ]
    return "\n".join(lines) + "\n"


def parse_buffer(text: str) -> tuple[Annotation, bool]:
    """Parse edited buffer -> (Annotation, quit). Lines route by [content]/[meta] header."""
    ann = Annotation()
    bucket = ann.content  # default before any header
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == QUIT:
            return ann, True
        if line == _CONTENT:
            bucket = ann.content
            continue
        if line == _META:
            bucket = ann.meta
            continue
        bucket.append(raw.rstrip())
    return ann, False


def _selfcheck():
    a = annotation_from_ocr(
        '{"meta":[{"text":"m1"}],"content":[{"text":"c1"},{"text":"c2"}]}'
    )
    assert a == Annotation(meta=["m1"], content=["c1", "c2"]), a

    buf = "# comment\n[content]\nhello\nworld\n[meta]\nmargin\n"
    ann, quit = parse_buffer(buf)
    assert not quit
    assert ann == Annotation(meta=["margin"], content=["hello", "world"]), ann

    # round-trip render -> parse
    ann2, _ = parse_buffer(render_buffer("x.jpg", ann, 0, 5))
    assert ann2 == ann, ann2

    # quit sentinel
    _, q = parse_buffer("[content]\nquit\n")
    assert q

    print("ok")


if __name__ == "__main__":
    _selfcheck()
