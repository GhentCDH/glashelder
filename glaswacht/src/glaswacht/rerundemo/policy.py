"""Static explainer panels: the pipeline, the policy and the exact prompt."""

import rerun as rr

from glaswacht.models.base import TransformersSafetyModel
from glaswacht.models.utils import get_prompt
from glaswacht.rerundemo.style import MODEL_FACTS, md, title

PIPELINE = """\
# GlasWacht

**Is this glass-plate scan safe to publish?**
Three vision-language models rate every scan independently against the same written policy. A conservative vote combines them.

```
 scan ──► LlavaGuard 7B ─┐
      ├─► Qwen3-VL 8B  ──┼─► majority vote ─► Safe / Potentially Problematic / Unsafe
      └─► Pixtral 12B  ─┘        (ties go to the more cautious rating)
```

| step | what happens |
|---|---|
| 1 | The policy (`taxonomy.txt`) is the prompt. It is identical for every model. |
| 2 | The reply is **pre-filled** with `{"rating": "`, so the model only completes the JSON. |
| 3 | The JSON is parsed. Unparseable output becomes **Unknown**. |
| 4 | Votes: clear majority wins. On a tie: Unsafe > Potentially Problematic > Safe > Unknown. |
| 5 | Humans validate in the web UI. |

Models run **one after another** (each is loaded, used on every image, then unloaded) because they do not fit in 24 GB together.
"""


def models_table(models: tuple[str, ...]) -> str:
    rows = ["| model | weights | quantisation | decoding |", "|---|---|---|---|"]
    for m in models:
        w, q, d = MODEL_FACTS.get(m, ("?", "?", "?"))
        rows.append(f"| **{title(m)}** | `{w}` | {q} | {d} |")
    return "\n".join(rows)


def taxonomy_md() -> str:
    """The policy text, with its categories turned into headings."""
    out = []
    for line in get_prompt().splitlines():
        if line[:2] in ("O1", "O2", "O3", "O4") and line[2:3] == ":":
            out.append(f"\n## {line}")
        elif line.strip() in ("Should not:", "Should Not:", "Can:"):
            out.append(f"\n**{line.strip()}**\n")
        else:
            out.append(line)
    return "\n".join(out)


def log_static(models: tuple[str, ...]) -> None:
    rr.log("policy/pipeline", md(PIPELINE), static=True)
    rr.log("policy/models", md("## Models\n\n" + models_table(models)), static=True)
    rr.log("policy/taxonomy", md(taxonomy_md()), static=True)
    prefill = TransformersSafetyModel.JSON_PREFILL
    rr.log(
        "policy/prompt",
        md(
            "## What the model is actually sent\n\n"
            "The full policy above, then the reply begins already as:\n\n"
            f"```json\n{prefill}\n```\n"
            "The first token the model writes is therefore the **rating itself**, which is "
            "why a single probability read-out at that position is meaningful.\n\n"
            "> **O4 note**: the policy says *Ethnographic content is always rated Potentially "
            "Problematic*. That rule lives in the prompt only, nothing enforces it in code."
        ),
        static=True,
    )
