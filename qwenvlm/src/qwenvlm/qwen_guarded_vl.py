import logging
import os
import re
from pathlib import Path

import pandas as pd

from qwenvlm.qwen_3_guard import qwen_guard
from qwenvlm.qwen_3_vl import qwen_use

logger = logging.getLogger(__name__)


def process_and_guard_image_folder(folder_path, prompt, results_path, tokens=128):
    df = pd.DataFrame(
        columns=pd.Index(
            [
                "image_name",
                "safety_label",
                "safety_type",
                "color",
                "generated_description",
            ]
        )
    )

    folder_path = Path(folder_path)
    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    for image_file in os.listdir(folder_path):
        if image_file.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
        ):
            image_path = folder_path / image_file
            logger.info(f"Generating description for image: {image_path}")

            description = qwen_use(str(image_path), prompt, tokens)

            # Normalize and strip common Markdown artifacts (Thanks Claude)

            description = description or ""
            # remove fenced code blocks ```...```
            description = re.sub(r"```.*?```", " ", description, flags=re.DOTALL)
            # replace newlines with space
            description = description.replace("\r", " ").replace("\n", " ")
            # inline code `code` -> code
            description = re.sub(r"`([^`]*)`", r"\1", description)
            # images ![alt](url) -> alt
            description = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"\1", description)
            # links [text](url) -> text
            description = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", description)
            # headings like ## Heading -> remove leading hashes
            description = re.sub(r"(^|\s)#{1,6}\s*", " ", description)
            # emphasis and bold markers *text* **text** _text_ -> text
            description = re.sub(r"(\*\*|\*|__|_)(.*?)\1", r"\2", description)
            # remove HTML tags
            description = re.sub(r"<[^>]+>", " ", description)
            # remove stray backticks, tildes, and bullets
            description = description.replace("`", "").replace("~", "")
            description = re.sub(r"(^|\s)[\-\*\+]\s+", " ", description)
            # collapse multiple spaces and trim
            description = re.sub(r"\s+", " ", description).strip()

            logger.info(f"Evaluating safety for image: {image_path}")
            safety_label, categories = qwen_guard(description)

            if safety_label == "Safe":
                row = [image_file, "Safe", "", "🟢", description]
            elif safety_label == "Controversial":
                row = [
                    image_file,
                    "Controversial",
                    ", ".join(categories),
                    "🟡",
                    description,
                ]
            else:
                row = [image_file, "Unsafe", ", ".join(categories), "🔴", description]

            df.loc[len(df)] = row

    for row in df.itertuples():
        logger.info(
            f"{row.color} {row.image_name} - {row.safety_label} ({row.safety_type}) : {row.generated_description}"  # ty:ignore
        )

    df.to_csv(results_path / "image_safety_report.csv", index=False)
