from qwenvlm.qwen_3_guard import qwen_guard
from qwenvlm.qwen_3_vl import qwen_use
import pandas as pd
import os 

import logging

logger = logging.getLogger(__name__)


def process_and_guard_image_folder(folder_path, prompt, results_path):
    from pathlib import Path
    from PIL import Image

    df = pd.DataFrame(columns=["image_name", "safety_label", "safety_type", "color" , "generated_description"])
    
    folder_path = Path(folder_path)
    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    for image_file in os.listdir(folder_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = folder_path / image_file
            logger.info(f"Generating description for image: {image_path}")

            description = qwen_use(str(image_path), prompt)
            description = description.strip().replace('\n', ' ')

            logger.info(f"Evaluating safety for image: {image_path}")
            safety_label, categories = qwen_guard(description)

            if safety_label == "Safe":
                row = [image_file, "Safe", "", "🟢", description ]
            elif safety_label == "Controversial":
                row = [image_file, "Controversial", ", ".join(categories), "🟠", description]
            else:
                row = [image_file, "Unsafe", ", ".join(categories), "🔴", description]
            
            df.loc[len(df)] = row

    for row in df.itertuples():
        logger.info(f"{row.color} {row.image_name} - {row.safety_label} ({row.safety_type}) : {row.generated_description}")
    
    df.to_csv(results_path / "image_safety_report.csv", index=False)
