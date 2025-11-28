from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import cv2 as cv
import gc
import base64

class qwen_inference:

    def __init__(self, prompt, model_name="Qwen/Qwen2-VL-7B-Instruct", min_pixels=256 * 28 * 28, max_pixels=1028 * 28 * 28) -> None:
        self.prompt = prompt
        self.model_name = model_name
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name, min_pixels=min_pixels, max_pixels=max_pixels)

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        del self.model
        del self.processor
        torch.cuda.empty_cache()
        gc.collect()

    def __message_constructor(self, image):
        self.message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image;base64,{image}",
                    },
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

    def __qwen_call(self, store_output=False, tokens=32):
        with torch.no_grad():
            text = self.processor.apply_chat_template(
                self.message, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(self.message)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to("cuda")
            generated_ids = self.model.generate(**inputs, max_new_tokens=tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return str(output_text).strip("[]'")

    def __image_processing(self, image_path):
        image = cv.imread(image_path)
        top, bottom, left, right = 10, 10, 10, 10
        padded_image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=[255, 255, 255])
        _, buffer = cv.imencode('.png', padded_image)
        image = base64.b64encode(buffer).decode('utf-8')
        self.__message_constructor(image)

    def run_inference(self, image_path, store_output=False, tokens=32):
        self.__image_processing(image_path)
        output_text = self.__qwen_call(store_output, tokens)
        return output_text