from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
from PIL import Image
import cv2 as cv


class pixtral_inference:
    
    # This class is used to run inference on the Pixtral model
    # It is a 4-bit quantized version of the Pixtral 12B model, so that it can run on a 4090 GPU

    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
        )

        model_id = "mistral-community/pixtral-12b"
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id,
                                                            device_map='cuda',
                                                            quantization_config=quantization_config)
        self.processor = AutoProcessor.from_pretrained(model_id)


    def __image_processing(self, image_path):
        image = cv.imread(image_path)
        top, bottom, left, right = 10, 10, 10, 10
        padded_image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=[255, 255, 255])
        pil_image = Image.fromarray(padded_image)
        return pil_image


    def run_inference(self, image_path, prompt):
        image = self.__image_processing(image_path)

        # Prepare the prompt
        prompt = f"<s>[INST]{prompt}:\n[IMG][/INST]"

        # Process inputs
        inputs = self.processor(images=image,text=prompt,return_tensors="pt").to("cuda")

        # Generate output
        outputs = self.model.generate(**inputs,max_new_tokens=1028,do_sample=True,temperature=0.01,top_p=0.9)

        # Decode and print the response
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return generated_text