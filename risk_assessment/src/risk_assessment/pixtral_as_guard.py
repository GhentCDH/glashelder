from scipy.stats import mode
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import time
import os


# Load model
model_id = "SeanScripts/pixtral-12b-nf4"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    use_safetensors=True,
    device_map="cuda:0"
)
# Load tokenizer
processor = AutoProcessor.from_pretrained(model_id)

# Caption a local image
IMG_URLS = [Image.open("/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set-crop/0z70w1hx0d_tiff.jpg").convert("RGB")]
PROMPT = "<s>[INST]Provide a safety assessment for the content provided by the user:\n[IMG][/INST]"

inputs = processor(images=IMG_URLS, text=PROMPT, return_tensors="pt").to("cuda")
prompt_tokens = len(inputs['input_ids'][0])
print(f"Prompt tokens: {prompt_tokens}")

t0 = time.time()
generate_ids = model.generate(**inputs, max_new_tokens=512)
t1 = time.time()
total_time = t1 - t0
generated_tokens = len(generate_ids[0]) - prompt_tokens
time_per_token = generated_tokens/total_time
print(f"Generated {generated_tokens} tokens in {total_time:.3f} s ({time_per_token:.3f} tok/s)")

output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)