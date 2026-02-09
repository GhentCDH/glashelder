from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image
import torch

selected_model = "AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf"

model = LlavaOnevisionForConditionalGeneration.from_pretrained(selected_model)
model.to('cuda:0')
model.eval()
processor = AutoProcessor.from_pretrained(selected_model)


policy_v1 = """Provide a safety assessment for the content provided by the user. Determine compliance or violation against our safety policy by reviewing the following policy categories:
O1: Hate, Humiliation, Harassment: 
Should not:
- Contain content that expresses hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Contain content that encourages discrimination or harassment against people based on those characteristics.
Can:
- Contain educational content that provides information on discrimination, humiliation, and harassment.
- Discuss historical events involving hate and harassment.

O2: Violence: 
Should not:
- Contain blood or gore.
Can:
- Contain fictional content (such as skeletons, that depicts violence or .

O3: Nudity Content: 
Should not:
- Contain visible genitalia (e.g., exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content. 
Can:
- Contain artistic nudity if the content does not include visible genitalia.
- Contain educational content and general information on the human body.
- Provide guidance on topics related to sexual health.

Assessment Steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
2. Assign a policy "category". If none of the policy categories are applicable, select "NA: None applying".
3. Provide a "rationale". Describe the user content and justify why it is considered safe or unsafe, referring to the specific policy category and its associated guidelines to illustrate any violations or compliance.
To provide your assessment use the following json template:
{
    "rating": str<"Unsafe"|"Safe">,
    "category": str<"O1: Hate, Humiliation, Harassment"|"O2: Violence"|"O3: Nudity Content"|"NA: None applying">,
    "rationale": str,
}
"""
policy = policy_v1
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": policy},
            ],
    },
]

text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image = Image.open("/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set-crop/0z70w1hx0d_tiff.jpg")

inputs = processor(text=text_prompt, images=image, return_tensors="pt")
inputs = {k: v.to('cuda:0') for k, v in inputs.items()}

# Generate
hyperparameters = {
    "max_new_tokens": 200,
    "do_sample": True,
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 50,
    "num_beams": 2,
    "use_cache": True,
}

with torch.inference_mode():
    output = model.generate(**inputs, **hyperparameters)

output_text = processor.decode(output[0], skip_special_tokens=True)
print(output_text)

