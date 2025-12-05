from transformers import AutoModelForCausalLM, AutoTokenizer
import re


def qwen_guard(input):
    model_name = "Qwen/Qwen3Guard-Gen-8B"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    def extract_label_and_categories(content):
        safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
        category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|None)"
        safe_label_match = re.search(safe_pattern, content)
        label = safe_label_match.group(1) if safe_label_match else None
        categories = re.findall(category_pattern, content)
        return label, categories

    # for prompt moderation
    messages = [
        {"role": "user", "content": input}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    label, categories = extract_label_and_categories(content)
    return label, categories