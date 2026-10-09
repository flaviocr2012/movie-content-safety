"""
Load the fine-tuned movie-safety model from Hugging Face Hub.

Usage:
    python models/load_finetuned.py

Requires:
    pip install transformers torch
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "flaviocr2023/movie-safety-llama-3.1-8b"

def load_model():
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    return model, tokenizer

def classify(prompt: str, model, tokenizer) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    model, tokenizer = load_model()
    prompt = "Is the movie 'The Conjuring' safe for children aged 5-10?"
    print(classify(prompt, model, tokenizer))