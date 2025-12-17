from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

dataset_path = "./tutor_v2.jsonl"
model_name = "Qwen/Qwen2.5-3B-Instruct"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
lora_path = "./lora-tutor-v2"
# lora_path = "./lora-tutor-llama-v1"

tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="./offload",   # carpeta para offloading
    torch_dtype=torch.float16     # ayuda a ahorrar VRAM
)

model = PeftModel.from_pretrained(
    base_model,
    lora_path,
    device_map="auto",
    offload_folder="./offload"
)

prompt = "Define 'complejidad temporal'"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))