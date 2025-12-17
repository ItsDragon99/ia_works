import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
dataset_path = "./tutor_v2.jsonl"
model_name = "Qwen/Qwen2.5-3B-Instruct"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
lora_path = "./lora-tutor-v2"
torch.cuda.empty_cache()

print("1. Cargando modelo y tokenizer...")

new_model_name = "qwen_tutor_v1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.gradient_checkpointing_enable()
model.config.use_cache = False
dataset = load_dataset(
    "json",
    data_files = dataset_path,
    split="train"
)


def formatting_prompts_func(example):
    prompt = example.get("instruction") or example.get("prompt")
    response = example.get("response") or example.get("answer")
    if prompt is None or response is None:
        raise KeyError(f"Expected instruction/prompt and response/answer keys, found {list(example.keys())}")
    def fmt(p, r):
        return (
            f"### Pregunta:\n{p}\n\n"
            f"### Respuesta esperada:\n{r}{tokenizer.eos_token}"
        )
    if isinstance(prompt, list):  # batched case
        return [fmt(p, r) for p, r in zip(prompt, response)]
    return fmt(prompt, response)


# data_collator = DataCollatorForCompletionOnlyLM(
#     response_template="### Respuesta esperada:\n",
#     tokenizer=tokenizer
# )

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,   
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="./resultados_qwen",
    num_train_epochs=6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,   
    bf16=True,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    optim="adamw_torch",
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,  # if this errors, use tokenizer=tokenizer instead
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    peft_config=peft_config,
    args=training_args,
)
print("2. Entrenando...")
trainer.train()

print("3. Guardando modelo...")
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)

print(" Entrenamiento terminado")