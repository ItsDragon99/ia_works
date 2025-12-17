import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings

# --- CONFIGURACIÓN ---
dataset_path = "./tutor_v2.jsonl"
model_name = "Qwen/Qwen2.5-3B-Instruct"
lora_path = "./lora-tutor-v2"
lora_model_name = "qwen_tutor_v1"

device = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=UserWarning)

print(f"Tutor  en {device}...")

# 1. Cargar Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 2. Cargar Modelo Base
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto"
)

# 3. Cargar LoRA (modelo entrenado)
model = PeftModel.from_pretrained(base_model, lora_model_name)
model.to(device)
model.eval()



# -------------------------------------------------
# FUNCIÓN CLAVE: LIMPIEZA Y CORTE DE RESPUESTA
# -------------------------------------------------
def limpiar_respuesta(texto):
    texto = texto.strip()
    if "\n" in texto:
        texto = texto.split("\n")[0]
    if "." in texto:
        partes = texto.split(".")
        if len(partes) > 1:
            texto = ".".join(partes[:2]).strip() + "."
        else:
            texto = partes[0].strip() + "."

    return texto


def generar_respuesta(pregunta):
    prompt = f"### Pregunta:\n{pregunta}\n\n### Respuesta esperada:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    texto = tokenizer.decode(out[0], skip_special_tokens=True)
    respuesta = texto.split("### Respuesta esperada:")[-1].strip()
    respuesta = limpiar_respuesta(respuesta)

    return respuesta


# --- BUCLE INTERACTIVO ---
while True:
    try:
        usuario = input(">> Yo merengues: ")
        if usuario.lower() in ["salir", "exit", "quit"]:
            print("Sale apa... Cuidate!")
            break
        if not usuario.strip():
            continue
        print("Perame carnal ando pensando...", end="\r")
        respuesta = generar_respuesta(usuario)

        print(f" Tutor: {respuesta}\n")

    except KeyboardInterrupt:
        print("\nSaliendo...")
        break
    except Exception as e:
        print(f"Error crítico: {e}")