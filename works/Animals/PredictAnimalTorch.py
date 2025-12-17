import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import mobilenet_v2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
MODEL_PATH = "../models/animals/animal_cnn_torch.pth"
CLASSES = ["canis lupus familiaris", "felis catus", "chelonia", "formicidae", "coccinellidae"]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

model = mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(1280, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

image_path = "../dataset/animals/test/mariquita.jpg"

img_pil = Image.open(image_path).convert("RGB")
img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(img_tensor)
    _, pred = torch.max(output, 1)
    print(_)
    
    label = CLASSES[pred.item()]

print(f"Predicción: {label}")
