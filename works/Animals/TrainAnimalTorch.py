import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2
import os

# ============================
# CONFIGURACIÓN PARA LAPTOP
# ============================
torch.set_num_threads(2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
BATCH = 8
EPOCHS = 20
SAVE_PATH = "../models/animals/animal_cnn_torch.pth"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
# ============================
# DATASET
# ============================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder("../dataset/animals", transform=train_transform)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=2)

# ============================
# MODELO
# ============================
model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, len(dataset.classes))
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ============================
# ENTRENAMIENTO
# ============================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Checkpoint guardado: {SAVE_PATH}")

print("Entrenamiento terminado.")
print("Clases:", dataset.classes)
