import os
import csv
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

IMG_DIR = "static/products/images"
CSV_IN = "static/products/products.csv"
EMB_OUT = "static/products/product_embeddings.npy"

# Load model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

df = pd.read_csv(CSV_IN)
embeddings = []

with torch.no_grad():
    for fname in tqdm(df["filename"]):
        path = os.path.join(IMG_DIR, fname)
        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0)
        emb = model(img).numpy().flatten()
        embeddings.append(emb)

np.save(EMB_OUT, np.array(embeddings))
print("Embeddings saved:", EMB_OUT)
