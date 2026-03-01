# backend/models/feature_extractor.py
import cv2, numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def img_to_embedding(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (224,224))
    x = img_to_array(crop_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    emb = resnet.predict(x)[0]
    emb = emb / np.linalg.norm(emb)
    return emb
