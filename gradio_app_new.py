from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from annoy import AnnoyIndex
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import os 
import gradio as gr
import requests
import random
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

df = pd.read_csv('df4.csv')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
import torchvision.models as models
mobilenet = models.mobilenet_v3_small(pretrained=True)
import torch.nn as nn
model = nn.Sequential(
*list(mobilenet.children())[:-1],  # Extract features up to avgpool layer
nn.Flatten()  # Flatten the output
)


def extract_features_from_image(image):
    image =  Image.fromarray((image * 255).astype('uint8'))
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        features = model(image).cpu().numpy().flatten()
    return features
def process_image(image):
    vector = extract_features_from_image(image)
    response = requests.post('http://annoy-db:5000/reco', json={'vector': vector.tolist()})
    if response.status_code == 200:
        indices = response.json()
        # Retrieve paths for the indices
        paths = df['path'].iloc[indices].tolist()

        # Get the paths of the most similar films
        fig, axs = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5))
        for i, path in enumerate(paths):
            img = Image.open(path)
            axs[i].imshow(img)
            axs[i].axis('off')
        return fig
    else:
        return "Error in API request"

iface = gr.Interface(fn=process_image, inputs="image", outputs="plot")
iface.launch(server_name="0.0.0.0") # the server will be accessible externally under this address

