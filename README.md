# AIF_GRADIO


## Table of Contents
- [Extracting image features](#Extracting-image-features)
- [AnnoyIndex](#AnnoyIndex)
- [Running project](#Running-project)


## Extracting image features
This project describes the application of Gradio with Docker compose to build a content-based recommendation system using movie posters. This is an existing database to download:

https://drive.google.com/file/d/1bv3XWCzT3H4dGAMEdM-pXIV2hF7vUzlj/view?usp=drive_link

PyTorch and torchvision are used to extract image features using the MobileNetV3 Small pre-trained model. 

### Images transformation

Images are transformed using the transforms.Compose class.Transforms applied include resizing images and converting them to tensors with transforms.ToTensor().

```python 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```
### Loading the pre-trained MobileNetV3 Small model

The MobileNetV3 Small model is loaded from torchvision.models with the parameter pretrained=True:

```python 
import torchvision.models as models
mobilenet = models.mobilenet_v3_small(pretrained=True)
```

### Building a new model

A new model is built using nn.Sequential. The layers of the MobileNetV3 Small model are included except for the last layer (avgpool), which is replaced by a flattening layer (nn.Flatten()):

```python 
import torch.nn as nn
model = nn.Sequential(
    *list(mobilenet.children())[:-1],  
    nn.Flatten() 
)
```

## AnnoyIndex

The purpose is to extract features from images using the MobileNetV3 Small model and store these features in a database. The database store image features and their paths:

https://drive.google.com/file/d/1oOw9lzGXlEqrr2ggigbwhR4Pfek4kOlN/view?usp=drive_link

These extracted features can then be used to compute the index based on these image representations:

https://drive.google.com/file/d/1Pzy5arNy-LDY85IbrEC4WSV0e8pZ3JnT/view?usp=drive_link

## Running project

The first step in this project is running the Docker compose:

```bash
docker-compose up
```

Then open `localhost:7860`, load an image and see the 5 most similar images based on the index and characteristics of all the images in the database. 





