# AIF_GRADIO


## Table of Contents
- [Extracting image features](#Extracting-image-features)
- [AnnoyIndex](#AnnoyIndex)
- [Generating similar images based on features](#Generating-similar-images-based-on-features)


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




