# test.py

# Visualization and data handling libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# PyTorch and torchvision libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Other utilities
from PIL import Image
import requests
from io import BytesIO

# Device setup
my_gpu = torch.cuda.is_available()
device = torch.device('cuda' if my_gpu else 'cpu')
num_classes = 70  # Adjust this to match your training

# Define deterministic transforms for inference (remove random augmentation)
inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # Do not include RandomHorizontalFlip for inference
])

# Load train dataset to access class names (assumes your "train" folder exists)
trainset = torchvision.datasets.ImageFolder("train", transform=inference_transforms)

# Download an example image
url = "https://cdn.shopify.com/s/files/1/0765/3946/1913/files/Borzoi-dog-white_600x600.webp?v=1735522657"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB")

# Instantiate the model architecture
# Note: The 'pretrained' argument is deprecated in favor of 'weights'
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier (fully connected) layer with your custom classifier
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes),
    nn.LogSoftmax(dim=1)
)

# Load the saved state dictionary (model1.pt)
state_dict = torch.load("model1.pt", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Move the model to the selected device and set it to evaluation mode
model = model.to(device)
model.eval()


def predictor(img, n=5):
    """
    Predicts the class of an input image and returns the top n predictions.

    Args:
        img (PIL.Image): Input image.
        n (int): Number of top probabilities to return.

    Returns:
        pred (str): The predicted class label.
        top_preds (dict): Dictionary of top n predictions with their probabilities.
    """
    # Apply the deterministic inference transforms
    img_tensor = inference_transforms(img)
    # Add batch dimension and move to device
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Forward pass through the model
    with torch.no_grad():
        output = model(img_tensor)

    # Get the top prediction
    top_class = output.data.max(1, keepdim=True)[1].cpu().numpy()
    pred = int(np.squeeze(top_class))
    pred_label = trainset.classes[pred]

    # Get probabilities and top n predictions
    probabilities = torch.exp(output.data.cpu()).squeeze()
    top_probs, top_indices = torch.topk(probabilities, n)
    top_preds = {
        trainset.classes[idx]: f"{round(float(prob) * 100, 2)}%"
        for idx, prob in zip(top_indices, top_probs)
    }

    return pred_label, top_preds


# Run prediction on the downloaded image
my_prediction, top_predictions = predictor(img, n=5)

print("Predicted Class:", my_prediction)
print("Top Predictions:", top_predictions)
