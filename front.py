from flask import Flask, request, render_template
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import random
import os
import base64
import json
from io import BytesIO

app = Flask(__name__)

# Load breed info JSON file
with open('breed_info.json') as f:
    breed_info = json.load(f)

# Device and model configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 70

# Define deterministic transforms for inference
inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Load the training dataset to access class names (ensure your "train" folder exists)
trainset = torchvision.datasets.ImageFolder("train", transform=inference_transforms)

# Instantiate the ResNet18 model with a custom classifier
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False

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
model = model.to(device)
model.eval()  # Set model to evaluation mode


def image_to_data_uri(img, format="JPEG"):
    """Convert a PIL image to a base64-encoded data URI."""
    buffered = BytesIO()
    img.save(buffered, format=format)
    encoded_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/{format.lower()};base64,{encoded_img}"


def predictor(img, n=5):
    """Transforms the image, performs a forward pass, and returns the top prediction and top n probabilities."""
    img_tensor = inference_transforms(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    # Get top prediction
    top_class = output.data.max(1, keepdim=True)[1].cpu().numpy()
    pred = int(top_class.squeeze())
    pred_label = trainset.classes[pred]

    # Get top n probabilities
    probabilities = torch.exp(output.data.cpu()).squeeze()
    top_probs, top_indices = torch.topk(probabilities, n)
    top_preds = {
        trainset.classes[idx]: f"{round(float(prob) * 100, 2)}%"
        for idx, prob in zip(top_indices, top_probs)
    }

    return pred_label, top_preds


def get_dog_image_data(breed):
    """Fetch a random dog image from the local 'train' directory for a given breed and return it as a data URI."""
    try:
        breed_folder = os.path.join("train", breed)  # Path to the breed folder in the 'train' directory

        # Check if the breed folder exists
        if not os.path.exists(breed_folder):
            raise FileNotFoundError(f"Breed folder for {breed} not found.")

        # List all jpg images in the breed folder
        breed_images = [f for f in os.listdir(breed_folder) if f.endswith('.jpg')]

        # If no images are found in the breed folder, raise an error
        if not breed_images:
            raise FileNotFoundError(f"No images found for breed {breed}.")

        # Pick a random image from the list of images
        random_image = random.choice(breed_images)
        image_path = os.path.join(breed_folder, random_image)

        # Open the image and convert it to a data URI
        img = Image.open(image_path).convert("RGB")
        return image_to_data_uri(img)

    except (FileNotFoundError, Exception) as e:
        print(f"Error: {e}")
        # In case of error, try to return a default image from a known path
        try:
            default_path = "static/default.jpg"
            img = Image.open(default_path).convert("RGB")
            return image_to_data_uri(img)
        except Exception as default_e:
            print(f"Default image error: {default_e}")
            return None


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    top_predictions = {}
    dog_image_data_uri = None
    uploaded_image_data_uri = None
    breed_description = None

    if request.method == "POST":
        if "file" not in request.files:
            prediction = "No file part in the request."
        else:
            file = request.files["file"]
            if file.filename == "":
                prediction = "No file selected."
            else:
                try:
                    # Read file bytes into memory
                    file_bytes = file.read()
                    # Create a PIL image from the bytes
                    uploaded_image = Image.open(BytesIO(file_bytes)).convert("RGB")

                    # Process the image for prediction
                    prediction, top_predictions = predictor(uploaded_image, n=5)

                    # Convert the uploaded image to a data URI for in-memory display
                    uploaded_image_data_uri = image_to_data_uri(uploaded_image)

                    # Get a corresponding dog image (converted to lowercase to match folder name)
                    breed = prediction.lower()
                    dog_image_data_uri = get_dog_image_data(breed)

                    # Lookup breed description from JSON; default message if not found
                    breed_description = breed_info.get(prediction, "No information available for this breed.")

                except Exception as e:
                    prediction = f"Error processing image: {e}"

    return render_template("index.html",
                           prediction=prediction,
                           top_predictions=top_predictions,
                           dog_image_data_uri=dog_image_data_uri,
                           uploaded_image_data_uri=uploaded_image_data_uri,
                           breed_description=breed_description)


if __name__ == "__main__":
    app.run(debug=True)
