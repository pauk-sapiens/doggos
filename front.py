from flask import Flask, request, render_template
from PIL import Image
import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
import os
import random
import base64
import json
from io import BytesIO

app = Flask(__name__)

# 1. Load breed info & class names
with open('breed_info.json') as f:
    breed_info = json.load(f)

# We'll load the folders only to get the raw class keys
train_folder = "dataset/train"
raw_classes = datasets.ImageFolder(train_folder, transform=None).classes

# Build human‐readable names: split off the WordNet ID, replace "_" with " ", title‐case
classes = [raw.split('-', 1)[1].replace('_', ' ').title() for raw in raw_classes]

# 2. Inference setup
device = torch.device('cpu')  # using quantized TorchScript

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
inference_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean, std),
])

# Load the quantized TorchScript model
model = torch.jit.load("model_prod.pt", map_location=device)
model.eval()

# 3. Utility functions
def image_to_data_uri(img: Image.Image, fmt="JPEG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()

def predictor(img: Image.Image, topk=5):
    tensor = inference_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)  # log‐probs
        probs = torch.exp(out).cpu().squeeze()
        top_probs, top_idxs = torch.topk(probs, topk)
    topk_preds = {
        classes[idx]: f"{p.item()*100:.2f}%"
        for idx, p in zip(top_idxs, top_probs)
    }
    top1 = top_idxs[0].item()
    return classes[top1], topk_preds, raw_classes[top1]

def get_dog_image_data(raw_breed: str):
    folder = os.path.join(train_folder, raw_breed)
    if not os.path.isdir(folder):
        return None
    imgs = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not imgs:
        return None
    choice = random.choice(imgs)
    img = Image.open(os.path.join(folder, choice)).convert("RGB")
    return image_to_data_uri(img)

# 4. Flask route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    top_predictions = {}
    uploaded_uri = None
    example_uri = None
    description = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            img = Image.open(BytesIO(file.read())).convert("RGB")
            # predictor now also returns raw folder name
            pred_name, top_predictions, raw_breed = predictor(img)
            prediction = pred_name
            uploaded_uri = image_to_data_uri(img)
            example_uri = get_dog_image_data(raw_breed)
            description = breed_info.get(pred_name, "No info available.")
        else:
            prediction = "No image uploaded."

    return render_template("index.html",
        prediction=prediction,
        top_predictions=top_predictions,
        uploaded_image_data_uri=uploaded_uri,
        dog_image_data_uri=example_uri,
        breed_description=description
    )

if __name__ == "__main__":
    app.run(debug=True)
