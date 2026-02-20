# thanks chatgpt :)

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLD = 0.5

# ---- Model Loader ----
def load_model(weights_path):
    model = models.resnet101(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ---- Load 3 Task Models ----
models_dict = {
    1: load_model("task_1_weights.pth"),
    2: load_model("task_2_weights.pth"),
    3: load_model("task_3_weights.pth"),
}

# ---- Preprocessing ----
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---- Route ----
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    task = request.form.get("task_id")
    if task is None or not task.isdigit() or int(task) not in [1, 2, 3]:
        return jsonify({"error": "task_id must be 1, 2, or 3"}), 400

    task = int(task)
    model = models_dict[task]

    image_file = request.files["image"]
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    label = "COMPLETE" if prob > THRESHOLD else "INCOMPLETE"

    return jsonify({
        "task": task,
        "result": label
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
