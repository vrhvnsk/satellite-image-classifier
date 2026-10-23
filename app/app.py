# Ivan Novikov, K62405902

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from shiny.express import ui, render, input


# main body / configgs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path(__file__).parent / "assets" / "weights" / "best_model.pth"

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

mean = [0.3454906814311202, 0.3821572564112961, 0.4091499240826845]
std = [0.09406774456234863, 0.06586975807942382, 0.05506130168200451]

transform = T.Compose([T.Resize((64, 64)), T.ToTensor(), T.Normalize(mean, std)])

#model import
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class ResNetSmall(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.layer1 = BasicBlock(3, 32)
        self.layer2 = BasicBlock(32, 64, stride=2)
        self.layer3 = BasicBlock(64, 128, stride=2)
        self.layer4 = BasicBlock(128, 256, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# model loading (nb!! keep away from uis!!)
model = ResNetSmall(num_classes=len(CLASS_NAMES))

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
_ = model.load_state_dict(state_dict)
_ = model.eval()

model = model.to(DEVICE)


#helpers

def get_image():
    file = input.file()
    if file is None: return None

    return Image.open(file[0]["datapath"]).convert("RGB")


def predict(img):
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))

    return CLASS_NAMES[idx], float(probs[idx]), probs


# and ui itself

ui.page_opts(
    title="Satellite Image Classifier",
    fillable=True
)

with ui.layout_sidebar():

    with ui.sidebar():
        ui.h3("Upload Image")
        ui.input_file("file", "Choose satellite image")

    with ui.layout_column_wrap(fill=True):

        #image display
        with ui.card():
            ui.card_header("Uploaded Image")

            @render.image
            def image_preview():
                img = get_image()
                if img is None: return None

                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                img.save(tmp.name)

                return {"src": tmp.name,
                        "width": "300px"}

        # prediction making
        with ui.card():
            ui.card_header("Prediction")

            @render.text
            def prediction():
                img = get_image()
                if img is None: return "Upload an image."
                cls, conf, _ = predict(img)
                return f"{cls}\nConfidence: {conf:.2%}"

        # certainty / prob table
        with ui.card():
            ui.card_header("Class Probabilities")

            @render.table
            def probability_table():
                img = get_image()
                if img is None: return pd.DataFrame()
                
                _, _, probs = predict(img)
                df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": probs}).sort_values("Probability", ascending=False)

                return df