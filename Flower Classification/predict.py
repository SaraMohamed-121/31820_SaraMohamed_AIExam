import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torchvision import transforms, models
from PIL import Image
import json
from torchvision import datasets
import matplotlib.pyplot as plt
import random


DATA_DIR = r"data\dataset"               
CAT_TO_NAME_PATH = r"data\cat_to_name.json"
MODEL_PATH = r"models\best_model.pth"
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"))
class_to_idx = test_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

# LOAD cat_to_name.json
with open(CAT_TO_NAME_PATH, "r") as f:
        cat_to_name = json.load(f)       

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(cat_to_name)

model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features

model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(512, num_classes)
)

assert model.fc[-1].out_features == len(idx_to_class), "Mismatch in number of classes!"

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# IMAGE TRANSFORMS 
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# Path to test images
test_folder = os.path.join(DATA_DIR, "test")
test_images = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith((".jpg", ".png"))]

#  PREDICTION FUNCTION 
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probs, 3)

    results = []
    for i in range(3):
        class_idx = top_classes[0][i].item()
        class_label = idx_to_class[class_idx]
        class_name = cat_to_name[class_label]
        confidence = top_probs[0][i].item()
        results.append((class_name, confidence))

    return results

# Select 5 random images
sample_imgs = random.sample(test_images, 5)

plt.figure(figsize=(20,5))

for i, img_path in enumerate(sample_imgs):
    predictions = predict_image(img_path)
    img = Image.open(img_path).convert("RGB")
    
    plt.subplot(1,5,i+1)
    plt.imshow(img)

    title_text = "\n".join(
        [f"{name}: {conf*100:.1f}%" for name, conf in predictions]
    )

    plt.title(title_text)
    plt.axis('off')

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Save sample predictions
plt.tight_layout()
plt.savefig("outputs/sample_predictions.png")
plt.show()

