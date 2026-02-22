import torch
from torchvision import transforms, models
from PIL import Image
import os
import json
from train import DEVICE, MODEL_SAVE_PATH, DATA_DIR

# LOAD cat_to_name.json
with open(os.path.join(DATA_DIR, "cat_to_name.json"), "r") as f:
    cat_to_name = json.load(f)

# LOAD MODEL 
num_classes = len(cat_to_name)
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# IMAGE TRANSFORMS 
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

#  PREDICTION FUNCTION 
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    class_name = cat_to_name[str(pred.item())]
    return class_name, conf.item()

# EXAMPLE USAGE 
if __name__ == "__main__":
    image_path = "dataset/test/1/image_06741.jpg" 
    pred_class, confidence = predict_image(image_path)
    print(f"Predicted: {pred_class} ({confidence*100:.2f}%)")