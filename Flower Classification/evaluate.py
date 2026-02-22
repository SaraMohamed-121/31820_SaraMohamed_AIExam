import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

from train import DEVICE, MODEL_SAVE_PATH, val_test_transforms, DATA_DIR

# LOAD cat_to_name.json 
with open(os.path.join(DATA_DIR, "cat_to_name.json"), "r") as f:
    cat_to_name = json.load(f)

# LOAD DATASET
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(test_dataset.classes)

# LOAD MODEL 
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# PREDICTIONS 
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# CLASSIFICATION REPORT 
class_names = [cat_to_name[str(idx)] for idx in range(num_classes)]
print(classification_report(all_labels, all_preds, target_names=class_names))

#  CONFUSION MATRIX 
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(15,12))
sns.heatmap(cm, annot=False, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig("outputs/confusion_matrix.png")
plt.show()