import os
import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    DATA_DIR = r"data\dataset"               
    CAT_TO_NAME_PATH = r"data\cat_to_name.json"
    MODEL_PATH = r"models\best_model.pth"
    OUTPUT_DIR = r"outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Class Names 
    with open(CAT_TO_NAME_PATH, "r") as f:
        cat_to_name = json.load(f)        

    # Setup Device 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    #  Data Transforms 
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    #  Load Dataset  
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR,"valid"), transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Mapping from index to class name
    class_to_idx = val_dataset.class_to_idx        
    idx_to_class = {v: k for k, v in class_to_idx.items()}  

    # Load Model
    num_classes =len(val_dataset.classes)
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(512, num_classes)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")

    # Evaluation
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to Names
    all_preds_names = [cat_to_name[idx_to_class[i]] for i in all_preds]
    all_labels_names = [cat_to_name[idx_to_class[i]] for i in all_labels]

    # Classification Report
    print("Classification Report")
    report =classification_report(all_labels_names, all_preds_names)
    print(report)

    # Confusion Matrix 
    cm = confusion_matrix(all_labels_names,all_preds_names)
    plt.figure(figsize=(20,20))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()
    print(f"Confusion matrix saved to {cm_path}")