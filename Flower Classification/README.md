# **Flower Classification **

## **Task Description**

This project implements a **deep learning image classification pipeline** to classify flower images into their correct species. The model uses **ResNet50 pretrained on ImageNet** and fine-tunes the final fully connected layers for multi-class classification.

## **Dataset**
- [Oxford 102 Flowers Dataset](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset)
- Folder structure:
    train/ | valid/ | test/

## **Project Structure**

```
flower_classification/
├── train.py                     # Training script with augmentation and accuracy/loss plots
├── evaluate.py                  # Evaluation script with classification report and confusion matrix
├── predict.py                   # Single image inference with sample predictions
├── data/
│   └── dataset/                 # Dataset folder
│       ├── train/
│       ├── valid/
│       └── test/
├── models/
│   └── best_model.pth           # Saved best model weights
├── outputs/
│   ├── training_curves.png      # Training & validation loss
│   ├── training_accuracy.png    # Training & validation accuracy
│   ├── confusion_matrix.png     # Confusion matrix
│   └── sample_predictions.png   # 5 sample predictions with confidence
├── cat_to_name.json             # Class index to name mapping
├── README.md
└── requirements.txt
```

---

## **How to Run**

1. Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
python train.py
```

3. Evaluate the model:

```bash
python evaluate.py
```

4. Predict a single image:

```bash
python predict.py
```

5. All outputs (plots, confusion matrix, sample predictions) will be saved in the `outputs/` folder.

## **Environment**
- Tested on Google Colab with GPU (Tesla T4)
- Compatible with CPU, but training will be slower

