# Flower Classification

## Overview

This project implements a **multi-class image classification system** for flower species recognition using **Transfer Learning with ResNet50**.

The model is trained on a public Kaggle dataset and optimized to achieve high accuracy while maintaining efficient training performance.

---

## Key Features

* Transfer Learning using pretrained **ResNet50**
* Data Augmentation pipeline (rotation, flip, color jitter, random crop)
* Train / Validation / Test split (70/15/15)
* Optimized training with Adam + Learning Rate Scheduler
* Evaluation using:

## Project Structure

```
flower_classification/
├── train.py
├── evaluate.py
├── predict.py
├── data/
│   ├── dataset/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── cat_to_name.json
├── models/
│   └── best_model.pth
├── outputs/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
├── requirements.txt
└── README.md
```

---

## Performance

* Achieved high validation accuracy
* Target: 85%+ test accuracy (depending on dataset used)

---

## How to Run

### Install dependencies

```
pip install -r requirements.txt
```

### Train model

```
python train.py
```

### Evaluate model

```
python evaluate.py
```

### Predict single image

```
python predict.py --image path_to_image.jpg
```


