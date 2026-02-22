# **YOLO-Based Image Manipulation**

## **Task Description**

This project implements a **YOLO-based image manipulation pipeline** that performs four operations using a single YOLO segmentation model:

1. **Object Removal** – Remove selected objects from an image.
2. **Object Replacement** – Replace a detected object with another image seamlessly.
3. **Background Removal** – Remove the background and produce a transparent PNG.
4. **Background Replacement** – Replace the background while preserving the foreground objects.

The project uses **YOLOv8-seg**, **OpenCV**, and precomputed segmentation masks to ensure efficient and accurate results.

---

## **Project Structure**

```
yolo_image_manipulation/
├── main.py                     # Main script calling all 4 functions
├── models/
│   └── yolov8n-seg.pt          # YOLO segmentation model
├── inputs/
│   ├── test_image.jpg
│   ├── replacement_object.png
│   └── new_background.jpg
├── outputs/
│   ├── object_removed.jpg
│   ├── object_replaced.jpg
│   ├── bg_removed.png
│   └── bg_replaced.jpg
├── object_removal.py
├── object_replacement.py
├── background_removal.py
├── background_replacement.py
└── requirements.txt
```

---

## **How to Run**

1. Install dependencies from `requirements.txt` (CPU-only version).
2. Place your test images in the `inputs/` folder.
3. Run the main script:

```bash
python main.py
```

4. The manipulated images will be saved in the `outputs/` folder.



