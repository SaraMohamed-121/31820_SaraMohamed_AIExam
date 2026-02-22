"""
    Sub-Task 1B
    Replaces a selected object using YOLO masks and OpenCV seamlessClone.
"""

import cv2
import numpy as np

def replace_object(image, masks, boxes, classes, replacement_path, output_path, target_class=None):
    
    h, w = image.shape[:2]

    if masks is None or len(masks) == 0:
        print("No objects detected.")
        return

    # Load replacement image
    replacement = cv2.imread(replacement_path, cv2.IMREAD_UNCHANGED)
    if replacement is None:
        print(f"Replacement image not found: {replacement_path}")
        return

    # COCO class names
    class_names = [
        'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
        'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
        'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
        'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
        'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
        'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
        'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
        'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
        'remote','keyboard','cell phone','microwave','oven','toaster','sink',
        'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
        'toothbrush'
    ]

    # Choose first detected object if no target_class specified
    if target_class is None:
        target_class = class_names[int(classes[0])]

    # Find the first object matching target_class
    found = False
    for i, cls_id in enumerate(classes):
        if class_names[int(cls_id)] == target_class:
            # Mask for this object
            mask = (masks[i].cpu().numpy() * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask, (w, h))
            found = True

            # Bounding box
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Resize replacement image to fit bounding box
            replacement_resized = cv2.resize(replacement, (x2 - x1, y2 - y1))

            # Prepare mask for seamlessClone
            mask_box = mask_resized[y1:y2, x1:x2]
            _, mask_bin = cv2.threshold(mask_box, 127, 255, cv2.THRESH_BINARY)
            mask_3c = cv2.merge([mask_bin]*3)

            # Center for seamlessClone
            center = ((x1 + x2)//2, (y1 + y2)//2)

            # Seamless clone
            output = cv2.seamlessClone(replacement_resized, image, mask_3c, center, cv2.NORMAL_CLONE)
            cv2.imwrite(output_path, output)
            print(f"Object replaced and saved at: {output_path}")
            break

    if not found:
        print(f"Target class '{target_class}' not found in image.")