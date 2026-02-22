"""
    Sub-Task 1A 
    Removes a selected object from the image using YOLOv8 masks and OpenCV inpainting.
    
    Parameters:
    - image: np.ndarray, original image (H x W x 3)
    - masks: list of torch.Tensor, YOLO segmentation masks (values 0-1)
    - boxes: list of bounding boxes (not used here but kept for API consistency)
    - classes: list of class IDs corresponding to masks
    - output_path: path to save the resulting image
    - target_class: string, optional COCO class name to remove
"""
import cv2
import numpy as np

def remove_object(image, masks, boxes, classes, output_path, target_class=None):
    
    if masks is None or len(masks) == 0:
        print("No objects detected.")
        return

    h, w = image.shape[:2]

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

    # Default: choose first detected object if no target_class specified
    if target_class is None:
        target_class = class_names[int(classes[0])]

    # Initialize final mask
    final_mask = np.zeros((h, w), dtype=np.uint8)

    # Combine masks of all objects matching the target class
    for i, cls_id in enumerate(classes):
        if class_names[int(cls_id)] == target_class:
            mask_np = (masks[i].cpu().numpy() * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (w, h))  # resize to match image
            final_mask = cv2.bitwise_or(final_mask, mask_resized)

    if np.sum(final_mask) == 0:
        print(f"Selected class '{target_class}' not found in image.")
        return

    # Optional: threshold to ensure binary mask
    _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)

    # Inpaint the selected region
    output = cv2.inpaint(image, final_mask, 7, cv2.INPAINT_TELEA)

    # Save output
    cv2.imwrite(output_path, output)
    print(f"Object removed and saved at: {output_path}")