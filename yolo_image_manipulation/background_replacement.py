"""
    Sub-Task 1D
    Replaces background using masks from YOLO segmentation.
"""

import cv2
import numpy as np

def replace_background(image, masks, output_path, new_bg_path, feather=False):
    
    h, w = image.shape[:2]

    if masks is None or len(masks) == 0:
        print("No objects detected.")
        return

    # Combine all foreground masks
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask_np, (w, h))
        combined_mask = cv2.bitwise_or(combined_mask, mask_resized)

    if feather:
        combined_mask = cv2.GaussianBlur(combined_mask, (15, 15), 0)

    # Read new background
    new_bg = cv2.imread(new_bg_path)
    if new_bg is None:
        print(f"Error: Background image not found at {new_bg_path}")
        return
    new_bg = cv2.resize(new_bg, (w, h))

    # Create 3-channel mask
    mask_3c = cv2.merge([combined_mask]*3)

    # Composite foreground on new background
    fg = cv2.bitwise_and(image, mask_3c)
    bg_mask = cv2.bitwise_not(combined_mask)
    bg_mask_3c = cv2.merge([bg_mask]*3)
    bg = cv2.bitwise_and(new_bg, bg_mask_3c)

    output = cv2.add(fg, bg)

    cv2.imwrite(output_path, output)
    print(f"Background replaced and saved at: {output_path}")