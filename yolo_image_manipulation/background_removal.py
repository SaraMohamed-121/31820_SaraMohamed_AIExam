"""
    Sub-Task 1C
    Removes background and creates transparent PNG.
"""

import cv2
import numpy as np

def remove_background(image, masks, output_path, feather=False):
    
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

    # feathering (smooth edges)
    if feather:
        combined_mask = cv2.GaussianBlur(combined_mask, (15,15), 0)

    # Invert mask to get background
    alpha = combined_mask.copy()
    alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)[1]

    # Convert image to BGRA
    b, g, r = cv2.split(image)
    rgba = cv2.merge([b, g, r, alpha])

    cv2.imwrite(output_path, rgba)
    print(f"Background removed and saved at: {output_path}")