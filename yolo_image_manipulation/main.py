from ultralytics import YOLO
from object_removal import remove_object
from object_replacement import replace_object
from background_removal import remove_background
from background_replacement import replace_background
import cv2

if __name__ == "__main__":
    image_path = "inputs/test_image3.jpg"
    model_path = "models/yolov8n-seg.pt"

    model = YOLO(model_path)
    image = cv2.imread(image_path)

    results = model(image)
    masks = results[0].masks.data
    boxes = results[0].boxes.xyxy
    classes = results[0].boxes.cls

    remove_object(image,masks,boxes,classes,output_path="outputs/object_removed3.jpg")
    replace_object(image,masks,boxes,classes,replacement_path="inputs/replacement_object3.jpg",output_path="outputs/object_replaced3.jpg")
    remove_background(image,masks,output_path="outputs/bg_removed3.png",feather=True)
    replace_background(image, masks,output_path="outputs/bg_replaced3.jpg",new_bg_path="inputs/new_background3.jfif",feather=True)