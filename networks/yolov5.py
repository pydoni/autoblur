import torch
import numpy as np

def yolov5_model_config(weights, confidence, iou, classes, max_det, device, _verbose= False):
    model = torch.hub.load("ultralytics/yolov5", "custom", path=weights)  
    model.conf = confidence  # NMS confidence threshold
    model.iou = iou  # NMS IoU threshold
    model.classes = classes  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    model.max_det = max_det  # maximum number of detections per image
    if device == "CPU":
        model.cpu()  # CPU
    if device == "GPU":
        model.cuda()  # GPU
    if device.isnumeric(): 
        model.to(device)
    
    return model

def yolov5_inference(image, compiled_model):
    results = compiled_model(image)
    results_df = results.pandas().xyxy[0]
    results_df[["xmin","ymin","xmax","ymax"]] = results_df[["xmin","ymin","xmax","ymax"]].apply(np.floor).astype("int")
    return results_df
