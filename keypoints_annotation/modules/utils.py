import cv2
import numpy as np
import os
import torch

def denormalize_bbox(row, width: int, height: int) -> list[float]:
    x1 = row.box0 * width
    y1 = row.box1 * height
    x2 = row.box2 * width
    y2 = row.box3 * height
    return [x1, y1, x2, y2]

def normalize_bbox(bbox, img_width, img_height):
    """
    Normalize bounding box coordinates to [0, 1] range ***
    """
    if bbox is None:
        return None
    
    # Convert to numpy array if it's a tensor
    if torch.is_tensor(bbox):
        bbox = bbox.cpu().numpy()
    
    x1, y1, x2, y2 = bbox
    
    # Normalize coordinates
    x1_norm = x1 / img_width
    y1_norm = y1 / img_height
    x2_norm = x2 / img_width
    y2_norm = y2 / img_height
    
    return [x1_norm, y1_norm, x2_norm, y2_norm]

def normalize_keypoints(keypoints, img_width, img_height):
    """
    Normalize keypoints coordinates to [0, 1] range
    """
    if keypoints is None:
        return None
    
    kpts = np.array(keypoints)
    kpts_normalized = kpts.copy()
    kpts_normalized[..., 0] /= img_width   # x coordinates
    kpts_normalized[..., 1] /= img_height  # y coordinates
    return kpts_normalized.tolist()
    

def get_clip_dimensions(clip_path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {clip_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def compute_iou(boxA, boxB):
    """
    boxA, boxB = [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def convert_to_serializable(obj):
    """
    Recursively convert any NumPy arrays in a dict/list to Python lists.
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    

def expand_box_width(bbox, ratio, img_width):
        box_width = bbox[2] - bbox[0]
        expanded_box_width = int(box_width * ratio)
        new_x1 = int(bbox[0] - (expanded_box_width - box_width)/2)
        new_x2 = int(bbox[2] + (expanded_box_width - box_width) /2)

        new_x1 = max(new_x1, 0) + 1
        new_x2 = min(new_x2, img_width) -1

        return (new_x1, bbox[1], new_x2, bbox[3])
