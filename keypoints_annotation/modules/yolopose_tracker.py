import cv2
import numpy as np
import copy
import pandas as pd
from ultralytics import YOLO
import os
import json
import torch
from torchvision.ops import box_iou
from modules.utils import (
    denormalize_bbox,
    get_clip_dimensions,
    normalize_keypoints,
    normalize_bbox
)

class YOLOPoseTracker:
    def __init__(
        self,
        root_path: str,
        input_dataset: str,
        tracker: str,
        model_path="yolov8m-pose.pt",
        min_track_length=5,
        max_persons=3
    ):
        self.root_path = root_path
        self.tracker = tracker
        self.max_persons = max_persons
        self.min_track_length = min_track_length
        self.track_history = {}
        self.model = YOLO(model_path)
        self.input_dataset = input_dataset
        self.annotation = self.reorganize_annotations() 
        self.result = None

    def reorganize_annotations(self) -> dict:
        """Return annotations with normalized bboxes (from CSV)"""
        df = pd.read_csv(self.input_dataset, header=0).iloc[:, :8]
        annotations = {}

        for _, row in df.iterrows():
            clip_key = f"{row.video_name}/{row.Clip_name}"
            clip_file = os.path.join(self.root_path, f"{clip_key}.mp4")

            if not os.path.exists(clip_file):
                print(f"[WARNING] Clip not found: {clip_file}, skipping...")
                continue
            
            # Store normalized bboxes (original from CSV)
            if clip_key not in annotations:
                annotations[clip_key] = {}
            
            if row.id not in annotations[clip_key]:
                annotations[clip_key][row.id] = {
                    "bbox": [row.box0, row.box1, row.box2, row.box3],  # Already normalized
                    "class": row.class_
                }

        return annotations  

    def annotater(self, save_dir=None):
        """Process all videos and save both raw and normalized annotations"""
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for clip_key in self.annotation.keys():
            clip_file = os.path.join(self.root_path, f"{clip_key}.mp4")
            results = self.model.track(
                source=clip_file,
                tracker=self.tracker,
                classes=0,
                stream=True
            )

            clip_results = {}
            normalized_clip_results = {} 

            for frame_id, result in enumerate(results):
                # Get image dimensions for normalization
                img_height, img_width = result.orig_shape

                # get boxes and ids
                boxes = result.boxes.xyxy.cpu() if result.boxes is not None else torch.empty((0, 4))
                ids = result.boxes.id.cpu().numpy().astype(int).tolist() if result.boxes is not None and result.boxes.id is not None else []

                # get keypoints
                keypoints = []
                normalized_keypoints = [] 
                #if result.keypoints is not None and result.keypoints.xy is not None and result.keypoints.conf is not None:
                if result.keypoints is not None and len(result.keypoints) > 0:
                    xy = result.keypoints.xy.cpu().numpy()
                    if result.keypoints.conf is None:
                        conf = np.zeros((xy.shape[0], xy.shape[1]), dtype=np.float32)
                    else:
                        conf = result.keypoints.conf.cpu().numpy()

                    
                    # Raw keypoints
                    keypoints = np.concatenate([xy, conf[..., None]], axis=-1).tolist()
                    
                    # Normalized keypoints
                    """normalized_xy = normalize_keypoints(xy, img_width, img_height)
                    normalized_keypoints = np.concatenate([normalized_xy, conf[..., None]], axis=-1).tolist()"""
                    xy_normalized = result.keypoints.xyn.cpu().numpy()  # Already normalized [0,1]
                    normalized_keypoints = np.concatenate([xy_normalized, conf[..., None]], axis=-1).tolist()

                # store per track per frame
                for i, track_id in enumerate(ids):
                    if track_id not in clip_results:
                        clip_results[track_id] = {}
                    if track_id not in normalized_clip_results:
                        normalized_clip_results[track_id] = {}

                 
                    # Raw data
                    clip_results[track_id][frame_id] = {
                        'keypoints': keypoints[i] if i < len(keypoints) else None,
                        'bbox': boxes[i].tolist() if i < len(boxes) else None
                    }

                    # Normalized data
                    normalized_bbox = normalize_bbox(boxes[i], img_width, img_height) if i < len(boxes) else None
                    
                    normalized_clip_results[track_id][frame_id] = {
                        'keypoints': normalized_keypoints[i] if i < len(normalized_keypoints) else None,
                        'bbox': normalized_bbox
                    }

            # Save raw json file 
            if clip_results:
                os.makedirs(os.path.join(save_dir, 'raw_pose_annotation'), exist_ok=True)
                json_path = os.path.join(save_dir, 'raw_pose_annotation', f"{clip_key.replace('/', '_')}.json")
                with open(json_path, "w") as f:
                    json.dump(clip_results, f, indent=2)

            # Save normalized json file
            if normalized_clip_results:
                os.makedirs(os.path.join(save_dir, 'normalized_raw_pose_annotation'), exist_ok=True)
                normalized_json_path = os.path.join(save_dir, 'normalized_raw_pose_annotation', f"{clip_key.replace('/', '_')}.json")
                with open(normalized_json_path, "w") as f:
                    json.dump(normalized_clip_results, f, indent=2)

        # Save RGB annotations (normalized - same as self.annotation)
        RGB_annotation = os.path.join(save_dir, f"RGB_annotation.json")
        with open(RGB_annotation, "w") as f:
            json.dump(self.annotation, f, indent=2)

        return None