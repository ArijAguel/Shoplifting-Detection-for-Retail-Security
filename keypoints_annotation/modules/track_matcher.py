import json
import os
import torch
from torchvision.ops import box_iou

class TrackMatcher:
    def __init__(self, root_path, annotations):
        self.annotations = annotations  #normalized bboxes
        self.root_path = root_path

    def match_tracks(self):
        """
        Match tracks based on the provided annotations.
        """
        for clip_key, clip_ann in self.annotations.items(): 
            print('-------matching normalized bbox for-------- ', clip_key)

            pred_bboxes = []
            pred_ids = []

            json_file = os.path.join(self.root_path, 'normalized_raw_pose_annotation', clip_key.replace("/", "_") + ".json")
            print('json_files', json_file)
            
            if not os.path.exists(json_file):
                print(f"Normalized prediction file not found: {json_file}")
                continue

            with open(json_file, "r") as f:
                pred_data = json.load(f)

            for track_id, frames in pred_data.items():
                sorted_frame_ids = sorted(frames.keys(), key=lambda x: int(x))
                num_frames = len(sorted_frame_ids)
                if num_frames == 0:
                    continue

                print('num_frames', num_frames)

                # mid-frame index
                mid_idx = num_frames // 2
                mid_frame_id = sorted_frame_ids[mid_idx]
                mid_frame_data = frames[mid_frame_id]

                bbox = mid_frame_data.get("bbox", None)
                if bbox is not None:
                    pred_bboxes.append(bbox)
                    pred_ids.append(track_id)

            if not pred_bboxes:
                continue

            pred_bboxes_tensor = torch.tensor(pred_bboxes, dtype=torch.float32)
            
            for ann_id, ann_data in clip_ann.items():
                # Both GT and predictions are now normalized
                gt_bbox = torch.tensor([ann_data["bbox"]], dtype=torch.float32)
                ious = box_iou(gt_bbox, pred_bboxes_tensor)
                best_track_id = pred_ids[torch.argmax(ious).item()]
                track_frames = pred_data[best_track_id]

                print('annotation id ', ann_id, 'with', best_track_id, ': iou', ious.max().item())

                
                keypoints = {
                    f: {"keypoints": frame_data.get("keypoints", None)}
                    for f, frame_data in track_frames.items()
                }

                save_dir = os.path.join(self.root_path, "matched_annotations")
                os.makedirs(save_dir, exist_ok=True)
                save_file = os.path.join(save_dir, clip_key.replace("/", "_") + ".json")

                if os.path.exists(save_file):
                    with open(save_file, "r") as f:
                        final_ann = json.load(f)
                else:
                    final_ann = {}

                final_ann[ann_id] = keypoints

                with open(save_file, "w") as f:
                    json.dump(final_ann, f, indent=2)