import yaml
import os
import json
import numpy as np
from modules.yolopose_tracker import YOLOPoseTracker
from modules.track_matcher import TrackMatcher
from modules.track_processor import TrackPostProcessor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")



def main():


    # Run detection + tracking (JSONs written to RESULTS_DIR)
    # Load config
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    if not os.path.exists(os.path.join(cfg['root_output'], 'normalised_raw_pose_annotation')):

        
        tracker = YOLOPoseTracker(
            root_path=cfg["root_path"],
            input_dataset=cfg["input_dataset"],
            tracker=cfg["tracker_type"],
            model_path=cfg["model_path"],
            min_track_length=cfg["min_track_length"],
            max_persons=cfg["max_persons"],
        )

    
        tracker.annotater(cfg['root_output'])
        annotations = tracker.annotation

    else:

        with open(os.path.join(cfg['root_output'], f"RGB_annotation.json"), "r") as f:
            annotations = yaml.safe_load(f)


    # #Run postprocess and bbox matcher 
    matcher = TrackMatcher(
        root_path=cfg["root_output"],
        annotations=annotations
    )

    matcher.match_tracks()
    postprocessor = TrackPostProcessor(
        input_folder="/home/stage/SL_Skeleton-based-detection/keypoints_annotation/results/matched_annotations",
        output_folder="/home/stage/SL_Skeleton-based-detection/keypoints_annotation/results/padded_json_folder",
        total_frames=30
    )
    postprocessor.process_all_files()





       

if __name__ == "__main__":
    main()
