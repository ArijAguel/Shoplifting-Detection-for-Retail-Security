import cv2
import numpy as np
from typing import List, Tuple, Optional
from config import SKELETON_CONNECTIONS

class KeypointVisualizer:
    """Class for visualizing keypoints on video frames."""

    def __init__(self, skeleton: List[Tuple[int, int]] = SKELETON_CONNECTIONS):
        self.skeleton = skeleton

    def draw_keypoints(
        self,
        frame: np.ndarray,
        keypoints_list: List[List[float]],
        connections: Optional[List[Tuple[int, int]]] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 4,
        thickness: int = 2,
        alpha: float = 0.6
    ) -> np.ndarray:
        """Draw keypoints and skeleton lines on a frame."""
        overlay = frame.copy()
        points = []
        for kp in keypoints_list:
            x, y, conf = kp
            if conf > 0:
                cv2.circle(overlay, (int(x), int(y)), radius, color, -1)
                points.append((int(x), int(y)))
            else:
                points.append(None)

        connections = connections or self.skeleton
        max_kp = len(keypoints_list)

        for idx1, idx2 in connections:
            if idx1 < max_kp and idx2 < max_kp:
                pt1, pt2 = points[idx1], points[idx2]
                if pt1 is not None and pt2 is not None:
                    cv2.line(overlay, pt1, pt2, color, thickness)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def visualize_video(
        self,
        detections: dict,
        video_path: str,
        output_path: str,
        marker_color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.6
    ):
        """
        Overlay keypoints from detections onto a video.

        detections format:
        {
            "person_id1": { "0": [...], "1": [...], ... },
            "person_id2": { "0": [...], "1": [...], ... },
            ...
        }
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_key = str(frame_idx)
            # loop over persons
            for person_id, frames in detections.items():
                if frame_key in frames:
                    keypoints_list = frames[frame_key]['keypoints']
                    frame = self.draw_keypoints(frame, keypoints_list,
                                                color=marker_color, alpha=alpha)

            out.write(frame)

        cap.release()
        out.release()
        print(f"[INFO] Visualization saved to: {output_path}")
