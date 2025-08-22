import json
import os

class TrackPostProcessor:
    def __init__(self, input_folder, output_folder, total_frames=30):
        """
        input_folder: folder containing raw JSON annotation files
        output_folder: folder where padded JSONs will be saved
        total_frames: number of frames to pad each track to
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.total_frames = total_frames
        os.makedirs(self.output_folder, exist_ok=True)

    def _pad_tracks(self, track_history):
        """
        Pad tracks to self.total_frames length.
        track_history: {track_id: [(frame_id, keypoints), ...]}
        """
        padded_tracks = {}
        for track_id, frames in track_history.items():
            frames = sorted(frames, key=lambda x: x[0])
            frame_ids = [f[0] for f in frames]
            keypoints_list = [f[1] for f in frames]

            padded_person = {}
            for i in range(self.total_frames):
                if i in frame_ids:
                    idx = frame_ids.index(i)
                    kp = keypoints_list[idx]
                else:
                    # padding logic: nearest available frame
                    prev_candidates = [f for f in frame_ids if f < i]
                    next_candidates = [f for f in frame_ids if f > i]
                    if prev_candidates:
                        closest_frame = max(prev_candidates)
                    elif next_candidates:
                        closest_frame = min(next_candidates)
                    else:
                        closest_frame = frame_ids[0]
                    idx = frame_ids.index(closest_frame)
                    kp = keypoints_list[idx]

                padded_person[str(i)] = {"keypoints": kp}

            padded_tracks[str(track_id)] = padded_person
        return padded_tracks

    def _process_file(self, input_path, output_path):
        """
        Load one JSON file, pad its tracks, save result.
        """
        with open(input_path, "r") as f:
            data = json.load(f)

        # Convert to {track_id: [(frame_id, keypoints), ...]}
        track_history = {}
        for track_id, frames in data.items():
            track_history[track_id] = []
            for frame_id, frame_data in frames.items():
                keypoints = frame_data["keypoints"]
                track_history[track_id].append((int(frame_id), keypoints))

        # Pad
        padded_tracks = self._pad_tracks(track_history)

        # Save
        with open(output_path, "w") as out_f:
            json.dump(padded_tracks, out_f, indent=2)

    def process_all_files(self):
        """
        Process all JSON files in input_folder and save results to output_folder.
        """
        for file_name in os.listdir(self.input_folder):
            if file_name.endswith(".json"):
                in_path = os.path.join(self.input_folder, file_name)
                out_path = os.path.join(self.output_folder, f"padded_{file_name}")
                self._process_file(in_path, out_path)
