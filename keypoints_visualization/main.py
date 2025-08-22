import argparse
from pathlib import Path
from visualizer import KeypointVisualizer
from utils import load_json

def parse_args():
    parser = argparse.ArgumentParser(description="Soft keypoint visualization")
    parser.add_argument('--json', required=True, help='Path to JSON file')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output', required=True, help='Path to save output video')
    parser.add_argument('--color', nargs=3, type=int, default=[0, 255, 0], help='BGR color')
    parser.add_argument('--alpha', type=float, default=0.6, help='Transparency for overlay')
    return parser.parse_args()

def main():
    args = parse_args()
    json_path = Path(args.json)
    video_path = Path(args.video)
    output_path = Path(args.output)

    detections = load_json(json_path)

    visualizer = KeypointVisualizer()
    visualizer.visualize_video(
        detections=detections,
        video_path=str(video_path),
        output_path=str(output_path),
        marker_color=tuple(args.color),
        alpha=args.alpha
    )

if __name__ == "__main__":
    main()
