from typing import List, Tuple

SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (0, 2),          # Nose to eyes
    (1, 3), (2, 4),          # Eyes to ears
    (0, 5), (0, 6),          # Nose to shoulders
    (5, 7), (7, 9),          # Left arm
    (6, 8), (8, 10),         # Right arm
    (5, 11), (6, 12),        # Shoulders to hips
    (11, 13), (13, 15),      # Left leg
    (12, 14), (14, 16)       # Right leg
]
