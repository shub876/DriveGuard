import time
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "face_landmarker.task"
)


class FaceDetector:
    def __init__(self):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True
        )

        self.landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(options)
        )

        self.last_timestamp_ms = -1

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = time.monotonic_ns() // 1_000_000

        timestamp_ms = max(
            timestamp_ms,
            self.last_timestamp_ms + 1
        )

        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        return result

    def close(self):
        self.landmarker.close()


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
