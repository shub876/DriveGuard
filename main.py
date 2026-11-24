import argparse
import time

import cv2

from camera import Camera
from face_detector import FaceDetector, draw_landmarks
from metrics import calculate_eye_metrics, extract_blendshapes

EAR_THRESHOLD = 0.15


def main():
    parser = argparse.ArgumentParser(description="DriveGuard")

    source_group = parser.add_mutually_exclusive_group()

    source_group.add_argument("--camera", type=int, default=0)
    source_group.add_argument("--video", type=str)

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)

    args = parser.parse_args()

    source = args.video if args.video is not None else args.camera

    try:
        camera = Camera(source, args.width, args.height)
    except RuntimeError as error:
        print(f"Error: {error}")
        return

    face_detector = None

    try:
        face_detector = FaceDetector()

        width, height = camera.get_resolution()
        source_fps = camera.get_fps()

        print("DriveGuard started.")
        print(f"Video source: {source}")
        print(f"Reported resolution: {width}x{height}")
        print(f"Reported FPS: {source_fps:.2f}")
        print("Press Q or ESC to exit.")

        fps = 0.0
        frame_count = 0
        start_time = time.perf_counter()

        while True:
            success, frame = camera.read()

            if not success:
                if args.video is not None:
                    print("End of video or read failure.")
                else:
                    print("Error: Could not read frame.")
                break

            result = face_detector.detect(frame)

            blendshapes = extract_blendshapes(result)

            blink_text = "Blink: N/A"
            gaze_text = "Downward gaze: N/A"

            if blendshapes:
                blink_left = blendshapes["eyeBlinkLeft"]
                blink_right = blendshapes["eyeBlinkRight"]

                down_left = blendshapes["eyeLookDownLeft"]
                down_right = blendshapes["eyeLookDownRight"]

                blink_text = (
                    f"Blink | L: {blink_left:.3f} | R: {blink_right:.3f}"
                )

                gaze_text = (
                    f"Down | L: {down_left:.3f} | R: {down_right:.3f}"
                )

            eye_text = "EAR: N/A"
            eye_status = "Eye state unavailable"
            eye_color = (0, 255, 255)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                height, width = frame.shape[:2]

                eye_metrics = calculate_eye_metrics(
                    landmarks, width, height
                )

                if eye_metrics is not None:
                    left_ear, right_ear, average_ear = eye_metrics

                    eye_text = (
                        f"EAR: {average_ear:.3f} | "
                        f"L: {left_ear:.3f} | "
                        f"R: {right_ear:.3f}"
                    )

                    left_closed = left_ear < EAR_THRESHOLD
                    right_closed = right_ear < EAR_THRESHOLD

                    if left_closed and right_closed:
                        eye_status = "Eyes closed"
                        eye_color = (0, 0, 255)

                    elif left_closed:
                        eye_status = "Left eye closed"
                        eye_color = (0, 255, 255)

                    elif right_closed:
                        eye_status = "Right eye closed"
                        eye_color = (0, 255, 255)

                    else:
                        eye_status = "Eyes open"
                        eye_color = (0, 255, 0)

                draw_landmarks(frame, landmarks)

                face_status = f"Face detected | Landmarks: {len(landmarks)}"
                face_color = (0, 255, 0)

            else:
                face_status = "No face detected"
                face_color = (0, 0, 255)

            frame_count += 1

            elapsed = time.perf_counter() - start_time

            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.perf_counter()

            height, width = frame.shape[:2]

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Resolution: {width}x{height}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                face_status,
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                face_color,
                2
            )

            cv2.putText(
                frame,
                eye_text,
                (15, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                eye_status,
                (15, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                eye_color,
                2
            )

            cv2.putText(
                frame,
                blink_text,
                (15, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                gaze_text,
                (15, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("DriveGuard", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        print("\nMonitoring interrupted.")

    except (RuntimeError, OSError, ValueError) as error:
        print(f"Error: {error}")

    finally:
        camera.release()
        cv2.destroyAllWindows()

        if face_detector is not None:
            face_detector.close()

        print("DriveGuard stopped.")


if __name__ == "__main__":
    main()
