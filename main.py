import argparse
import time

import cv2

from datetime import datetime
from camera import Camera
from face_detector import FaceDetector, draw_landmarks
from metrics import calculate_eye_metrics, extract_blendshapes
from eye_monitor import classify_eyes, EyeMonitor, CRITICAL_DURATION
from alarm import Alarm

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
    eye_monitor = EyeMonitor()
    alarm = None

    try:
        face_detector = FaceDetector()

        if args.video is None:
            alarm = Alarm()

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

            if args.video is not None:
                timestamp = camera.get_position_seconds()
            else:
                timestamp = time.monotonic()

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

                    eye_status = classify_eyes(
                        left_ear,
                        right_ear,
                        blendshapes
                    )

                    if eye_status == "Eyes closed":
                        eye_color = (0, 0, 255)

                    elif eye_status == "Eyes open":
                        eye_color = (0, 255, 0)

                    else:
                        eye_color = (0, 255, 255)

                draw_landmarks(frame, landmarks)

                face_status = f"Face detected | Landmarks: {len(landmarks)}"
                face_color = (0, 255, 0)

            else:
                face_status = "No face detected"
                face_color = (0, 0, 255)

            event = eye_monitor.update(
                eye_status,
                timestamp
            )

            if event is not None:
                duration, interrupted = event

                if args.video is not None:
                    event_time = f"Video {timestamp:.2f}s"
                else:
                    event_time = datetime.now().strftime("%H:%M:%S")

                severity = (
                    "CRITICAL"
                    if duration >= CRITICAL_DURATION
                    else "WARNING"
                )

                if interrupted:
                    print(
                        f"[{event_time}] [{severity}] - "
                        f"Eye closure observed for {duration:.2f}s "
                        f"(end unconfirmed)"
                    )

                else:
                    print(
                        f"[{event_time}] [{severity}] - "
                        f"Driver's eyes were closed for "
                        f"approximately {duration:.2f}s"
                    )

            monitor_status = eye_monitor.status

            if alarm is not None:
                alarm.update(
                    monitor_status,
                    timestamp
                )

            if monitor_status == "CRITICAL":
                monitor_color = (0, 0, 255)

            elif monitor_status in (
                "WARNING",
                "UNCERTAIN",
                "UNAVAILABLE",
                "TIMING UNAVAILABLE"
            ):
                monitor_color = (0, 255, 255)

            else:
                monitor_color = (0, 255, 0)

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

            cv2.putText(
                frame,
                f"Blinks: {eye_monitor.blink_count}",
                (15, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Eye closure: {eye_monitor.closed_seconds:.2f}s",
                (15, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Eye monitor: {monitor_status}",
                (15, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                monitor_color,
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
        if alarm is not None:
            alarm.close()

        event = eye_monitor.finish_closure(interrupted=True)

        if event is not None:
            duration, _ = event

            if args.video is not None:
                event_time = (
                    f"Video {eye_monitor.last_timestamp:.2f}s"
                )
            else:
                event_time = datetime.now().strftime("%H:%M:%S")

            severity = (
                "CRITICAL"
                if duration >= CRITICAL_DURATION
                else "WARNING"
            )

            print(
                f"[{event_time}] [{severity}] - "
                f"Eye closure observed for {duration:.2f}s "
                f"(monitoring stopped)"
            )

        camera.release()
        cv2.destroyAllWindows()

        if face_detector is not None:
            face_detector.close()

        print("DriveGuard stopped.")


if __name__ == "__main__":
    main()
