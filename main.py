
import argparse
import time

import cv2

from camera import Camera


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

    try:
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

            cv2.imshow("DriveGuard", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        print("\nMonitoring interrupted.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("DriveGuard stopped.")


if __name__ == "__main__":
    main()
