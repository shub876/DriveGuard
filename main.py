import cv2


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        camera.release()
        return

    print("DriveGuard started.")
    print("Press Q to exit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Error: Could not read frame.")
                break

            cv2.imshow("DriveGuard", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("DriveGuard stopped.")


if __name__ == "__main__":
    main()