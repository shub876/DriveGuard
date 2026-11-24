import cv2


class Camera:
    def __init__(self, source=0, width=1280, height=720):
        self.source = source
        self.capture = cv2.VideoCapture(source)

        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Could not open video source: {source}")

        if isinstance(source, int):
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        return self.capture.read()

    def get_resolution(self):
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return width, height

    def get_fps(self):
        return self.capture.get(cv2.CAP_PROP_FPS)

    def get_position_seconds(self):
        return self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    def release(self):
        self.capture.release()
