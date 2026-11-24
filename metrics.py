import math


RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)


def landmark_to_pixel(landmark, width, height):
    return (
        landmark.x * width,
        landmark.y * height
    )


def distance(point1, point2):
    return math.hypot(
        point2[0] - point1[0],
        point2[1] - point1[1]
    )


def calculate_ear(points):
    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
    horizontal = distance(p1, p4)

    if horizontal < 1e-6:
        return None

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


def calculate_eye_metrics(landmarks, width, height):
    right_points = [
        landmark_to_pixel(landmarks[i], width, height)
        for i in RIGHT_EYE
    ]

    left_points = [
        landmark_to_pixel(landmarks[i], width, height)
        for i in LEFT_EYE
    ]

    right_ear = calculate_ear(right_points)
    left_ear = calculate_ear(left_points)

    if right_ear is None or left_ear is None:
        return None

    average_ear = (left_ear + right_ear) / 2.0

    return left_ear, right_ear, average_ear

def extract_blendshapes(result):
    if not result.face_blendshapes:
        return {}

    return {
        category.category_name: category.score
        for category in result.face_blendshapes[0]
    }
