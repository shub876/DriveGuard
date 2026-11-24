EAR_THRESHOLD = 0.15

BLINK_OPEN_THRESHOLD = 0.35
BLINK_CLOSED_THRESHOLD = 0.65


def classify_eye(ear, blink_score):
    if ear is None or blink_score is None:
        return "UNAVAILABLE"

    if (
        ear < EAR_THRESHOLD
        and blink_score >= BLINK_CLOSED_THRESHOLD
    ):
        return "CLOSED"

    if (
        ear >= EAR_THRESHOLD
        and blink_score <= BLINK_OPEN_THRESHOLD
    ):
        return "OPEN"

    return "UNCERTAIN"


def classify_eyes(left_ear, right_ear, blendshapes):
    left_blink = blendshapes.get("eyeBlinkLeft")
    right_blink = blendshapes.get("eyeBlinkRight")

    left_state = classify_eye(left_ear, left_blink)
    right_state = classify_eye(right_ear, right_blink)

    if "UNAVAILABLE" in (left_state, right_state):
        return "Eye state unavailable"

    if "UNCERTAIN" in (left_state, right_state):
        return "Eye state uncertain"

    if left_state == "CLOSED" and right_state == "CLOSED":
        return "Eyes closed"

    if left_state == "CLOSED":
        return "Left eye closed"

    if right_state == "CLOSED":
        return "Right eye closed"

    return "Eyes open"
