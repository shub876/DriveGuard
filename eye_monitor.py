EAR_THRESHOLD = 0.15
EAR_PARTIAL_THRESHOLD = 0.10

BLINK_OPEN_THRESHOLD = 0.40
BLINK_CLOSED_THRESHOLD = 0.65

WINK_EAR_THRESHOLD = 0.20
WINK_BLINK_THRESHOLD = 0.44
WINK_ASYMMETRY_THRESHOLD = 0.10


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
        and blink_score < BLINK_CLOSED_THRESHOLD
    ):
        return "OPEN"

    if (
        ear >= EAR_PARTIAL_THRESHOLD
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

    if left_state == "CLOSED" and right_state == "CLOSED":
        return "Eyes closed"

    blink_difference = left_blink - right_blink

    if (
        left_ear < WINK_EAR_THRESHOLD
        and left_blink >= WINK_BLINK_THRESHOLD
        and blink_difference >= WINK_ASYMMETRY_THRESHOLD
        and right_state == "OPEN"
        and right_blink <= BLINK_OPEN_THRESHOLD
    ):
        return "Left eye closed"

    if (
        right_ear < WINK_EAR_THRESHOLD
        and right_blink >= WINK_BLINK_THRESHOLD
        and blink_difference <= -WINK_ASYMMETRY_THRESHOLD
        and left_state == "OPEN"
        and left_blink <= BLINK_OPEN_THRESHOLD
    ):
        return "Right eye closed"

    if "UNCERTAIN" in (left_state, right_state):
        return "Eye state uncertain"

    if left_state == "CLOSED":
        return "Left eye closed"

    if right_state == "CLOSED":
        return "Right eye closed"

    return "Eyes open"


MIN_BLINK_DURATION = 0.08
MAX_BLINK_DURATION = 0.40

WARNING_DURATION = 2.00
CRITICAL_DURATION = 5.50

MAX_FRAME_GAP = 0.35
MAX_UNCERTAIN_DURATION = 0.75


class EyeMonitor:
    def __init__(self):
        self.blink_count = 0

        self.closure_active = False
        self.closed_seconds = 0.0

        self.last_timestamp = None
        self.last_eye_status = None
        self.uncertain_since = None

        self.status = "UNAVAILABLE"

    def finish_closure(self, interrupted=False, count_blink=False):
        duration = self.closed_seconds
        event = None

        if self.closure_active:
            if (
                not interrupted
                and count_blink
                and MIN_BLINK_DURATION <= duration <= MAX_BLINK_DURATION
            ):
                self.blink_count += 1

            elif duration >= WARNING_DURATION:
                event = (duration, interrupted)

        self.closure_active = False
        self.closed_seconds = 0.0
        self.uncertain_since = None

        return event

    def update(self, eye_status, timestamp):
        event = None

        previous_timestamp = self.last_timestamp
        previous_status = self.last_eye_status

        self.last_timestamp = timestamp
        self.last_eye_status = eye_status

        elapsed = 0.0

        if previous_timestamp is not None:
            elapsed = timestamp - previous_timestamp

            if not 0 < elapsed <= MAX_FRAME_GAP:
                event = self.finish_closure(interrupted=True)

                if elapsed <= 0:
                    self.status = "TIMING UNAVAILABLE"
                    return event

                elapsed = 0.0
                previous_status = None

        if eye_status == "Eye state unavailable":
            interrupted_event = self.finish_closure(interrupted=True)

            self.status = "UNAVAILABLE"
            return event or interrupted_event

        if self.uncertain_since is not None:
            uncertain_duration = timestamp - self.uncertain_since

            if uncertain_duration > MAX_UNCERTAIN_DURATION:
                interrupted_event = self.finish_closure(interrupted=True)
                event = event or interrupted_event

        if self.closure_active and previous_status == "Eyes closed":
            self.closed_seconds += elapsed

        if eye_status == "Eye state uncertain":
            if self.closure_active:
                if self.uncertain_since is None:
                    self.uncertain_since = timestamp

                if self.closed_seconds >= CRITICAL_DURATION:
                    self.status = "CRITICAL"

                elif self.closed_seconds >= WARNING_DURATION:
                    self.status = "WARNING"

                else:
                    self.status = "UNCERTAIN"

            else:
                self.status = "UNCERTAIN"

            return event

        self.uncertain_since = None

        if eye_status == "Eyes closed":
            self.closure_active = True

            if self.closed_seconds >= CRITICAL_DURATION:
                self.status = "CRITICAL"

            elif self.closed_seconds >= WARNING_DURATION:
                self.status = "WARNING"

            else:
                self.status = "MONITORING"

        else:
            completed_event = self.finish_closure(
                count_blink=eye_status == "Eyes open"
            )

            event = event or completed_event

            self.status = "NORMAL"

        return event
