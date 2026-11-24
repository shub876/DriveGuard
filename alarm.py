import math
import wave
import winsound

from array import array
from pathlib import Path
from tempfile import TemporaryDirectory


SAMPLE_RATE = 44100
AMPLITUDE = 0.25

UNCERTAIN_DELAY = 2.0


def generate_sound(path, segments):
    samples = array("h")

    for frequency, duration in segments:
        sample_count = int(SAMPLE_RATE * duration)

        for i in range(sample_count):
            if frequency == 0:
                sample = 0
            else:
                sample = int(
                    32767 * AMPLITUDE
                    * math.sin(
                        2 * math.pi * frequency * i / SAMPLE_RATE
                    )
                )

            samples.append(sample)

    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(samples.tobytes())


class Alarm:
    def __init__(self):
        self.temp_dir = TemporaryDirectory(prefix="driveguard_")
        directory = Path(self.temp_dir.name)

        self.warning_sound = directory / "warning.wav"
        self.critical_sound = directory / "critical.wav"
        self.uncertain_sound = directory / "uncertain.wav"

        generate_sound(
            self.warning_sound,
            [(850, 0.20), (0, 0.30)]
        )

        generate_sound(
            self.critical_sound,
            [(1200, 1.0)]
        )

        generate_sound(
            self.uncertain_sound,
            [(650, 0.08), (0, 0.10), (650, 0.08)]
        )

        self.current_status = None
        self.uncertain_since = None
        self.uncertain_played = False

    def play(self, path, loop=False):
        flags = (
            winsound.SND_FILENAME
            | winsound.SND_ASYNC
            | winsound.SND_NODEFAULT
        )

        if loop:
            flags |= winsound.SND_LOOP

        winsound.PlaySound(str(path), flags)

    def stop(self):
        winsound.PlaySound(None, 0)

    def update(self, status, timestamp):
        if status != self.current_status:
            self.stop()

            self.current_status = status
            self.uncertain_since = None
            self.uncertain_played = False

            if status == "CRITICAL":
                self.play(self.critical_sound, loop=True)

            elif status == "WARNING":
                self.play(self.warning_sound, loop=True)

            elif status == "UNCERTAIN":
                self.uncertain_since = timestamp

        if status == "UNCERTAIN":
            if (
                self.uncertain_since is not None
                and not self.uncertain_played
                and timestamp - self.uncertain_since >= UNCERTAIN_DELAY
            ):
                self.play(self.uncertain_sound)
                self.uncertain_played = True

    def close(self):
        try:
            self.stop()
        finally:
            self.temp_dir.cleanup()
