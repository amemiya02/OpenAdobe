"""Frame-accurate time handling.

All timeline positions are stored as integer frames at the project rate. Agents may pass
times as frames (int), seconds ("2.5s"), timecode ("00:00:02:15") or clock ("00:00:02.500").
"""

from __future__ import annotations

import re
from fractions import Fraction

from pydantic import BaseModel

_TC = re.compile(r"^(\d+):(\d{2}):(\d{2})[:;](\d{2})$")
_CLOCK = re.compile(r"^(?:(\d+):)?(\d+):(\d{2}(?:\.\d+)?)$")


class Rate(BaseModel):
    num: int = 30
    den: int = 1

    @classmethod
    def parse(cls, value: str | float) -> Rate:
        text = str(value).strip()
        if "/" in text:
            num, den = text.split("/", 1)
            return cls(num=int(num), den=int(den))
        known = {"23.976": (24000, 1001), "29.97": (30000, 1001), "59.94": (60000, 1001)}
        if text in known:
            return cls(num=known[text][0], den=known[text][1])
        frac = Fraction(text).limit_denominator(1001)
        return cls(num=frac.numerator, den=frac.denominator)

    @property
    def fps(self) -> Fraction:
        return Fraction(self.num, self.den)

    def __str__(self) -> str:
        return f"{self.num}/{self.den}"

    def to_seconds(self, frames: int) -> float:
        return float(Fraction(frames) / self.fps)

    def to_frames(self, seconds: float) -> int:
        return round(Fraction(str(seconds)) * self.fps)

    def nominal(self) -> int:
        return round(self.fps)

    def timecode(self, frames: int) -> str:
        """Non-drop-frame timecode (HH:MM:SS:FF) using the nominal integer rate."""
        base = self.nominal()
        sign = "-" if frames < 0 else ""
        frames = abs(frames)
        ff = frames % base
        total_s = frames // base
        return f"{sign}{total_s // 3600:02d}:{total_s // 60 % 60:02d}:{total_s % 60:02d}:{ff:02d}"

    def clock(self, frames: int) -> str:
        secs = self.to_seconds(frames)
        sign = "-" if secs < 0 else ""
        secs = abs(secs)
        return f"{sign}{int(secs // 3600):02d}:{int(secs // 60 % 60):02d}:{secs % 60:06.3f}"


def parse_time(value: float | str, rate: Rate) -> int:
    """Convert an agent-supplied time to frames. Bare numbers are frames; '1.5s' is seconds."""
    if isinstance(value, bool):
        raise ValueError(f"invalid time: {value!r}")  # noqa: TRY004 - agents get one error type
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"fractional frame count {value!r}; use '{value}s' for seconds")
        return int(value)
    text = value.strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if text.endswith("ms"):
        return rate.to_frames(float(text[:-2]) / 1000)
    if text.endswith("s"):
        return rate.to_frames(float(text[:-1]))
    if text.endswith("f"):
        return int(text[:-1])
    if m := _TC.match(text):
        h, mi, s, f = (int(g) for g in m.groups())
        return (h * 3600 + mi * 60 + s) * rate.nominal() + f
    if m := _CLOCK.match(text):
        h = int(m.group(1) or 0)
        return rate.to_frames(h * 3600 + int(m.group(2)) * 60 + float(m.group(3)))
    raise ValueError(f"unrecognised time {value!r}; use frames (120), seconds ('4s') or timecode ('00:00:04:00')")
