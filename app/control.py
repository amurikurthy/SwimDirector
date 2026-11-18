"""Speed controller for one PTZ axis."""
import math


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class AxisController:
    """Feedforward + PI on position error expressed in "speed-seconds"
    (error_units / units-per-second-per-speed-step): kp 1 closes an error in ~1 s.
    `ff` is the speed step that matches the swimmer's own motion."""

    def __init__(self, kp=1.2, ki=0.3, deadband=0.3, max_speed=45, min_speed=2, slew=150,
                 integral_limit=10, coast_seconds=1.5, **_):
        self.kp, self.ki = kp, ki
        self.deadband, self.max_speed, self.min_speed = deadband, max_speed, min_speed
        self.slew, self.integral_limit, self.coast_seconds = slew, integral_limit, coast_seconds
        self.reset()

    def reset(self):
        self.i_term = 0.0
        self.out = 0.0
        self.ff = 0.0

    def _finish(self, target, dt):
        step = self.slew * dt
        self.out += clamp(target - self.out, -step, step)
        self.out = clamp(self.out, -self.max_speed, self.max_speed)
        o = self.out
        if 0.5 <= abs(o) < self.min_speed:
            o = math.copysign(self.min_speed, o)
        return o

    def update(self, err, dt, ff=0.0):
        e = 0.0 if abs(err) < self.deadband else err - math.copysign(self.deadband, err)
        self.i_term = clamp(self.i_term + self.ki * e * dt, -self.integral_limit, self.integral_limit)
        self.ff = ff
        return self._finish(ff + self.kp * e + self.i_term, dt)

    def coast(self, dt):
        k = max(0.0, 1 - dt / max(self.coast_seconds, 1e-3))
        self.i_term *= k
        self.ff *= k
        return self._finish(self.ff + self.i_term, dt)
