"""One-shot open-loop slalom commands and measured, backend-independent scoring."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import brentq


@dataclass(frozen=True)
class SlalomConfig:
    enabled: bool = False
    pole_count: int = 5
    spacing_m: float = 1.0
    pole_radius_m: float = 0.075
    pole_height_m: float = 0.8
    pole_mass_kg: float = 1.0
    pole_friction: float = 0.6
    amplitude_m: float = 0.5
    speed_m_s: float = 0.2
    first_pole_x_m: float = 1.5
    lead_m: float = 0.5
    segment_duration_s: float = 0.8
    settle_s: float = 2.0
    tip_angle_deg: float = 30.0
    tip_hold_s: float = 0.2
    max_pole_displacement_m: float = 0.1
    pass_clearance_m: float = 0.34

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("slalom.enabled must be boolean")
        if isinstance(self.pole_count, bool) or not isinstance(self.pole_count, Integral) or self.pole_count not in (4, 5):
            raise ValueError("slalom.pole_count must be 4 or 5")
        for name in (
            "spacing_m", "pole_radius_m", "pole_height_m", "pole_mass_kg",
            "amplitude_m", "speed_m_s", "first_pole_x_m", "lead_m",
            "segment_duration_s", "tip_angle_deg", "tip_hold_s",
            "max_pole_displacement_m", "pass_clearance_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"slalom.{name} must be finite and positive")
        for name in ("pole_friction", "settle_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0.0:
                raise ValueError(f"slalom.{name} must be finite and non-negative")
        if self.first_pole_x_m < self.spacing_m:
            raise ValueError("slalom.first_pole_x_m must be at least spacing_m")
        if self.tip_angle_deg >= 90.0:
            raise ValueError("slalom.tip_angle_deg must be less than 90 degrees")
        if not self.pole_radius_m < self.pass_clearance_m < self.amplitude_m:
            raise ValueError("slalom.pass_clearance_m must exceed pole_radius_m and be less than amplitude_m")
        if 2.0 * self.pole_radius_m >= self.spacing_m:
            raise ValueError("slalom poles must not overlap")


def _pole_centers(config: SlalomConfig) -> np.ndarray:
    return np.column_stack((
        config.first_pole_x_m + np.arange(config.pole_count) * config.spacing_m,
        np.full(config.pole_count, config.amplitude_m),
    ))


def _finish_x(config: SlalomConfig) -> float:
    return config.first_pole_x_m + config.pole_count * config.spacing_m + config.lead_m


class SlalomCommandProgram:
    """A finite table of held body twists; neither pose nor episode resets drive it.

    Each half-wave has tangent angle ``sign * phi * sin(pi * s / L)``.
    The root for phi fixes lateral/forward displacement, and a small offline
    speed fit closes the discretized arcs without lateral commands or feedback.
    The displayed segment index clamps at the last hold after the route ends.
    """

    commands: np.ndarray
    reference_points: np.ndarray
    pole_centers: np.ndarray
    ctrl_dt: float
    segment_steps: int
    segment_duration_s: float
    total_control_steps: int
    duration_s: float
    finish_x_m: float

    @classmethod
    def from_config(
        cls, config: SlalomConfig, *, ctrl_dt: float, prediction_horizon: int = 1,
    ) -> SlalomCommandProgram:
        if not isinstance(config, SlalomConfig):
            raise TypeError("slalom config must be SlalomConfig")
        if isinstance(ctrl_dt, bool) or not np.isfinite(ctrl_dt) or ctrl_dt <= 0.0:
            raise ValueError("slalom ctrl_dt must be finite and positive")
        if (isinstance(prediction_horizon, bool) or not isinstance(prediction_horizon, Integral)
                or prediction_horizon < 1):
            raise ValueError("slalom prediction_horizon must be a positive integer")
        segment_steps = max(1, int(np.floor(config.segment_duration_s / ctrl_dt + 0.5)))
        if segment_steps < prediction_horizon:
            raise ValueError("slalom segment hold is shorter than prediction_horizon")
        segment_dt = segment_steps * float(ctrl_dt)

        def integrals(phi: float) -> tuple[float, float]:
            cosine = quad(lambda u: np.cos(phi * np.sin(np.pi * u)), 0.0, 1.0)[0]
            sine = quad(lambda u: np.sin(phi * np.sin(np.pi * u)), 0.0, 1.0)[0]
            return cosine, sine

        ratio = 2.0 * config.amplitude_m / config.spacing_m

        def residual(phi: float) -> float:
            cosine, sine = integrals(phi)
            return sine / cosine - ratio

        upper = np.nextafter(np.pi / 2.0, 0.0)
        if residual(upper) <= 0.0:
            raise ValueError("slalom amplitude/spacing requires a tangent angle of at least pi/2")
        phi = brentq(residual, 0.0, upper, xtol=1e-14)
        cosine, _ = integrals(phi)
        length = config.spacing_m / cosine
        count = max(8, int(np.floor(length / (config.speed_m_s * segment_dt) + 0.5)))
        u = np.linspace(0.0, 1.0, count + 1)
        heading = phi * np.sin(np.pi * u)
        heading[[0, -1]] = 0.0
        dense_u = np.linspace(0.0, 1.0, max(8193, count * 32 + 1))
        dense_heading = phi * np.sin(np.pi * dense_u)
        dense_points = length * np.column_stack((
            cumulative_trapezoid(np.cos(dense_heading), dense_u, initial=0.0),
            cumulative_trapezoid(np.sin(dense_heading), dense_u, initial=0.0),
        ))
        points = np.column_stack((np.interp(u, dense_u, dense_points[:, 0]),
                                  np.interp(u, dense_u, dense_points[:, 1])))
        turn = np.diff(heading)
        middle = (heading[:-1] + heading[1:]) / 2.0
        factor = segment_dt * np.sinc(turn / (2.0 * np.pi))
        displacement = factor[:, None] * np.column_stack((np.cos(middle), np.sin(middle)))
        speed = np.linalg.norm(np.diff(points, axis=0), axis=1) / factor
        target = np.array([config.spacing_m, 2.0 * config.amplitude_m])
        speed += displacement @ np.linalg.solve(
            displacement.T @ displacement, target - displacement.T @ speed,
        )
        if not np.isfinite(speed).all() or np.any(speed <= 0.0):
            raise ValueError("slalom arcs are under-resolved; reduce segment_duration_s")
        half_wave = np.column_stack((speed, np.zeros(count), turn / segment_dt))
        parts = []

        def append_straight(distance: float) -> None:
            if distance == 0.0:
                return
            holds = max(1, int(np.floor(distance / (config.speed_m_s * segment_dt) + 0.5)))
            straight = np.zeros((holds, 3))
            straight[:, 0] = distance / (holds * segment_dt)
            parts.append(straight)

        append_straight(config.first_pole_x_m - config.spacing_m)
        for wave in range(config.pole_count + 1):
            command = half_wave.copy()
            command[:, 2] *= 1.0 if wave % 2 == 0 else -1.0
            parts.append(command)
        append_straight(config.lead_m)
        commands = np.concatenate(parts)
        turns = commands[:, 2] * segment_dt
        headings = np.concatenate(([0.0], np.cumsum(turns)))
        middle_headings = (headings[:-1] + headings[1:]) / 2.0
        distances = commands[:, 0] * segment_dt * np.sinc(turns / (2.0 * np.pi))
        reference = np.vstack((np.zeros(2), np.cumsum(
            distances[:, None] * np.column_stack((np.cos(middle_headings), np.sin(middle_headings))),
            axis=0,
        )))
        program = cls()
        program.commands = commands
        program.reference_points = reference
        program.pole_centers = _pole_centers(config)
        program.ctrl_dt = float(ctrl_dt)
        program.segment_steps = segment_steps
        program.segment_duration_s = segment_dt
        program.total_control_steps = len(commands) * segment_steps
        program.duration_s = program.total_control_steps * program.ctrl_dt
        program.finish_x_m = _finish_x(config)
        for array in (program.commands, program.reference_points, program.pole_centers):
            array.setflags(write=False)
        return program

    def segment_index_at_control_step(self, control_step: int) -> int:
        if isinstance(control_step, bool) or not isinstance(control_step, Integral) or control_step < 0:
            raise ValueError("slalom control_step must be a non-negative integer")
        return min(int(control_step) // self.segment_steps, len(self.commands) - 1)

    def command_at_control_step(self, control_step: int) -> np.ndarray:
        index = self.segment_index_at_control_step(control_step)
        return np.zeros(3) if control_step >= self.total_control_steps else self.commands[index].copy()

    def time_at_control_step(self, control_step: int) -> float:
        self.segment_index_at_control_step(control_step)
        return int(control_step) * self.ctrl_dt

    def validate_environment(self, env_cfg: Any, *, keyboard: bool) -> None:
        from unilab.visualization.demo_commands import _validate_demo_commands

        _validate_demo_commands(np.vstack((self.commands, np.zeros((1, 3)))), env_cfg, keyboard=keyboard)


class SlalomEvaluator:
    """Score one robot from actual gate crossings and physical pole observations.

    Inputs are one environment's arrays: XY base, [P,3] pole positions,
    [P,4] wxyz orientations, and [P] current robot/pole contact booleans.
    Failure latches; contacts alone do not fail a trial. The evaluator never
    influences commands, and the command clock alone cannot establish success.
    """

    def __init__(self, config: SlalomConfig, pole_centers: np.ndarray):
        if not isinstance(config, SlalomConfig):
            raise TypeError("slalom config must be SlalomConfig")
        centers = np.asarray(pole_centers, dtype=np.float64)
        if centers.shape != (config.pole_count, 2) or not np.isfinite(centers).all():
            raise ValueError("slalom pole_centers must be finite [pole_count,2]")
        if not np.allclose(centers, _pole_centers(config), rtol=0.0, atol=1e-9):
            raise ValueError("slalom pole_centers must match the configured course")
        self.config = config
        self.pole_centers = centers.copy()
        self._previous_base = np.zeros(2)
        self._base = np.zeros(2)
        self._passed = 0
        self._failure_reasons: list[str] = []
        self._tip_elapsed = np.zeros(config.pole_count)
        self._fallen = np.zeros(config.pole_count, dtype=bool)
        self._displaced = np.zeros(config.pole_count, dtype=bool)
        self._contact = np.zeros(config.pole_count, dtype=bool)
        self._contact_samples = np.zeros(config.pole_count, dtype=np.int64)
        self._route_finished = False
        self._settle_elapsed = 0.0

    def _fail(self, reason: str) -> None:
        if reason not in self._failure_reasons:
            self._failure_reasons.append(reason)

    def update(
        self, base_xy: np.ndarray, pole_positions: np.ndarray,
        pole_quats: np.ndarray, contacts: np.ndarray, robot_done: bool,
        dt: float, route_finished: bool,
    ) -> None:
        count = self.config.pole_count
        base = np.asarray(base_xy, dtype=np.float64)
        positions = np.asarray(pole_positions, dtype=np.float64)
        quats = np.asarray(pole_quats, dtype=np.float64)
        contact = np.asarray(contacts)
        if base.shape != (2,) or not np.isfinite(base).all():
            raise ValueError("slalom base_xy must be finite [2]")
        if positions.shape != (count, 3) or not np.isfinite(positions).all():
            raise ValueError("slalom pole_positions must be finite [pole_count,3]")
        if quats.shape != (count, 4) or not np.isfinite(quats).all():
            raise ValueError("slalom pole_quats must be finite [pole_count,4] wxyz")
        norms = np.sum(quats * quats, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("slalom pole_quats must have nonzero norm")
        if contact.shape != (count,) or contact.dtype.kind != "b":
            raise ValueError("slalom contacts must be boolean [pole_count]")
        if isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("slalom dt must be finite and positive")
        if not isinstance(robot_done, (bool, np.bool_)) or not isinstance(route_finished, (bool, np.bool_)):
            raise ValueError("slalom robot_done and route_finished must be boolean")

        self._contact |= contact
        self._contact_samples += contact
        z_up = np.clip(1.0 - 2.0 * (quats[:, 1] ** 2 + quats[:, 2] ** 2) / norms, -1.0, 1.0)
        tipped = z_up < np.cos(np.deg2rad(self.config.tip_angle_deg))
        self._tip_elapsed = np.where(tipped, self._tip_elapsed + dt, 0.0)
        self._fallen |= self._tip_elapsed + 1e-12 >= self.config.tip_hold_s
        self._displaced |= np.linalg.norm(positions[:, :2] - self.pole_centers, axis=1) > self.config.max_pole_displacement_m
        for index in np.flatnonzero(self._fallen):
            self._fail(f"pole_{index + 1}_fallen")
        for index in np.flatnonzero(self._displaced):
            self._fail(f"pole_{index + 1}_displaced")
        if robot_done:
            self._fail("robot_done")

        previous = self._previous_base
        crossed = np.flatnonzero((previous[0] < self.pole_centers[:, 0])
                                 & (base[0] >= self.pole_centers[:, 0]))
        if len(crossed) > 1:
            self._fail("skipped_gate")
        for index in crossed:
            if index != self._passed:
                self._fail("out_of_order_gate")
                continue
            fraction = (self.pole_centers[index, 0] - previous[0]) / (base[0] - previous[0])
            crossing_y = previous[1] + fraction * (base[1] - previous[1])
            side = 1.0 if index % 2 == 0 else -1.0
            clearance = side * (crossing_y - self.pole_centers[index, 1])
            if clearance + 1e-12 < self.config.pass_clearance_m:
                self._fail(f"pole_{index + 1}_wrong_side_or_clearance")
            else:
                self._passed += 1
        self._base = base.copy()
        self._previous_base = base.copy()
        self._route_finished |= bool(route_finished)
        ready = (self._route_finished and self._passed == count
                 and self._base[0] >= _finish_x(self.config)
                 and not self._failure_reasons)
        self._settle_elapsed = self._settle_elapsed + float(dt) if ready else 0.0

    def report(self) -> dict[str, Any]:
        success = (not self._failure_reasons and self._route_finished
                   and self._passed == self.config.pole_count
                   and self._base[0] >= _finish_x(self.config)
                   and self._settle_elapsed + 1e-12 >= self.config.settle_s)
        return {
            "success": bool(success),
            "failed": bool(self._failure_reasons),
            "failure_reasons": list(self._failure_reasons),
            "passed_poles": int(self._passed),
            "pole_count": int(self.config.pole_count),
            "route_finished": self._route_finished,
            "settle_elapsed_s": self._settle_elapsed,
            "pole_fallen": self._fallen.tolist(),
            "pole_displaced": self._displaced.tolist(),
            "pole_contact": self._contact.tolist(),
            "contact_samples": self._contact_samples.tolist(),
            "finish_x_m": _finish_x(self.config),
        }
