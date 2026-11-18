"""Конфигурации: левая нога 4R антропоморфного робота (только данные)."""
from __future__ import annotations

from dataclasses import dataclass

from .kinematics import Chain, Joint
from .transforms import Point3


@dataclass(frozen=True)
class Body:
    """Точечная масса звена, заданная в системе этого звена."""
    frame: str
    mass: float        # кг
    com: tuple         # м, в системе звена


@dataclass(frozen=True)
class LegModel:
    chain: Chain
    bodies: tuple
    markers: dict      # имя -> Point3 в системе звена
    g: float = 9.81    # м/с^2, ось z мира направлена вверх


def leg4r(thigh=0.40, shank=0.40, hip_y=0.10) -> LegModel:
    """Оси: x вперёд, y влево, z вверх; в нуле нога вытянута вниз.

    q = [hip_roll (x), hip_pitch (y), knee (y), ankle (y)].
    Положительный тангаж (поворот вокруг y) уводит стопу НАЗАД,
    поэтому положительное колено — сгибание.
    """
    chain = Chain("pelvis", [
        Joint("hip_roll",  (0.0, hip_y, 0.0), (1, 0, 0)),
        Joint("hip_pitch", (0.0, 0.0, 0.0),   (0, 1, 0)),
        Joint("knee",      (0.0, 0.0, -thigh), (0, 1, 0)),
        Joint("ankle",     (0.0, 0.0, -shank), (0, 1, 0)),
    ])
    bodies = (
        Body("hip_roll",  0.5, (0.0, 0.0, -0.05)),
        Body("hip_pitch", 4.0, (0.0, 0.0, -thigh / 2)),
        Body("knee",      2.0, (0.0, 0.0, -shank / 2)),
        Body("ankle",     0.8, (0.04, 0.0, -0.04)),
    )
    markers = {
        "toe":  Point3((0.12, 0.0, -0.08), "ankle"),
        "heel": Point3((-0.06, 0.0, -0.08), "ankle"),
    }
    return LegModel(chain, bodies, markers)
