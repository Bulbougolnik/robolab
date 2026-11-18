"""Общий обход цепи: прямая кинематика и якобиан точки. Единственный FK в библиотеке."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .transforms import SE3, SO3, Point3, Vector3


@dataclass(frozen=True)
class Joint:
    """Вращательный сустав: фиксированный сдвиг от родителя, затем поворот на q вокруг axis.

    name  — имя системы ПОСЛЕ сустава (система звена).
    p0/R0 — положение и ориентация предсуставной системы в системе родителя.
    axis  — ось вращения (единичная) в системе звена.
    """
    name: str
    p0: tuple
    axis: tuple
    R0: SO3 = field(default_factory=SO3.identity)
    limits: tuple | None = None

    def __post_init__(self):
        a = np.asarray(self.axis, float)
        if a.shape != (3,) or abs(np.linalg.norm(a) - 1.0) > 1e-9:
            raise ValueError(f"{self.name}: ось должна быть единичной")


class Chain:
    """Последовательная цепь base -> joint_1 -> ... -> joint_n."""

    def __init__(self, base: str, joints: list[Joint]):
        names = [base] + [j.name for j in joints]
        if len(set(names)) != len(names):
            raise ValueError("имена систем должны быть уникальны")
        self.base, self.joints = base, list(joints)

    @property
    def n(self) -> int:
        return len(self.joints)

    def _q(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape != (self.n,) or not np.isfinite(q).all():
            raise ValueError(f"q: {self.n} конечных чисел (рад)")
        return q

    def frames(self, q) -> dict[str, SE3]:
        """{имя: ^base T_имя} для всех звеньев; каждое умножение проверяет имена."""
        q = self._q(q)
        T = SE3.identity(self.base)
        out = {self.base: T}
        prev = self.base
        for j, qi in zip(self.joints, q):
            pre = f"{j.name}@pre"
            origin = SE3(j.R0, j.p0, prev, pre)
            rot = SE3(SO3.exp(np.asarray(j.axis) * qi), np.zeros(3), pre, j.name)
            T = T @ origin @ rot
            out[j.name] = T
            prev = j.name
        return out

    def fk(self, q, frame: str | None = None) -> SE3:
        fr = self.frames(q)
        return fr[frame or self.joints[-1].name]

    def point(self, q, p_local: Point3) -> Point3:
        """Точка из системы звена -> система base."""
        return self.fk(q, p_local.frame).apply_point(p_local)

    def point_jacobian(self, q, p_local: Point3) -> np.ndarray:
        """J (3 x n): d(положение точки в base)/dq. Столбец j = z_j x (p - o_j)."""
        fr = self.frames(q)
        target = p_local.frame
        p = fr[target].apply_point(p_local).xyz
        J = np.zeros((3, self.n))
        for k, j in enumerate(self.joints):
            Tj = fr[j.name]
            z = Tj.R.apply(np.asarray(j.axis))
            J[:, k] = np.cross(z, p - Tj.p)
            if j.name == target:
                break
        return J
