"""Точки, векторы, повороты SO(3) и позы SE(3) с именованными системами координат.

SO3 — линейный оператор (наследует MatrixOperator).
SE3 — НЕ линейный оператор на R^3: содержит SO3 и перенос (композиция).
Единицы: метры и радианы; правые ортонормированные системы.
"""
from __future__ import annotations

import numpy as np

from .operators import MatrixOperator, TOL


# ---------- функции ----------
def skew(w) -> np.ndarray:
    """[w]_x : [w]_x v = w x v."""
    w = np.asarray(w, dtype=float).reshape(3)
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def vee(S) -> np.ndarray:
    """Обратная к skew операция."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def exp_so3(w) -> np.ndarray:
    """Формула Родрига для вектора поворота w = theta*n; ряд при малом угле."""
    w = np.asarray(w, dtype=float).reshape(3)
    th = np.linalg.norm(w)
    K = skew(w)
    if th < 1e-4:
        A, B = 1.0 - th**2 / 6.0, 0.5 - th**2 / 24.0
    else:
        A, B = np.sin(th) / th, (1.0 - np.cos(th)) / th**2
    return np.eye(3) + A * K + B * K @ K


def log_so3(R) -> np.ndarray:
    """Главный вектор поворота, theta in [0, pi]. При theta = pi знак оси неоднозначен."""
    R = np.asarray(R, dtype=float)
    a = vee(R - R.T)                       # = 2 sin(theta) n
    s, c = np.linalg.norm(a) / 2.0, (np.trace(R) - 1.0) / 2.0
    th = np.arctan2(s, c)                  # точен и у 0, и у pi
    if th < 1e-8:
        return a / 2.0
    if np.pi - th < 1e-2:                  # ось из симметричной части: n n^T
        N = ((R + R.T) / 2.0 - c * np.eye(3)) / (1.0 - c)
        i = int(np.argmax(np.diag(N)))
        n = N[:, i] / np.sqrt(N[i, i])
        if a @ n < 0:
            n = -n
        return th * n
    return th / (2.0 * np.sin(th)) * a


def project_so3(M) -> np.ndarray:
    """Явное исправление шумной матрицы: ближайший поворот (SVD)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


# ---------- типы ----------
class SO3(MatrixOperator):
    """R^T R = I, det R = 1. Конструктор проверяет оба обещания."""

    INVARIANT = "R^T R = I, det R = 1 (сохраняет скалярное произведение и ориентацию)"

    def __init__(self, R):
        R = np.asarray(R, dtype=float)
        if R.shape != (3, 3) or not np.isfinite(R).all():
            raise ValueError("R: форма 3x3 и конечные числа")
        if not np.allclose(R.T @ R, np.eye(3), rtol=0, atol=TOL):
            raise ValueError("R не ортогональна")
        if not np.isclose(np.linalg.det(R), 1.0, rtol=0, atol=TOL):
            raise ValueError("R меняет ориентацию")
        super().__init__(R)

    # фабрики
    @classmethod
    def identity(cls):
        return cls(np.eye(3))

    @classmethod
    def exp(cls, w):
        return cls(exp_so3(w))

    @classmethod
    def Rx(cls, a): return cls.exp([a, 0, 0])

    @classmethod
    def Ry(cls, a): return cls.exp([0, a, 0])

    @classmethod
    def Rz(cls, a): return cls.exp([0, 0, a])

    # операции
    def log(self) -> np.ndarray:
        return log_so3(self._A)

    def compose(self, other: "SO3") -> "SO3":
        if not isinstance(other, SO3):
            raise TypeError("Нужна SO3")
        return SO3(self._A @ other._A)

    def inverse(self) -> "SO3":
        return SO3(self._A.T)

    def adjoint(self) -> "SO3":
        return self.inverse()          # R^T = R^-1


class Point3:
    """Место в пространстве: поворот И перенос. Координаты в метрах."""
    __slots__ = ("xyz", "frame")

    def __init__(self, xyz, frame: str):
        xyz = np.array(xyz, dtype=float).reshape(-1)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("xyz: три конечных числа")
        xyz.setflags(write=False)
        self.xyz, self.frame = xyz, frame

    def __repr__(self): return f"Point3({self.xyz.tolist()}, '{self.frame}')"


class Vector3:
    """Смещение/направление: только поворот, без переноса."""
    __slots__ = ("xyz", "frame")

    def __init__(self, xyz, frame: str):
        xyz = np.array(xyz, dtype=float).reshape(-1)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("xyz: три конечных числа")
        xyz.setflags(write=False)
        self.xyz, self.frame = xyz, frame

    def __repr__(self): return f"Vector3({self.xyz.tolist()}, '{self.frame}')"


class SE3:
    """Поза ^dst T_src = (R, p): координаты из системы src переводит в dst.

    Содержит SO3 (композиция), не наследует LinearOperator на R^3.
    """

    def __init__(self, R, p, dst: str, src: str):
        self.R = R if isinstance(R, SO3) else SO3(R)
        p = np.array(p, dtype=float).reshape(-1)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("p: три конечных числа")
        p.setflags(write=False)
        self.p, self.dst, self.src = p, dst, src

    @classmethod
    def identity(cls, frame: str):
        return cls(SO3.identity(), np.zeros(3), frame, frame)

    def _check(self, frame: str):
        if frame != self.src:
            raise ValueError(f"Система '{frame}' не совпадает с src='{self.src}'")

    def apply_point(self, p: Point3) -> Point3:
        if not isinstance(p, Point3):
            raise TypeError("Нужна Point3")
        self._check(p.frame)
        return Point3(self.R.apply(p.xyz) + self.p, self.dst)

    def apply_vector(self, v: Vector3) -> Vector3:
        if not isinstance(v, Vector3):
            raise TypeError("Нужна Vector3")
        self._check(v.frame)
        return Vector3(self.R.apply(v.xyz), self.dst)

    def __matmul__(self, other):
        if not isinstance(other, SE3):
            return NotImplemented
        if self.src != other.dst:
            raise ValueError("Несогласованные промежуточные системы")
        return SE3(self.R.compose(other.R), self.R.apply(other.p) + self.p,
                   self.dst, other.src)

    def inverse(self) -> "SE3":
        Rt = self.R.inverse()
        return SE3(Rt, -Rt.apply(self.p), self.src, self.dst)

    def homogeneous(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R.matrix, self.p
        return T

    def __repr__(self): return f"SE3(dst='{self.dst}', src='{self.src}', p={self.p.tolist()})"
