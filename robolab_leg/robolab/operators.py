"""Линейные операторы: интерфейс (ABC) и три матричные реализации.

LinearOperator (ABC) -> MatrixOperator -> Projection, MassMatrix
(SO3 наследует MatrixOperator в transforms.py).
Модуль не импортирует MuJoCo и другие модули robolab.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

TOL = 1e-9


class LinearOperator(ABC):
    """Контракт: A : R^n -> R^m, A(ax + by) = aAx + bAy."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """(m, n): размерность образа и области определения."""

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """Копия матрицы оператора; её изменение не портит объект."""

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def adjoint(self) -> "LinearOperator": ...

    @abstractmethod
    def passport(self) -> dict: ...

    def __call__(self, x):
        return self.apply(x)

    def is_linear(self, trials: int = 5, tol: float = 1e-9, seed: int = 0) -> bool:
        """Численная проверка линейности на случайных входах (не доказательство)."""
        rng = np.random.default_rng(seed)
        m, n = self.shape
        for _ in range(trials):
            u, v = rng.normal(size=n), rng.normal(size=n)
            a, b = rng.normal(size=2)
            if not np.allclose(self.apply(a * u + b * v),
                               a * self.apply(u) + b * self.apply(v), rtol=0, atol=tol):
                return False
        return True


class MatrixOperator(LinearOperator):
    """Оператор, заданный матрицей. Хранит собственную неизменяемую копию."""

    INVARIANT = "не задан (общий оператор)"

    def __init__(self, A):
        A = np.array(A, dtype=float, copy=True)
        if A.ndim != 2 or not np.isfinite(A).all():
            raise ValueError("A: двумерный массив из конечных чисел")
        A.setflags(write=False)
        self._A = A

    @property
    def shape(self):
        return self._A.shape

    @property
    def matrix(self):
        return self._A.copy()

    def apply(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self._A.shape[1]:
            raise ValueError(f"ожидалась размерность {self._A.shape[1]}, получено {x.shape[0]}")
        return self._A @ x

    def adjoint(self):
        return MatrixOperator(self._A.T)

    def passport(self) -> dict:
        """Поля паспорта: пространство, линейность, инвариант, ядро/образ, сопряжённый, спектр."""
        m, n = self.shape
        sv = np.linalg.svd(self._A, compute_uv=False)
        tol = max(m, n) * np.finfo(float).eps * (sv.max() if sv.size else 1.0)
        rank = int((sv > tol).sum())
        return {
            "space": f"R^{n} -> R^{m}",
            "linear": self.is_linear(),
            "invariant": self.INVARIANT,
            "kernel_dim": n - rank,
            "image_dim": rank,
            "adjoint": "A^T",
            "singular_values": sv.tolist(),
            "invertible": bool(m == n and rank == n),
        }


class Projection(MatrixOperator):
    """Ортогональный проектор: P^2 = P, P = P^T. Не поворот: теряет компоненту."""

    INVARIANT = "P^2 = P, P = P^T; rank = dim образа"

    def __init__(self, P, tol: float = TOL):
        super().__init__(P)
        A = self._A
        if A.shape[0] != A.shape[1]:
            raise ValueError("проектор должен быть квадратным")
        if not np.allclose(A @ A, A, rtol=0, atol=tol):
            raise ValueError("P^2 != P: не проектор")
        if not np.allclose(A, A.T, rtol=0, atol=tol):
            raise ValueError("P != P^T: проектор не ортогональный")

    @classmethod
    def from_normal(cls, n) -> "Projection":
        """P = n n^T для единичной нормали (нормируем явно, нулевую отвергаем)."""
        n = np.asarray(n, dtype=float).reshape(-1)
        nn = np.linalg.norm(n)
        if not np.isfinite(nn) or nn < 1e-12:
            raise ValueError("нормаль нулевая")
        n = n / nn
        return cls(np.outer(n, n))

    def split(self, v):
        """v = (компонента в образе) + (остаток); для силы: нормальная и касательная."""
        v = np.asarray(v, dtype=float)
        par = self.apply(v)
        return par, v - par

    def adjoint(self):
        return self  # P^T = P


class MassMatrix(MatrixOperator):
    """Матрица инерции M(q): симметрична и положительно определена."""

    INVARIANT = "M = M^T, M > 0 (кинетическая энергия T = 1/2 dq^T M dq > 0)"

    def __init__(self, M, tol: float = TOL):
        super().__init__(M)
        A = self._A
        if A.shape[0] != A.shape[1]:
            raise ValueError("M должна быть квадратной")
        if not np.allclose(A, A.T, rtol=0, atol=tol):
            raise ValueError("M не симметрична")
        if np.linalg.eigvalsh((A + A.T) / 2).min() <= 0:
            raise ValueError("M не положительно определена")

    def kinetic_energy(self, dq) -> float:
        dq = np.asarray(dq, dtype=float)
        return float(0.5 * dq @ self._A @ dq)

    def inverse(self) -> "MassMatrix":
        return MassMatrix(np.linalg.inv(self._A))

    def adjoint(self):
        return self
