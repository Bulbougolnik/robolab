"""Энергия, M(q), g(q) — только через общий Chain (своего FK здесь нет).

Модель: точечные массы звеньев (вращательные инерции не учитываются).
"""
from __future__ import annotations

import numpy as np

from .models import LegModel
from .operators import MassMatrix
from .transforms import Point3


def com_world(model: LegModel, q):
    return [model.chain.point(q, Point3(b.com, b.frame)).xyz for b in model.bodies]


def mass_matrix(model: LegModel, q) -> MassMatrix:
    """M(q) = sum_i m_i J_i^T J_i."""
    M = np.zeros((model.chain.n,) * 2)
    for b in model.bodies:
        J = model.chain.point_jacobian(q, Point3(b.com, b.frame))
        M += b.mass * J.T @ J
    return MassMatrix(M)


def kinetic_energy(model: LegModel, q, dq) -> float:
    return mass_matrix(model, q).kinetic_energy(dq)


def potential_energy(model: LegModel, q) -> float:
    return float(sum(b.mass * model.g * c[2] for b, c in zip(model.bodies, com_world(model, q))))


def gravity_vector(model: LegModel, q) -> np.ndarray:
    """g(q) = dU/dq = sum_i m_i g J_i[z, :]."""
    g = np.zeros(model.chain.n)
    for b in model.bodies:
        g += b.mass * model.g * model.chain.point_jacobian(q, Point3(b.com, b.frame))[2]
    return g
