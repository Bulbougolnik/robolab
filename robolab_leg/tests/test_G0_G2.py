"""Протоколы G0–G2 и проверка ноги 4R. Запуск: python -m pytest -q"""
import numpy as np
import pytest

from robolab import (SO3, SE3, Point3, Vector3, Projection, MassMatrix, MatrixOperator,
                     LinearOperator, exp_so3, log_so3, leg4r, dynamics)

A = dict(rtol=0, atol=1e-9)


# ---------- G0 ----------
def test_G0_rotation_90_maps_ex_to_ey():
    assert np.allclose(SO3.Rz(np.pi / 2).apply([1, 0, 0]), [0, 1, 0], **A)


def test_G0_reflection_and_stretch_rejected():
    with pytest.raises(ValueError):
        SO3(np.diag([1, 1, -1.0]))          # сохраняет длины, но отражает
    with pytest.raises(ValueError):
        SO3(np.diag([2, 1, 0.5]))           # det = 1, но растягивает


def test_G0_projection_passport():
    P = Projection.from_normal([0, 0, 2.0])
    assert np.allclose(P.matrix @ P.matrix, P.matrix, **A)
    assert P.passport()["kernel_dim"] == 2
    with pytest.raises(ValueError):
        Projection(SO3.Rz(0.3).matrix)      # поворот — не проектор


def test_G0_exp_log_special_angles():
    n = np.array([1, 2, 2.0]) / 3
    for th in (0, 1e-12, 0.3, np.pi - 1e-8, np.pi):
        R = exp_so3(th * n)
        assert np.allclose(exp_so3(log_so3(R)), R, rtol=0, atol=4e-8)


def test_G0_matrix_copy_is_protected():
    R = SO3.Rz(0.4)
    R.matrix[0, 0] = 99.0
    assert R.matrix[0, 0] != 99.0


def test_G0_hierarchy():
    assert issubclass(SO3, MatrixOperator) and issubclass(MatrixOperator, LinearOperator)
    assert issubclass(Projection, MatrixOperator) and issubclass(MassMatrix, MatrixOperator)
    assert not issubclass(SE3, LinearOperator)      # SE3 содержит SO3, не наследует
    assert not issubclass(Projection, SO3)
    with pytest.raises(TypeError):
        LinearOperator()                            # ABC


# ---------- G1 ----------
def test_G1_point_vs_vector():
    T = SE3(SO3.Rz(np.pi / 2), [1, 2, 0], "W", "B")
    assert np.allclose(T.apply_point(Point3([1, 0, 0], "B")).xyz, [1, 3, 0], **A)
    assert np.allclose(T.apply_vector(Vector3([1, 0, 0], "B")).xyz, [0, 1, 0], **A)


def test_G1_frame_and_type_errors():
    T = SE3(SO3.identity(), [0, 0, 0], "W", "B")
    with pytest.raises(ValueError):
        T.apply_point(Point3([0, 0, 0], "C"))
    with pytest.raises(TypeError):
        T.apply_point(Vector3([0, 0, 0], "B"))


# ---------- G2 ----------
def test_G2_composition_uses_parent_rotation():
    T1 = SE3(SO3.Rz(np.pi / 2), [1, 0, 0], "W", "B")
    T2 = SE3(SO3.identity(), [0.4, 0, 0], "B", "S")
    assert np.allclose((T1 @ T2).p, [1, 0.4, 0], **A)
    # другой порядок даёт другой результат (некоммутативность)
    Ta = SE3(SO3.Rz(np.pi / 2), [0, 0, 0], "W", "B") @ SE3(SO3.identity(), [0.4, 0, 0], "B", "S")
    Tb = SE3(SO3.identity(), [0.4, 0, 0], "W", "B") @ SE3(SO3.Rz(np.pi / 2), [0, 0, 0], "B", "S")
    assert np.allclose(Ta.p, [0, 0.4, 0], **A) and np.allclose(Tb.p, [0.4, 0, 0], **A)


def test_G2_inverse_and_mismatch():
    T = SE3(SO3.exp([0.3, -0.2, 0.5]), [0.1, -0.4, 0.7], "W", "B")
    I = T @ T.inverse()
    assert np.allclose(I.homogeneous(), np.eye(4), **A) and (I.dst, I.src) == ("W", "W")
    with pytest.raises(ValueError):
        T @ T                                       # src='B' != dst='W'


# ---------- Нога 4R ----------
L1 = L2 = 0.40
q0 = np.array([0.0, -0.3, 0.6, -0.3])


def test_leg_fk_matches_planar_formula():
    leg = leg4r()
    q = np.array([0.0, 0.35, 0.8, -0.2])
    ankle = leg.chain.fk(q).p
    assert np.allclose(ankle, [-L1 * np.sin(q[1]) - L2 * np.sin(q[1] + q[2]), 0.10,
                               -L1 * np.cos(q[1]) - L2 * np.cos(q[1] + q[2])], **A)


def test_leg_zero_pose_foot_below_pelvis():
    assert np.allclose(leg4r().chain.fk(np.zeros(4)).p, [0, 0.10, -0.80], **A)


def test_leg_jacobian_vs_finite_differences():
    leg, eps = leg4r(), 1e-6
    p = leg.markers["toe"]
    q = np.array([0.2, -0.3, 0.7, -0.4])
    J = leg.chain.point_jacobian(q, p)
    for i in range(4):
        d = np.zeros(4); d[i] = eps
        num = (leg.chain.point(q + d, p).xyz - leg.chain.point(q - d, p).xyz) / (2 * eps)
        assert np.allclose(J[:, i], num, rtol=0, atol=1e-7)


def test_leg_mass_matrix_spd_and_energy_independent_check():
    leg, eps = leg4r(), 1e-6
    q, dq = np.array([0.1, -0.4, 0.9, -0.5]), np.array([0.3, -0.2, 0.5, 0.1])
    M = dynamics.mass_matrix(leg, q)
    assert np.allclose(M.matrix, M.matrix.T, **A)
    # независимо: скорости центров масс конечными разностями, T = sum 1/2 m v^2
    T = 0.0
    for b in leg.bodies:
        from robolab import Point3 as P
        c = P(b.com, b.frame)
        v = (leg.chain.point(q + eps * dq, c).xyz - leg.chain.point(q - eps * dq, c).xyz) / (2 * eps)
        T += 0.5 * b.mass * v @ v
    assert np.isclose(M.kinetic_energy(dq), T, rtol=1e-6)


def test_leg_gravity_is_gradient_of_potential():
    leg, eps = leg4r(), 1e-6
    q = np.array([0.2, -0.3, 0.7, -0.4])
    g = dynamics.gravity_vector(leg, q)
    for i in range(4):
        d = np.zeros(4); d[i] = eps
        num = (dynamics.potential_energy(leg, q + d) - dynamics.potential_energy(leg, q - d)) / (2 * eps)
        assert np.isclose(g[i], num, rtol=0, atol=1e-6)


def test_leg_ground_force_projection():
    """Реакция опоры: нормальная и касательная составляющие через Projection."""
    P = Projection.from_normal([0, 0, 1])
    normal, tangent = P.split([30.0, -5.0, 400.0])
    assert np.allclose(normal, [0, 0, 400], **A) and np.allclose(tangent, [30, -5, 0], **A)
